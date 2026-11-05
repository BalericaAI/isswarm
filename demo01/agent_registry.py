"""
Guardian Telemetry Swarm
agent_registry.py

Registry and baseline-resolution layer for monitored Glean agents.

Primary questions:

    "Is this agent monitored?"
    "Who owns it?"
    "Which behavioral baseline applies to this run?"
    "Which baseline version was active when this run occurred?"

This module performs NO:

- telemetry retrieval
- security analysis
- CrewAI execution
- LangGraph execution
- risk scoring
- enforcement

Architecture:

Glean WORKFLOW_RUN
      |
      | workflow_id / application_id
      v
AgentRegistry
      |
      +--> registered?
      +--> enabled?
      +--> owner?
      +--> risk tier?
      +--> baseline version?
      |
      v
AgentBaseline
      |
      v
Guardian Telemetry Swarm
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from models import AgentBaseline


# ---------------------------------------------------------------------------
# Risk tier
# ---------------------------------------------------------------------------


class AgentRiskTier(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Registry models
# ---------------------------------------------------------------------------


class RegistryBaseModel(BaseModel):
    """
    Strict registry model.

    Registry mistakes should fail loudly rather than silently creating
    unexpected monitoring behavior.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class BaselineVersion(RegistryBaseModel):
    """
    One behavioral-baseline version.

    Multiple versions may exist for the same monitored agent.

    Example:

        v1.0
            no MCP

        v2.0
            approved Jira MCP

    Historical runs can therefore be evaluated against the baseline
    that was active when that run occurred.
    """

    version: str

    file: str

    effective_from: datetime | None = None

    effective_until: datetime | None = None

    enabled: bool = True

    notes: str | None = None


class MonitoredAgentRegistration(RegistryBaseModel):
    """
    Registry entry for one monitored Glean agent.
    """

    agent_id: str

    agent_name: str

    enabled: bool = True

    owner: str

    risk_tier: AgentRiskTier = AgentRiskTier.MODERATE

    # Glean identifiers.
    #
    # Some environments may use one or both depending on how Customer
    # Event telemetry is exported.
    workflow_ids: list[str] = Field(
        default_factory=list
    )

    application_ids: list[str] = Field(
        default_factory=list
    )

    namespaces: list[str] = Field(
        default_factory=list
    )

    baselines: list[BaselineVersion]

    description: str | None = None

    tags: list[str] = Field(
        default_factory=list
    )


class RegistryDocument(RegistryBaseModel):
    """
    Root structure of registry/monitored_agents.json.
    """

    registry_version: str = "1.0"

    agents: list[MonitoredAgentRegistration]


class ResolvedAgent(RegistryBaseModel):
    """
    Result returned after resolving live Glean telemetry against the
    monitoring registry.
    """

    registration: MonitoredAgentRegistration

    baseline_version: BaselineVersion

    baseline: AgentBaseline


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentRegistryError(RuntimeError):
    """Base registry exception."""


class RegistryConfigurationError(AgentRegistryError):
    """Invalid or missing registry configuration."""


class AgentNotRegisteredError(AgentRegistryError):
    """Requested Glean agent is not registered."""


class AgentMonitoringDisabledError(AgentRegistryError):
    """Agent exists in the registry but monitoring is disabled."""


class BaselineNotFoundError(AgentRegistryError):
    """No applicable baseline can be resolved."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """
    Resolve monitored Glean agents and their behavioral baselines.

    Typical usage:

        registry = AgentRegistry.from_environment()

        resolved = registry.resolve(
            workflow_id="workflow-123",
            run_timestamp="2026-11-05T12:30:00Z",
        )

        baseline = resolved.baseline
    """

    def __init__(
        self,
        registry_file: str,
    ) -> None:

        self.registry_file = Path(
            registry_file
        )

        self.registry_root = (
            self.registry_file.parent
        )

        self.document = self._load_registry()

        self._build_indexes()


    # ------------------------------------------------------------------
    # Environment constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(
        cls,
    ) -> "AgentRegistry":

        registry_file = os.getenv(
            "AGENT_REGISTRY_FILE",
            "registry/monitored_agents.json",
        )

        return cls(
            registry_file=registry_file
        )


    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _load_registry(
        self,
    ) -> RegistryDocument:

        if not self.registry_file.exists():

            raise RegistryConfigurationError(
                "Agent registry file not found: "
                f"{self.registry_file}"
            )


        try:

            with self.registry_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                raw = json.load(
                    file
                )

        except json.JSONDecodeError as exc:

            raise RegistryConfigurationError(
                "Agent registry contains invalid JSON: "
                f"{self.registry_file}"
            ) from exc


        try:

            document = RegistryDocument.model_validate(
                raw
            )

        except ValidationError as exc:

            raise RegistryConfigurationError(
                "Agent registry failed Pydantic validation."
            ) from exc


        if not document.agents:

            raise RegistryConfigurationError(
                "Agent registry contains no agents."
            )


        return document


    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    def _build_indexes(
        self,
    ) -> None:
        """
        Build deterministic lookup indexes.

        Duplicate identifiers are rejected because ambiguous ownership
        would be dangerous in a security-monitoring system.
        """

        self._agent_id_index: dict[
            str,
            MonitoredAgentRegistration
        ] = {}

        self._workflow_index: dict[
            str,
            MonitoredAgentRegistration
        ] = {}

        self._application_index: dict[
            str,
            MonitoredAgentRegistration
        ] = {}


        for registration in self.document.agents:

            # ------------------------------------------------------
            # Agent ID
            # ------------------------------------------------------

            agent_key = _normalize_identifier(
                registration.agent_id
            )

            if agent_key in self._agent_id_index:

                raise RegistryConfigurationError(
                    "Duplicate agent_id in registry: "
                    f"{registration.agent_id}"
                )

            self._agent_id_index[
                agent_key
            ] = registration


            # ------------------------------------------------------
            # Workflow IDs
            # ------------------------------------------------------

            for workflow_id in (
                registration.workflow_ids
            ):

                key = _normalize_identifier(
                    workflow_id
                )

                if key in self._workflow_index:

                    other = self._workflow_index[
                        key
                    ]

                    raise RegistryConfigurationError(
                        "workflow_id is registered to multiple agents: "
                        f"{workflow_id} -> "
                        f"{other.agent_name}, "
                        f"{registration.agent_name}"
                    )

                self._workflow_index[
                    key
                ] = registration


            # ------------------------------------------------------
            # Application IDs
            # ------------------------------------------------------

            for application_id in (
                registration.application_ids
            ):

                key = _normalize_identifier(
                    application_id
                )

                if key in self._application_index:

                    other = self._application_index[
                        key
                    ]

                    raise RegistryConfigurationError(
                        "application_id is registered to multiple agents: "
                        f"{application_id} -> "
                        f"{other.agent_name}, "
                        f"{registration.agent_name}"
                    )

                self._application_index[
                    key
                ] = registration


    # ------------------------------------------------------------------
    # Public resolution API
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        workflow_id: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
        run_timestamp: str | datetime | None = None,
    ) -> ResolvedAgent:
        """
        Resolve a live Glean run to:

            MonitoredAgentRegistration
            BaselineVersion
            AgentBaseline

        Identifier precedence:

            workflow_id
                ->
            application_id
                ->
            agent_id

        A run timestamp may be supplied to resolve the baseline that was
        active when the event occurred.
        """

        registration = self.find_registration(
            workflow_id=workflow_id,
            application_id=application_id,
            agent_id=agent_id,
        )


        if registration is None:

            raise AgentNotRegisteredError(
                self._build_not_registered_message(
                    workflow_id=workflow_id,
                    application_id=application_id,
                    agent_id=agent_id,
                )
            )


        if not registration.enabled:

            raise AgentMonitoringDisabledError(
                f"Monitoring is disabled for agent: "
                f"{registration.agent_name}"
            )


        timestamp = _parse_datetime(
            run_timestamp
        )


        baseline_version = (
            self.resolve_baseline_version(
                registration=registration,
                run_timestamp=timestamp,
            )
        )


        baseline = self.load_baseline(
            baseline_version
        )


        # ----------------------------------------------------------
        # Defensive identity check
        # ----------------------------------------------------------

        if (
            baseline.agent_id
            and baseline.agent_id
            != registration.agent_id
        ):

            raise RegistryConfigurationError(
                "Baseline agent_id does not match registry agent_id: "
                f"{baseline.agent_id} != "
                f"{registration.agent_id}"
            )


        if (
            baseline.agent_name
            != registration.agent_name
        ):

            raise RegistryConfigurationError(
                "Baseline agent_name does not match registry agent_name: "
                f"{baseline.agent_name} != "
                f"{registration.agent_name}"
            )


        return ResolvedAgent(
            registration=registration,
            baseline_version=baseline_version,
            baseline=baseline,
        )


    # ------------------------------------------------------------------
    # Registration lookup
    # ------------------------------------------------------------------

    def find_registration(
        self,
        *,
        workflow_id: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
    ) -> MonitoredAgentRegistration | None:
        """
        Find an agent without raising if it is unknown.

        Useful for live monitoring:

            registration = registry.find_registration(...)

            if registration is None:
                generate UNREGISTERED_AGENT finding
        """

        if workflow_id:

            registration = self._workflow_index.get(
                _normalize_identifier(
                    workflow_id
                )
            )

            if registration:
                return registration


        if application_id:

            registration = self._application_index.get(
                _normalize_identifier(
                    application_id
                )
            )

            if registration:
                return registration


        if agent_id:

            registration = self._agent_id_index.get(
                _normalize_identifier(
                    agent_id
                )
            )

            if registration:
                return registration


        return None


    # ------------------------------------------------------------------
    # Baseline version resolution
    # ------------------------------------------------------------------

    def resolve_baseline_version(
        self,
        registration: MonitoredAgentRegistration,
        run_timestamp: datetime | None = None,
    ) -> BaselineVersion:
        """
        Resolve the applicable behavioral baseline.

        If run_timestamp is supplied:

            choose baseline where

                effective_from <= run_timestamp

            and

                run_timestamp < effective_until

            when effective_until exists.

        If no timestamp is supplied:

            choose the most recent enabled baseline.
        """

        enabled_versions = [
            version
            for version in registration.baselines
            if version.enabled
        ]


        if not enabled_versions:

            raise BaselineNotFoundError(
                f"No enabled baseline exists for "
                f"{registration.agent_name}"
            )


        # ----------------------------------------------------------
        # Historical lookup
        # ----------------------------------------------------------

        if run_timestamp is not None:

            applicable = [
                version
                for version in enabled_versions
                if _baseline_applies_at(
                    version,
                    run_timestamp,
                )
            ]


            if not applicable:

                raise BaselineNotFoundError(
                    "No baseline applies to run timestamp "
                    f"{run_timestamp.isoformat()} for "
                    f"{registration.agent_name}"
                )


            return max(
                applicable,
                key=_baseline_sort_key,
            )


        # ----------------------------------------------------------
        # Current baseline
        # ----------------------------------------------------------

        return max(
            enabled_versions,
            key=_baseline_sort_key,
        )


    # ------------------------------------------------------------------
    # Baseline loading
    # ------------------------------------------------------------------

    def load_baseline(
        self,
        version: BaselineVersion,
    ) -> AgentBaseline:
        """
        Load and validate one baseline JSON file.
        """

        path = Path(
            version.file
        )


        # Registry baseline paths are relative to the repository root
        # conventionally, but first allow an absolute path.
        if not path.is_absolute():

            # Try current working directory first.
            cwd_path = Path.cwd() / path

            if cwd_path.exists():
                path = cwd_path

            else:
                # Then resolve relative to registry folder.
                path = (
                    self.registry_root
                    / path
                )


        if not path.exists():

            raise BaselineNotFoundError(
                f"Baseline file not found: "
                f"{path}"
            )


        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                raw = json.load(
                    file
                )

        except json.JSONDecodeError as exc:

            raise BaselineNotFoundError(
                f"Baseline contains invalid JSON: "
                f"{path}"
            ) from exc


        try:

            return AgentBaseline.model_validate(
                raw
            )

        except ValidationError as exc:

            raise BaselineNotFoundError(
                f"Baseline failed Pydantic validation: "
                f"{path}"
            ) from exc


    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def is_monitored(
        self,
        *,
        workflow_id: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """
        True only when the agent is registered AND monitoring is enabled.
        """

        registration = self.find_registration(
            workflow_id=workflow_id,
            application_id=application_id,
            agent_id=agent_id,
        )

        return bool(
            registration
            and registration.enabled
        )


    def list_monitored_agents(
        self,
    ) -> list[MonitoredAgentRegistration]:
        """
        Return enabled monitored agents.
        """

        return [
            registration
            for registration in self.document.agents
            if registration.enabled
        ]


    def list_all_agents(
        self,
    ) -> list[MonitoredAgentRegistration]:
        """
        Return all registry entries.
        """

        return list(
            self.document.agents
        )


    def reload(
        self,
    ) -> None:
        """
        Reload registry from disk.

        Useful for a long-running monitor so agents can be onboarded
        without restarting the process.
        """

        self.document = self._load_registry()

        self._build_indexes()


    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def validate_all_baselines(
        self,
    ) -> dict[str, list[str]]:
        """
        Validate every enabled baseline in the registry.

        Returns:

            {
                "Agent Name": [
                    "v1.0: OK",
                    "v2.0: ERROR ..."
                ]
            }

        Useful during startup or CI/CD.
        """

        results: dict[
            str,
            list[str]
        ] = {}


        for registration in self.document.agents:

            agent_results: list[str] = []


            for version in registration.baselines:

                if not version.enabled:

                    agent_results.append(
                        f"{version.version}: DISABLED"
                    )

                    continue


                try:

                    self.load_baseline(
                        version
                    )

                    agent_results.append(
                        f"{version.version}: OK"
                    )

                except Exception as exc:

                    agent_results.append(
                        f"{version.version}: ERROR - {exc}"
                    )


            results[
                registration.agent_name
            ] = agent_results


        return results


    # ------------------------------------------------------------------
    # Error message helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_not_registered_message(
        *,
        workflow_id: str | None,
        application_id: str | None,
        agent_id: str | None,
    ) -> str:

        identifiers = []


        if workflow_id:
            identifiers.append(
                f"workflow_id={workflow_id}"
            )


        if application_id:
            identifiers.append(
                f"application_id={application_id}"
            )


        if agent_id:
            identifiers.append(
                f"agent_id={agent_id}"
            )


        if not identifiers:
            return (
                "Agent could not be resolved because no "
                "identifier was supplied."
            )


        return (
            "No monitored agent registration matches: "
            + ", ".join(
                identifiers
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_identifier(
    value: str,
) -> str:
    """
    Normalize lookup identifiers.

    Glean IDs themselves are treated as opaque values.

    Only surrounding whitespace and case are normalized.
    """

    return value.strip().lower()


def _parse_datetime(
    value: str | datetime | None,
) -> datetime | None:
    """
    Parse ISO-8601 run timestamp.

    Naive timestamps are interpreted as UTC for deterministic behavior.
    """

    if value is None:
        return None


    if isinstance(
        value,
        datetime,
    ):

        parsed = value

    else:

        text = value.strip()

        if not text:
            return None


        if text.endswith(
            "Z"
        ):
            text = (
                text[:-1]
                + "+00:00"
            )


        parsed = datetime.fromisoformat(
            text
        )


    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )


    return parsed.astimezone(
        timezone.utc
    )


def _baseline_applies_at(
    version: BaselineVersion,
    timestamp: datetime,
) -> bool:
    """
    Determine whether one baseline version applies at a timestamp.
    """

    start = version.effective_from

    end = version.effective_until


    if start is not None:

        start = _ensure_utc(
            start
        )

        if timestamp < start:
            return False


    if end is not None:

        end = _ensure_utc(
            end
        )

        if timestamp >= end:
            return False


    return True


def _baseline_sort_key(
    version: BaselineVersion,
) -> tuple[datetime, str]:
    """
    Sort baseline versions primarily by effective date.

    Version string is only a deterministic secondary key because semantic
    version strings should not be assumed to sort numerically as strings.
    """

    effective = (
        _ensure_utc(
            version.effective_from
        )
        if version.effective_from
        else datetime.min.replace(
            tzinfo=timezone.utc
        )
    )


    return (
        effective,
        version.version,
    )


def _ensure_utc(
    value: datetime,
) -> datetime:

    if value.tzinfo is None:

        return value.replace(
            tzinfo=timezone.utc
        )


    return value.astimezone(
        timezone.utc
    )
