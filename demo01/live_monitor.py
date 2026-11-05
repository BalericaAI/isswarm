"""
Guardian Telemetry Swarm
live_monitor.py

Continuous live-monitoring coordinator for Glean agent runs.

Responsibilities:

1. Discover new Glean WORKFLOW_RUN events in Splunk.
2. Filter to actual Glean Agent executions.
3. Wait briefly for related telemetry to settle.
4. Prevent duplicate analysis with a SQLite run ledger.
5. Resolve the executing agent through AgentRegistry.
6. Load the correct versioned behavioral baseline.
7. Retrieve complete run telemetry.
8. Normalize telemetry.
9. Execute the Guardian LangGraph once.
10. Persist the Guardian result.
11. Record processing status.

This module performs NO specialist security reasoning.

Architecture:

Glean
  |
  v
Customer Event Logs
  |
  v
Splunk
  |
  v
SplunkRunDiscovery
  |
  v
LiveMonitor
  |
  +--> AgentRegistry
  |
  +--> TelemetryClient
  |
  +--> TelemetryNormalizer
  |
  +--> Guardian LangGraph
  |
  v
GuardianGraphResult
  |
  +--> output/*.json
  |
  +--> SQLite processing ledger


Important:

This is Demo #2.

Guardian observes.

Guardian does NOT perform enforcement.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict

from agent_registry import (
    AgentMonitoringDisabledError,
    AgentNotRegisteredError,
    AgentRegistry,
)
from graph_state import GuardianGraphInput
from guardian_graph import (
    build_guardian_graph,
    run_guardian_graph,
)
from telemetry_client import (
    TelemetryClient,
    TelemetryClientError,
)
from telemetry_normalizer import TelemetryNormalizer


# ---------------------------------------------------------------------------
# Live-monitor models
# ---------------------------------------------------------------------------


class LiveModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class DiscoveredWorkflowRun(LiveModel):
    """
    Minimal information needed to decide whether Guardian should process
    a newly observed Glean WorkflowRun.
    """

    run_id: str

    workflow_id: str | None = None

    application_id: str | None = None

    root_workflow_id: str | None = None

    top_agent_workflow_id: str | None = None

    feature: str | None = None

    initiator: str | None = None

    platform: str | None = None

    namespace: str | None = None

    workflow_created_at: datetime | None = None

    splunk_event_time: datetime | None = None


class LedgerStatus(StrEnum):

    PROCESSING = "PROCESSING"

    PROCESSED = "PROCESSED"

    UNREGISTERED = "UNREGISTERED"

    MONITORING_DISABLED = "MONITORING_DISABLED"

    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Splunk workflow-run discovery
# ---------------------------------------------------------------------------


class SplunkRunDiscovery:
    """
    Discover recent Glean WORKFLOW_RUN events from Splunk.

    Full telemetry retrieval remains the responsibility of
    telemetry_client.py.

    This class only answers:

        "What new Glean Agent runs exist?"
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        index: str,
        verify_tls: bool = True,
        timeout_seconds: int = 30,
        earliest_time: str = "-15m",
        latest_time: str = "now",
    ) -> None:

        self.base_url = base_url.rstrip("/")

        self.token = token

        self.index = index

        self.verify_tls = verify_tls

        self.timeout_seconds = timeout_seconds

        self.earliest_time = earliest_time

        self.latest_time = latest_time


    @classmethod
    def from_environment(
        cls,
    ) -> "SplunkRunDiscovery":

        base_url = os.getenv(
            "SPLUNK_BASE_URL"
        )

        token = os.getenv(
            "SPLUNK_TOKEN"
        )

        index = os.getenv(
            "SPLUNK_GLEAN_INDEX"
        )


        missing = []

        if not base_url:
            missing.append(
                "SPLUNK_BASE_URL"
            )

        if not token:
            missing.append(
                "SPLUNK_TOKEN"
            )

        if not index:
            missing.append(
                "SPLUNK_GLEAN_INDEX"
            )


        if missing:

            raise RuntimeError(
                "Missing Splunk configuration: "
                + ", ".join(
                    missing
                )
            )


        return cls(
            base_url=base_url,

            token=token,

            index=index,

            verify_tls=_env_bool(
                "SPLUNK_VERIFY_TLS",
                True,
            ),

            timeout_seconds=_env_int(
                "SPLUNK_TIMEOUT_SECONDS",
                30,
            ),

            earliest_time=os.getenv(
                "LIVE_DISCOVERY_EARLIEST_TIME",
                "-15m",
            ),

            latest_time=os.getenv(
                "LIVE_DISCOVERY_LATEST_TIME",
                "now",
            ),
        )


    def discover(
        self,
    ) -> list[DiscoveredWorkflowRun]:
        """
        Discover recent Glean Agent WorkflowRun events.

        We intentionally filter:

            Type = WORKFLOW_RUN
            Feature = AGENT

        so ordinary AI Answers and other workflow-backed Glean features
        do not automatically become Guardian-monitored applications.
        """

        search = (
            f'search index="{self.index}" '
            'Type="WORKFLOW_RUN" '
            'WorkflowRun.Feature="AGENT" '
            "| sort 0 _time"
        )


        results = self._search(
            search
        )


        discovered: dict[
            str,
            DiscoveredWorkflowRun
        ] = {}


        for result in results:

            candidate = (
                self._parse_workflow_run(
                    result
                )
            )


            if candidate is None:
                continue


            # RunId is globally unique for the top-level invocation.
            discovered[
                candidate.run_id
            ] = candidate


        return list(
            discovered.values()
        )


    # ------------------------------------------------------------------
    # Splunk REST
    # ------------------------------------------------------------------

    def _search(
        self,
        search: str,
    ) -> list[dict[str, Any]]:

        endpoint = (
            f"{self.base_url}"
            "/services/search/v2/jobs/export"
        )


        headers = {
            "Authorization": (
                f"Bearer {self.token}"
            ),
            "Accept": "application/json",
        }


        data = {
            "search": search,
            "output_mode": "json",
            "earliest_time": (
                self.earliest_time
            ),
            "latest_time": (
                self.latest_time
            ),
        }


        response = requests.post(
            endpoint,
            headers=headers,
            data=data,
            timeout=self.timeout_seconds,
            verify=self.verify_tls,
            stream=True,
        )


        response.raise_for_status()


        results: list[
            dict[str, Any]
        ] = []


        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue


            try:

                item = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue


            if isinstance(
                item,
                dict,
            ):

                results.append(
                    item
                )


        return results


    # ------------------------------------------------------------------
    # WorkflowRun extraction
    # ------------------------------------------------------------------

    def _parse_workflow_run(
        self,
        result: dict[str, Any],
    ) -> DiscoveredWorkflowRun | None:

        payload = result.get(
            "result",
            result,
        )


        if not isinstance(
            payload,
            dict,
        ):
            return None


        raw = payload.get(
            "_raw"
        )


        if isinstance(
            raw,
            str,
        ):

            try:

                event = json.loads(
                    raw
                )

            except json.JSONDecodeError:

                return None

        elif isinstance(
            raw,
            dict,
        ):

            event = raw

        else:

            event = payload


        json_payload = event.get(
            "jsonPayload",
            event,
        )


        if not isinstance(
            json_payload,
            dict,
        ):
            return None


        event_type = (
            json_payload.get(
                "Type"
            )
            or json_payload.get(
                "type"
            )
        )


        if str(
            event_type
        ).upper() != "WORKFLOW_RUN":

            return None


        workflow = (
            json_payload.get(
                "WorkflowRun"
            )
            or json_payload.get(
                "workflowRun"
            )
            or json_payload.get(
                "workflow_run"
            )
        )


        if not isinstance(
            workflow,
            dict,
        ):
            return None


        run_id = _get_case_insensitive(
            workflow,
            "RunId",
            "runId",
            "run_id",
        )


        if not run_id:
            return None


        feature = _get_case_insensitive(
            workflow,
            "Feature",
            "feature",
        )


        # Defensive second check.
        if (
            feature
            and str(
                feature
            ).upper()
            != "AGENT"
        ):

            return None


        workflow_created_at = _parse_datetime(
            _get_case_insensitive(
                workflow,
                "WorkflowCreatedAt",
                "workflowCreatedAt",
                "workflow_created_at",
            )
        )


        splunk_event_time = _parse_datetime(
            payload.get(
                "_time"
            )
        )


        return DiscoveredWorkflowRun(
            run_id=str(
                run_id
            ),

            workflow_id=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "WorkflowId",
                    "workflowId",
                    "workflow_id",
                )
            ),

            application_id=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "ApplicationId",
                    "applicationId",
                    "application_id",
                )
            ),

            root_workflow_id=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "RootWorkflowId",
                    "rootWorkflowId",
                    "root_workflow_id",
                )
            ),

            top_agent_workflow_id=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "TopAgentWorkflowId",
                    "topAgentWorkflowId",
                    "top_agent_workflow_id",
                )
            ),

            feature=_optional_string(
                feature
            ),

            initiator=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "Initiator",
                    "initiator",
                )
            ),

            platform=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "Platform",
                    "platform",
                )
            ),

            namespace=_optional_string(
                _get_case_insensitive(
                    workflow,
                    "Namespace",
                    "namespace",
                )
            ),

            workflow_created_at=(
                workflow_created_at
            ),

            splunk_event_time=(
                splunk_event_time
            ),
        )


# ---------------------------------------------------------------------------
# SQLite processed-run ledger
# ---------------------------------------------------------------------------


class RunLedger:
    """
    Minimal persistent ledger preventing duplicate Guardian processing.

    SQLite is sufficient for this PoC.

    Production could replace it with:

        DynamoDB
        PostgreSQL
        Redis
        Splunk KV Store
        another durable state service
    """

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = Path(
            path
        )


        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        self._initialize()


    def _connect(
        self,
    ) -> sqlite3.Connection:

        connection = sqlite3.connect(
            self.path
        )


        connection.row_factory = (
            sqlite3.Row
        )


        return connection


    def _initialize(
        self,
    ) -> None:

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    application_id TEXT,
                    agent_name TEXT,
                    processing_status TEXT NOT NULL,
                    guardian_status TEXT,
                    severity TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    first_seen_at TEXT NOT NULL,
                    processed_at TEXT
                )
                """
            )


            connection.commit()


    def should_process(
        self,
        run_id: str,
        max_attempts: int,
    ) -> bool:

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT
                    processing_status,
                    attempts
                FROM processed_runs
                WHERE run_id = ?
                """,
                (
                    run_id,
                ),
            ).fetchone()


        if row is None:
            return True


        status = row[
            "processing_status"
        ]

        attempts = int(
            row[
                "attempts"
            ]
        )


        # Successful and intentionally handled runs are terminal.
        if status in {
            LedgerStatus.PROCESSED.value,
            LedgerStatus.UNREGISTERED.value,
            LedgerStatus.MONITORING_DISABLED.value,
        }:

            return False


        # Errors can retry.
        if (
            status
            == LedgerStatus.ERROR.value
            and attempts < max_attempts
        ):

            return True


        return False


    def mark_processing(
        self,
        run: DiscoveredWorkflowRun,
        agent_name: str | None,
    ) -> None:

        now = _utc_now_iso()


        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO processed_runs (
                    run_id,
                    workflow_id,
                    application_id,
                    agent_name,
                    processing_status,
                    attempts,
                    first_seen_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)

                ON CONFLICT(run_id)
                DO UPDATE SET
                    processing_status = excluded.processing_status,
                    attempts = processed_runs.attempts + 1,
                    agent_name = excluded.agent_name
                """,
                (
                    run.run_id,
                    run.workflow_id,
                    run.application_id,
                    agent_name,
                    LedgerStatus.PROCESSING.value,
                    now,
                ),
            )


            connection.commit()


    def mark_processed(
        self,
        run_id: str,
        agent_name: str,
        guardian_status: str,
        severity: str,
    ) -> None:

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE processed_runs
                SET
                    agent_name = ?,
                    processing_status = ?,
                    guardian_status = ?,
                    severity = ?,
                    last_error = NULL,
                    processed_at = ?
                WHERE run_id = ?
                """,
                (
                    agent_name,
                    LedgerStatus.PROCESSED.value,
                    guardian_status,
                    severity,
                    _utc_now_iso(),
                    run_id,
                ),
            )


            connection.commit()


    def mark_terminal(
        self,
        run: DiscoveredWorkflowRun,
        status: LedgerStatus,
        agent_name: str | None = None,
    ) -> None:

        self.mark_processing(
            run,
            agent_name,
        )


        with self._connect() as connection:

            connection.execute(
                """
                UPDATE processed_runs
                SET
                    processing_status = ?,
                    processed_at = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    _utc_now_iso(),
                    run.run_id,
                ),
            )


            connection.commit()


    def mark_error(
        self,
        run: DiscoveredWorkflowRun,
        error: str,
    ) -> None:

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT run_id
                FROM processed_runs
                WHERE run_id = ?
                """,
                (
                    run.run_id,
                ),
            ).fetchone()


            if row is None:

                connection.execute(
                    """
                    INSERT INTO processed_runs (
                        run_id,
                        workflow_id,
                        application_id,
                        processing_status,
                        attempts,
                        last_error,
                        first_seen_at
                    )
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        run.run_id,
                        run.workflow_id,
                        run.application_id,
                        LedgerStatus.ERROR.value,
                        error,
                        _utc_now_iso(),
                    ),
                )

            else:

                connection.execute(
                    """
                    UPDATE processed_runs
                    SET
                        processing_status = ?,
                        last_error = ?,
                        processed_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        LedgerStatus.ERROR.value,
                        error,
                        _utc_now_iso(),
                        run.run_id,
                    ),
                )


            connection.commit()


# ---------------------------------------------------------------------------
# Live Monitor
# ---------------------------------------------------------------------------


class LiveMonitor:
    """
    Coordinates live Guardian processing of Glean Agent runs.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        discovery: SplunkRunDiscovery,
        telemetry_client: TelemetryClient,
        ledger: RunLedger,
        output_dir: str = "output/live",
        settle_seconds: int = 30,
        max_attempts: int = 3,
    ) -> None:

        self.registry = registry

        self.discovery = discovery

        self.telemetry_client = (
            telemetry_client
        )

        self.ledger = ledger

        self.output_dir = Path(
            output_dir
        )


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.settle_seconds = (
            settle_seconds
        )

        self.max_attempts = (
            max_attempts
        )


        # Compile the LangGraph once and reuse it.
        self.graph = (
            build_guardian_graph()
        )


    # ------------------------------------------------------------------
    # One polling cycle
    # ------------------------------------------------------------------

    def poll_once(
        self,
    ) -> int:
        """
        Discover and process currently available runs.

        Returns number of Guardian analyses completed.
        """

        # Allow registry updates without service restart.
        self.registry.reload()


        runs = self.discovery.discover()


        runs.sort(
            key=lambda run: (
                run.workflow_created_at
                or datetime.min.replace(
                    tzinfo=timezone.utc
                )
            )
        )


        completed = 0


        for run in runs:

            if not self.ledger.should_process(
                run.run_id,
                self.max_attempts,
            ):

                continue


            if not self._telemetry_has_settled(
                run
            ):

                continue


            if self.process_run(
                run
            ):

                completed += 1


        return completed


    # ------------------------------------------------------------------
    # Individual run
    # ------------------------------------------------------------------

    def process_run(
        self,
        run: DiscoveredWorkflowRun,
    ) -> bool:
        """
        Process one Glean Agent run.

        Returns True only when a Guardian graph analysis completed.
        """

        print(
            f"[DISCOVERED] {run.run_id}"
        )


        # ----------------------------------------------------------
        # Resolve monitored agent
        # ----------------------------------------------------------

        registration = (
            self.registry.find_registration(
                workflow_id=(
                    run.workflow_id
                ),
                application_id=(
                    run.application_id
                ),
            )
        )


        if registration is None:

            print(
                "  [REVIEW] Unregistered Glean agent."
            )


            self._write_unregistered_event(
                run
            )


            self.ledger.mark_terminal(
                run=run,
                status=(
                    LedgerStatus.UNREGISTERED
                ),
            )


            return False


        if not registration.enabled:

            print(
                "  [SKIP] Monitoring disabled for "
                f"{registration.agent_name}"
            )


            self.ledger.mark_terminal(
                run=run,
                status=(
                    LedgerStatus.MONITORING_DISABLED
                ),
                agent_name=(
                    registration.agent_name
                ),
            )


            return False


        self.ledger.mark_processing(
            run=run,
            agent_name=(
                registration.agent_name
            ),
        )


        try:

            # ------------------------------------------------------
            # Resolve correct historical baseline
            # ------------------------------------------------------

            resolved = self.registry.resolve(
                workflow_id=(
                    run.workflow_id
                ),
                application_id=(
                    run.application_id
                ),
                run_timestamp=(
                    run.workflow_created_at
                ),
            )


            baseline = (
                resolved.baseline
            )


            print(
                f"  Agent      : "
                f"{registration.agent_name}"
            )

            print(
                f"  Risk Tier  : "
                f"{registration.risk_tier}"
            )

            print(
                f"  Baseline   : "
                f"{resolved.baseline_version.version}"
            )


            # ------------------------------------------------------
            # Retrieve complete evidence bundle
            # ------------------------------------------------------

            raw_bundle = (
                self.telemetry_client
                .get_run_telemetry(
                    run_id=run.run_id
                )
            )


            # ------------------------------------------------------
            # Normalize
            # ------------------------------------------------------

            normalizer = TelemetryNormalizer(
                baseline=baseline
            )


            telemetry = normalizer.normalize(
                workflow_run_event=(
                    raw_bundle.workflow_run_event
                ),

                action_events=(
                    raw_bundle.action_events
                ),

                mcp_events=(
                    raw_bundle.mcp_events
                ),

                protect_findings=(
                    raw_bundle.protect_findings
                ),

                final_output_text=(
                    raw_bundle.final_output_text
                ),
            )


            # ------------------------------------------------------
            # Retrieval warnings become explicit evidence gaps.
            # ------------------------------------------------------

            _propagate_retrieval_warnings(
                telemetry=telemetry,
                warnings=(
                    raw_bundle.retrieval_warnings
                ),
            )


            # ------------------------------------------------------
            # Guardian analysis
            #
            # NO streaming here.
            #
            # Each specialist should execute exactly once.
            # ------------------------------------------------------

            graph_input = GuardianGraphInput(
                baseline=baseline,
                telemetry=telemetry,
            )


            result = run_guardian_graph(
                graph_input=graph_input,
                graph=self.graph,
            )


            decision = (
                result.guardian_decision
            )


            if decision is None:

                raise RuntimeError(
                    "Guardian graph completed without "
                    "a GuardianDecision."
                )


            # ------------------------------------------------------
            # Persist result
            # ------------------------------------------------------

            output_path = (
                self._save_guardian_result(
                    result=result,
                    baseline_version=(
                        resolved.baseline_version.version
                    ),
                    risk_tier=(
                        registration.risk_tier.value
                    ),
                )
            )


            self.ledger.mark_processed(
                run_id=run.run_id,
                agent_name=(
                    registration.agent_name
                ),
                guardian_status=(
                    decision.guardian_status.value
                ),
                severity=(
                    decision.severity.value
                ),
            )


            print(
                f"  Guardian   : "
                f"{decision.guardian_status}"
            )

            print(
                f"  Severity   : "
                f"{decision.severity}"
            )

            print(
                f"  Output     : "
                f"{output_path}"
            )

            print()


            return True


        except (
            AgentNotRegisteredError,
            AgentMonitoringDisabledError,
            TelemetryClientError,
            Exception,
        ) as exc:

            # The broad Exception catch is deliberate at this service
            # boundary: one bad run must not terminate monitoring of
            # all subsequent enterprise agents.

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )


            print(
                f"  [ERROR] {error_text}"
            )


            self.ledger.mark_error(
                run=run,
                error=error_text,
            )


            return False


    # ------------------------------------------------------------------
    # Settling window
    # ------------------------------------------------------------------

    def _telemetry_has_settled(
        self,
        run: DiscoveredWorkflowRun,
    ) -> bool:
        """
        Give related telemetry time to arrive.

        WORKFLOW_RUN may appear before:

            ACTION
            MCP_USAGE
            Protect findings
            final-output telemetry

        Processing immediately can incorrectly convert ingestion latency
        into "missing evidence".

        For the PoC, use a small configurable settling period.
        """

        event_time = (
            run.splunk_event_time
            or run.workflow_created_at
        )


        if event_time is None:

            # If event timing is unavailable, process rather than
            # silently ignoring the run.
            return True


        age = (
            datetime.now(
                timezone.utc
            )
            - event_time
        ).total_seconds()


        return (
            age
            >= self.settle_seconds
        )


    # ------------------------------------------------------------------
    # Unregistered-agent event
    # ------------------------------------------------------------------

    def _write_unregistered_event(
        self,
        run: DiscoveredWorkflowRun,
    ) -> Path:
        """
        Record an unknown executing agent as a security-review event.

        We cannot run the normal behavioral swarm because no baseline
        exists.
        """

        payload = {
            "run_id": run.run_id,

            "workflow_id": (
                run.workflow_id
            ),

            "application_id": (
                run.application_id
            ),

            "guardian_status": "REVIEW",

            "severity": "MEDIUM",

            "reason_codes": [
                "UNREGISTERED_AGENT"
            ],

            "summary": (
                "A Glean agent executed but no Guardian "
                "behavioral baseline is registered for it."
            ),

            "recommended_actions": [
                "IDENTIFY_AGENT_OWNER",
                "REVIEW_AGENT_CONFIGURATION",
                "REGISTER_AGENT_BASELINE",
            ],

            "enforcement_performed": False,

            "observed_at": (
                _utc_now_iso()
            ),
        }


        path = (
            self.output_dir
            / (
                f"{_safe_filename(run.run_id)}"
                "_unregistered_agent.json"
            )
        )


        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )


        return path


    # ------------------------------------------------------------------
    # Guardian result persistence
    # ------------------------------------------------------------------

    def _save_guardian_result(
        self,
        result,
        baseline_version: str,
        risk_tier: str,
    ) -> Path:

        path = (
            self.output_dir
            / (
                f"{_safe_filename(result.run_id)}"
                "_guardian_result.json"
            )
        )


        payload = result.model_dump(
            mode="json",
            exclude_none=True,
        )


        # Add operational metadata outside the core graph contract.
        payload[
            "monitoring_metadata"
        ] = {
            "baseline_version": (
                baseline_version
            ),
            "agent_risk_tier": (
                risk_tier
            ),
            "processed_at": (
                _utc_now_iso()
            ),
            "monitor_mode": "LIVE",
        }


        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )


        return path


# ---------------------------------------------------------------------------
# Retrieval-warning propagation
# ---------------------------------------------------------------------------


def _propagate_retrieval_warnings(
    telemetry,
    warnings: list[str],
) -> None:
    """
    Convert retrieval-layer warnings into explicit evidence gaps.

    This fixes an important distinction:

        no Protect findings

    versus

        Protect telemetry was not retrieved.
    """

    missing = set(
        telemetry.missing_evidence
    )


    for warning in warnings:

        upper = warning.upper()


        if "MCP" in upper:

            missing.add(
                "MCP_TELEMETRY_UNAVAILABLE"
            )


        if "PROTECT" in upper:

            missing.add(
                "PROTECT_TELEMETRY_UNAVAILABLE"
            )


        if (
            "FINAL OUTPUT" in upper
            or "OUTPUT TEXT" in upper
        ):

            missing.add(
                "FINAL_OUTPUT_UNAVAILABLE"
            )


    telemetry.missing_evidence = sorted(
        missing
    )


    telemetry.evidence_complete = (
        len(
            telemetry.missing_evidence
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Service runner
# ---------------------------------------------------------------------------


def run_live_monitor() -> None:
    """
    Build the live-monitoring service from environment configuration.
    """

    registry = (
        AgentRegistry.from_environment()
    )


    discovery = (
        SplunkRunDiscovery.from_environment()
    )


    telemetry_client = (
        TelemetryClient.from_environment()
    )


    ledger = RunLedger(
        os.getenv(
            "LIVE_MONITOR_LEDGER",
            "state/guardian_runs.db",
        )
    )


    monitor = LiveMonitor(
        registry=registry,

        discovery=discovery,

        telemetry_client=(
            telemetry_client
        ),

        ledger=ledger,

        output_dir=os.getenv(
            "LIVE_OUTPUT_DIR",
            "output/live",
        ),

        settle_seconds=_env_int(
            "LIVE_SETTLE_SECONDS",
            30,
        ),

        max_attempts=_env_int(
            "LIVE_MAX_ATTEMPTS",
            3,
        ),
    )


    poll_seconds = _env_int(
        "LIVE_POLL_SECONDS",
        30,
    )


    run_once = _env_bool(
        "LIVE_RUN_ONCE",
        False,
    )


    print(
        "=" * 70
    )

    print(
        "GUARDIAN LIVE MONITOR"
    )

    print(
        "=" * 70
    )

    print(
        f"Registry : "
        f"{registry.registry_file}"
    )

    print(
        f"Poll     : "
        f"{poll_seconds}s"
    )

    print(
        f"Settle   : "
        f"{monitor.settle_seconds}s"
    )

    print()


    while True:

        try:

            count = monitor.poll_once()


            if count:

                print(
                    f"[MONITOR] Completed "
                    f"{count} Guardian analysis run(s)."
                )


        except KeyboardInterrupt:

            print()

            print(
                "Guardian live monitor stopped."
            )

            return


        except Exception as exc:

            # Service-level isolation:
            #
            # a temporary Splunk or registry problem must not permanently
            # terminate the monitoring process.
            print(
                "[MONITOR ERROR] "
                f"{type(exc).__name__}: {exc}"
            )


        if run_once:

            return


        time.sleep(
            poll_seconds
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _get_case_insensitive(
    mapping: dict[str, Any],
    *names: str,
) -> Any:

    for name in names:

        if name in mapping:

            return mapping[
                name
            ]


    lowered = {
        str(key).lower(): value
        for key, value in mapping.items()
    }


    for name in names:

        if name.lower() in lowered:

            return lowered[
                name.lower()
            ]


    return None


def _parse_datetime(
    value: Any,
) -> datetime | None:

    if value is None:

        return None


    if isinstance(
        value,
        datetime,
    ):

        parsed = value

    else:

        text = str(
            value
        ).strip()


        if not text:

            return None


        if text.endswith(
            "Z"
        ):

            text = (
                text[:-1]
                + "+00:00"
            )


        try:

            parsed = datetime.fromisoformat(
                text
            )

        except ValueError:

            return None


    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )


    return parsed.astimezone(
        timezone.utc
    )


def _optional_string(
    value: Any,
) -> str | None:

    if value is None:
        return None


    text = str(
        value
    ).strip()


    return (
        text
        or None
    )


def _safe_filename(
    value: str,
) -> str:

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_"
    )


    return "".join(
        character
        if character in allowed
        else "_"
        for character in value
    )


def _utc_now_iso() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _env_bool(
    name: str,
    default: bool = False,
) -> bool:

    value = os.getenv(
        name
    )


    if value is None:
        return default


    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_int(
    name: str,
    default: int,
) -> int:

    value = os.getenv(
        name
    )


    if value is None:
        return default


    try:

        return int(
            value
        )

    except ValueError:

        return default


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    run_live_monitor()
