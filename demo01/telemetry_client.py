"""
Guardian Telemetry Swarm
telemetry_client.py

Retrieves raw telemetry for one monitored Glean agent run.

Supported modes:

    mock
        Read telemetry from local JSON fixtures.

    splunk
        Retrieve Glean Customer Event telemetry from Splunk.

Future modes could include:

    s3
    bigquery
    athena
    glean_export

This file performs RETRIEVAL ONLY.

It does NOT:
- classify behavior
- compare against the baseline
- calculate security risk
- invoke CrewAI
- invoke LangGraph
- make security decisions

Pipeline:

Glean
  |
  v
Customer Event Logs
  |
  v
Splunk / Mock Fixture
  |
  v
telemetry_client.py
  |
  v
RawTelemetryBundle
  |
  v
telemetry_normalizer.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Raw telemetry bundle
# ---------------------------------------------------------------------------


@dataclass
class RawTelemetryBundle:
    """
    Raw evidence retrieved for a single monitored agent run.

    These objects have NOT yet been normalized.

    telemetry_normalizer.py converts this bundle into NormalizedTelemetry.
    """

    run_id: str

    workflow_run_event: dict[str, Any]

    action_events: list[dict[str, Any]] = field(
        default_factory=list
    )

    mcp_events: list[dict[str, Any]] = field(
        default_factory=list
    )

    protect_findings: list[dict[str, Any]] = field(
        default_factory=list
    )

    final_output_text: str | None = None

    source: str = "UNKNOWN"

    retrieval_warnings: list[str] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TelemetryClientError(RuntimeError):
    """Base exception for telemetry retrieval failures."""


class TelemetryNotFoundError(TelemetryClientError):
    """Raised when the requested Glean run cannot be located."""


class TelemetryConfigurationError(TelemetryClientError):
    """Raised when required telemetry configuration is missing."""


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class TelemetryClient:
    """
    Retrieve raw telemetry from mock files or Splunk.

    Recommended usage:

        client = TelemetryClient.from_environment()

        bundle = client.get_run_telemetry(
            run_id="run-12345"
        )
    """

    SUPPORTED_MODES = {
        "mock",
        "splunk",
    }

    def __init__(
        self,
        mode: str = "mock",
        mock_file: str | None = None,
        splunk_base_url: str | None = None,
        splunk_token: str | None = None,
        splunk_index: str | None = None,
        splunk_verify_tls: bool = True,
        splunk_timeout_seconds: int = 30,
        splunk_earliest_time: str = "-24h",
        splunk_latest_time: str = "now",
    ) -> None:

        self.mode = mode.strip().lower()

        if self.mode not in self.SUPPORTED_MODES:
            raise TelemetryConfigurationError(
                f"Unsupported telemetry mode: {self.mode}"
            )

        self.mock_file = mock_file

        self.splunk_base_url = (
            splunk_base_url.rstrip("/")
            if splunk_base_url
            else None
        )

        self.splunk_token = splunk_token

        self.splunk_index = splunk_index

        self.splunk_verify_tls = splunk_verify_tls

        self.splunk_timeout_seconds = splunk_timeout_seconds

        self.splunk_earliest_time = splunk_earliest_time

        self.splunk_latest_time = splunk_latest_time

    # ------------------------------------------------------------------
    # Environment constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> "TelemetryClient":
        """
        Construct the client using environment variables.

        Example:

            TELEMETRY_MODE=mock
            MOCK_TELEMETRY_FILE=mock_telemetry/normal_run.json

        or:

            TELEMETRY_MODE=splunk
            SPLUNK_BASE_URL=https://splunk.example.com:8089
            SPLUNK_TOKEN=...
            SPLUNK_GLEAN_INDEX=glean
            SPLUNK_VERIFY_TLS=true
            SPLUNK_TIMEOUT_SECONDS=30
            SPLUNK_EARLIEST_TIME=-24h
            SPLUNK_LATEST_TIME=now
        """

        mode = os.getenv(
            "TELEMETRY_MODE",
            "mock",
        )

        return cls(
            mode=mode,

            mock_file=os.getenv(
                "MOCK_TELEMETRY_FILE"
            ),

            splunk_base_url=os.getenv(
                "SPLUNK_BASE_URL"
            ),

            splunk_token=os.getenv(
                "SPLUNK_TOKEN"
            ),

            splunk_index=os.getenv(
                "SPLUNK_GLEAN_INDEX"
            ),

            splunk_verify_tls=_env_bool(
                "SPLUNK_VERIFY_TLS",
                default=True,
            ),

            splunk_timeout_seconds=_env_int(
                "SPLUNK_TIMEOUT_SECONDS",
                default=30,
            ),

            splunk_earliest_time=os.getenv(
                "SPLUNK_EARLIEST_TIME",
                "-24h",
            ),

            splunk_latest_time=os.getenv(
                "SPLUNK_LATEST_TIME",
                "now",
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_run_telemetry(
        self,
        run_id: str,
    ) -> RawTelemetryBundle:
        """
        Retrieve all currently available evidence for one run.
        """

        if not run_id.strip():
            raise ValueError(
                "run_id must not be empty"
            )

        if self.mode == "mock":
            return self._load_mock(
                run_id
            )

        if self.mode == "splunk":
            return self._load_from_splunk(
                run_id
            )

        raise TelemetryConfigurationError(
            f"Unsupported telemetry mode: {self.mode}"
        )

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _load_mock(
        self,
        run_id: str,
    ) -> RawTelemetryBundle:

        if not self.mock_file:
            raise TelemetryConfigurationError(
                "MOCK_TELEMETRY_FILE is required "
                "when TELEMETRY_MODE=mock"
            )

        path = Path(
            self.mock_file
        )

        if not path.exists():
            raise TelemetryConfigurationError(
                f"Mock telemetry file not found: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise TelemetryClientError(
                f"Invalid JSON in mock telemetry file: {path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise TelemetryClientError(
                "Mock telemetry fixture must contain "
                "a JSON object."
            )

        fixture_run_id = data.get(
            "run_id"
        )

        if (
            fixture_run_id
            and str(fixture_run_id) != run_id
        ):
            raise TelemetryNotFoundError(
                f"Fixture contains run_id="
                f"{fixture_run_id}, not {run_id}"
            )

        workflow_run_event = data.get(
            "workflow_run_event"
        )

        if not isinstance(
            workflow_run_event,
            dict,
        ):
            raise TelemetryNotFoundError(
                "Mock fixture does not contain "
                "workflow_run_event"
            )

        return RawTelemetryBundle(
            run_id=run_id,

            workflow_run_event=(
                workflow_run_event
            ),

            action_events=_safe_list_of_dicts(
                data.get(
                    "action_events",
                    [],
                )
            ),

            mcp_events=_safe_list_of_dicts(
                data.get(
                    "mcp_events",
                    [],
                )
            ),

            protect_findings=_safe_list_of_dicts(
                data.get(
                    "protect_findings",
                    [],
                )
            ),

            final_output_text=data.get(
                "final_output_text"
            ),

            source="MOCK",

            retrieval_warnings=_safe_list_of_strings(
                data.get(
                    "retrieval_warnings",
                    [],
                )
            ),
        )

    # ------------------------------------------------------------------
    # Splunk mode
    # ------------------------------------------------------------------

    def _load_from_splunk(
        self,
        run_id: str,
    ) -> RawTelemetryBundle:

        self._validate_splunk_configuration()

        warnings: list[str] = []

        # --------------------------------------------------------------
        # WORKFLOW_RUN
        # --------------------------------------------------------------

        workflow_results = self._splunk_search(
            self._build_workflow_run_search(
                run_id
            )
        )

        workflow_event = self._first_event(
            workflow_results
        )

        if workflow_event is None:
            raise TelemetryNotFoundError(
                f"No WORKFLOW_RUN telemetry found "
                f"for run_id={run_id}"
            )

        # --------------------------------------------------------------
        # ACTION
        # --------------------------------------------------------------

        action_results = self._splunk_search(
            self._build_action_search(
                run_id
            )
        )

        action_events = [
            event
            for result in action_results
            if (
                event := self._result_to_event(
                    result
                )
            )
            is not None
        ]

        # --------------------------------------------------------------
        # MCP_USAGE
        # --------------------------------------------------------------

        mcp_events: list[
            dict[str, Any]
        ] = []

        mcp_search = os.getenv(
            "SPLUNK_MCP_SEARCH"
        )

        if mcp_search:
            rendered_search = (
                mcp_search.replace(
                    "{run_id}",
                    self._escape_splunk_value(
                        run_id
                    ),
                )
            )

            mcp_results = self._splunk_search(
                rendered_search
            )

            mcp_events = [
                event
                for result in mcp_results
                if (
                    event := self._result_to_event(
                        result
                    )
                )
                is not None
            ]

        else:
            warnings.append(
                "MCP telemetry was not queried because "
                "SPLUNK_MCP_SEARCH is not configured."
            )

        # --------------------------------------------------------------
        # Protect findings
        # --------------------------------------------------------------

        protect_findings: list[
            dict[str, Any]
        ] = []

        protect_search = os.getenv(
            "SPLUNK_PROTECT_SEARCH"
        )

        if protect_search:
            rendered_search = (
                protect_search.replace(
                    "{run_id}",
                    self._escape_splunk_value(
                        run_id
                    ),
                )
            )

            protect_results = (
                self._splunk_search(
                    rendered_search
                )
            )

            protect_findings = [
                self._result_payload(
                    item
                )
                for item in protect_results
            ]

        else:
            warnings.append(
                "Protect findings were not queried because "
                "SPLUNK_PROTECT_SEARCH is not configured."
            )

        # --------------------------------------------------------------
        # Final output
        # --------------------------------------------------------------

        final_output_text = None

        output_search = os.getenv(
            "SPLUNK_OUTPUT_SEARCH"
        )

        if output_search:
            rendered_search = (
                output_search.replace(
                    "{run_id}",
                    self._escape_splunk_value(
                        run_id
                    ),
                )
            )

            output_results = (
                self._splunk_search(
                    rendered_search
                )
            )

            final_output_text = (
                self._extract_output_text(
                    output_results
                )
            )

        else:
            warnings.append(
                "Final output text was not queried because "
                "SPLUNK_OUTPUT_SEARCH is not configured."
            )

        return RawTelemetryBundle(
            run_id=run_id,

            workflow_run_event=workflow_event,

            action_events=action_events,

            mcp_events=mcp_events,

            protect_findings=protect_findings,

            final_output_text=final_output_text,

            source="SPLUNK",

            retrieval_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Splunk search construction
    # ------------------------------------------------------------------

    def _build_workflow_run_search(
        self,
        run_id: str,
    ) -> str:
        """
        Retrieve the Glean WORKFLOW_RUN event.

        We use spath because Glean logs generally arrive as structured
        JSON and field extraction can vary depending on the Splunk
        ingestion configuration.
        """

        escaped_run_id = (
            self._escape_splunk_value(
                run_id
            )
        )

        return (
            f'search index="{self.splunk_index}" '
            'Type="WORKFLOW_RUN" '
            f'WorkflowRun.RunId="{escaped_run_id}" '
            "| head 1"
        )

    def _build_action_search(
        self,
        run_id: str,
    ) -> str:
        """
        Glean ACTION records now expose WorkflowRunId, allowing actions
        to be associated with the workflow run that caused them.
        """

        escaped_run_id = (
            self._escape_splunk_value(
                run_id
            )
        )

        return (
            f'search index="{self.splunk_index}" '
            'Type="ACTION" '
            f'Action.WorkflowRunId="{escaped_run_id}"'
        )

    # ------------------------------------------------------------------
    # Splunk REST API
    # ------------------------------------------------------------------

    def _splunk_search(
        self,
        search: str,
    ) -> list[dict[str, Any]]:
        """
        Execute a Splunk search using the v2 streaming export endpoint.

        POST:
            /services/search/v2/jobs/export

        Results are requested as JSON.
        """

        endpoint = (
            f"{self.splunk_base_url}"
            "/services/search/v2/jobs/export"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.splunk_token}"
            ),
            "Accept": "application/json",
        }

        data = {
            "search": search,
            "output_mode": "json",
            "earliest_time": (
                self.splunk_earliest_time
            ),
            "latest_time": (
                self.splunk_latest_time
            ),
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=data,
                timeout=self.splunk_timeout_seconds,
                verify=self.splunk_verify_tls,
                stream=True,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise TelemetryClientError(
                f"Splunk search request failed: {exc}"
            ) from exc

        return self._parse_splunk_stream(
            response
        )

    @staticmethod
    def _parse_splunk_stream(
        response: requests.Response,
    ) -> list[dict[str, Any]]:
        """
        Splunk's export endpoint may stream one JSON result object
        per line.

        Invalid or empty lines are ignored.
        """

        results: list[
            dict[str, Any]
        ] = []

        for raw_line in response.iter_lines(
            decode_unicode=True
        ):

            if not raw_line:
                continue

            line = raw_line.strip()

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
    # Splunk result extraction
    # ------------------------------------------------------------------

    def _first_event(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        for result in results:

            event = self._result_to_event(
                result
            )

            if event is not None:
                return event

        return None

    def _result_to_event(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Convert a Splunk result into the original Glean event.

        Common Splunk export shapes include:

            {
                "result": {
                    "_raw": "{...JSON...}"
                }
            }

        or already extracted fields.
        """

        payload = self._result_payload(
            result
        )

        raw = payload.get(
            "_raw"
        )

        if isinstance(
            raw,
            dict,
        ):
            return raw

        if isinstance(
            raw,
            str,
        ):
            try:
                parsed = json.loads(
                    raw
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except json.JSONDecodeError:
                pass

        # If the ingestion pipeline already extracted the
        # Glean event structure, use it directly.
        if (
            "jsonPayload" in payload
            or "Type" in payload
            or "type" in payload
        ):
            return payload

        return None

    @staticmethod
    def _result_payload(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Splunk export usually wraps fields inside "result".
        """

        nested = result.get(
            "result"
        )

        if isinstance(
            nested,
            dict,
        ):
            return nested

        return result

    # ------------------------------------------------------------------
    # Final output extraction
    # ------------------------------------------------------------------

    def _extract_output_text(
        self,
        results: list[dict[str, Any]],
    ) -> str | None:
        """
        Attempt to retrieve a final response field from a configured
        Splunk query.

        Because organizations may map Glean output fields differently,
        this intentionally checks several common aliases.
        """

        possible_fields = (
            "final_output",
            "output",
            "response",
            "response_text",
            "message",
            "text",
        )

        for result in results:

            payload = self._result_payload(
                result
            )

            for field_name in possible_fields:

                value = payload.get(
                    field_name
                )

                if isinstance(
                    value,
                    str,
                ) and value.strip():
                    return value.strip()

        return None

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_splunk_configuration(
        self,
    ) -> None:

        missing = []

        if not self.splunk_base_url:
            missing.append(
                "SPLUNK_BASE_URL"
            )

        if not self.splunk_token:
            missing.append(
                "SPLUNK_TOKEN"
            )

        if not self.splunk_index:
            missing.append(
                "SPLUNK_GLEAN_INDEX"
            )

        if missing:
            raise TelemetryConfigurationError(
                "Missing Splunk configuration: "
                + ", ".join(
                    missing
                )
            )

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_splunk_value(
        value: str,
    ) -> str:
        """
        Escape characters that could modify a Splunk quoted value.

        run_id values should normally be opaque identifiers anyway,
        but never concatenate completely untrusted strings into SPL.
        """

        return (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "")
            .replace("\r", "")
        )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


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


def _safe_list_of_dicts(
    value: Any,
) -> list[dict[str, Any]]:

    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        item
        for item in value
        if isinstance(
            item,
            dict,
        )
    ]


def _safe_list_of_strings(
    value: Any,
) -> list[str]:

    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        str(item)
        for item in value
    ]
