"""
Guardian Telemetry Swarm
telemetry_normalizer.py

Deterministically converts raw Glean telemetry into NormalizedTelemetry.

NO LLM.
NO CrewAI.
NO LangGraph.

Responsibilities:
- unwrap Glean Customer Event records
- normalize WORKFLOW_RUN metadata
- normalize workflow steps
- extract document citations
- join ACTION events by WorkflowRunId
- normalize MCP_USAGE events
- normalize optional Glean Protect findings
- inspect final output deterministically
- calculate counts
- record missing evidence

Security principles:

OBSERVATION != JUDGMENT
UNKNOWN != SAFE
MISSING EVIDENCE != NORMAL
NORMALIZATION != SECURITY DECISION
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from models import (
    ActionExecution,
    AgentBaseline,
    DocumentAccess,
    InitiatorType,
    MCPUsage,
    NormalizedTelemetry,
    OutputObservation,
    ProtectAction,
    ProtectFinding,
    RunMetadata,
    RunStatus,
    Severity,
    StepExecution,
    TelemetryCounts,
)


# ---------------------------------------------------------------------------
# Deterministic capability metadata
# ---------------------------------------------------------------------------

# These registries should eventually move into configuration files.
#
# Do NOT ask an LLM whether a tool is write-capable.
#
# If a tool/action is absent from these registries, the normalizer marks
# classification evidence as incomplete rather than assuming the action
# is harmless.

DEFAULT_ACTION_CAPABILITIES: dict[str, dict[str, bool]] = {
    "READ_DOCUMENT": {
        "write_capable": False,
        "external": False,
    },
    "SEARCH": {
        "write_capable": False,
        "external": False,
    },
    "EMPLOYEE_SEARCH": {
        "write_capable": False,
        "external": False,
    },
    "SEND_EMAIL": {
        "write_capable": True,
        "external": True,
    },
    "SEND_MESSAGE": {
        "write_capable": True,
        "external": True,
    },
    "WRITE_DOCUMENT": {
        "write_capable": True,
        "external": False,
    },
    "SHARE_DOCUMENT": {
        "write_capable": True,
        "external": True,
    },
    "DELETE_DOCUMENT": {
        "write_capable": True,
        "external": False,
    },
    "MOVE_DOCUMENT": {
        "write_capable": True,
        "external": False,
    },
    "CHANGE_PERMISSIONS": {
        "write_capable": True,
        "external": False,
    },
    "CREATE_RECORD": {
        "write_capable": True,
        "external": False,
    },
    "DELETE_RECORD": {
        "write_capable": True,
        "external": False,
    },
}


DEFAULT_MCP_CAPABILITIES: dict[str, dict[str, bool]] = {
    # Populate with tenant-approved MCP tools as they are discovered.
    #
    # Example:
    #
    # "JIRA_CREATE_ISSUE": {
    #     "write_capable": True,
    #     "external": True,
    # }
}


# ---------------------------------------------------------------------------
# Regexes used ONLY for deterministic output observations
# ---------------------------------------------------------------------------

URL_RE = re.compile(
    r"https?://[^\s]+",
    re.IGNORECASE,
)

CODE_FENCE_RE = re.compile(
    r"```.*?```",
    re.DOTALL,
)

POSSIBLE_SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
]

POSSIBLE_PII_PATTERNS = [
    # SSN-like structure only.
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),

    # Email address.
    re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------


class TelemetryNormalizer:
    """
    Converts raw telemetry into the strict NormalizedTelemetry Pydantic model.

    Expected inputs:

    workflow_run_event:
        One Glean WORKFLOW_RUN customer event.

    action_events:
        Zero or more Glean ACTION customer events.

    mcp_events:
        Zero or more Glean MCP_USAGE customer events.

    protect_findings:
        Optional normalized/raw Protect finding dictionaries.

    final_output_text:
        Optional final agent response.

    baseline:
        Behavioral contract for the monitored agent.
    """

    def __init__(
        self,
        baseline: AgentBaseline,
        action_capabilities: dict[str, dict[str, bool]] | None = None,
        mcp_capabilities: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        self.baseline = baseline

        self.action_capabilities = {
            **DEFAULT_ACTION_CAPABILITIES,
            **(action_capabilities or {}),
        }

        self.mcp_capabilities = {
            **DEFAULT_MCP_CAPABILITIES,
            **(mcp_capabilities or {}),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        workflow_run_event: dict[str, Any],
        action_events: Iterable[dict[str, Any]] | None = None,
        mcp_events: Iterable[dict[str, Any]] | None = None,
        protect_findings: Iterable[dict[str, Any]] | None = None,
        final_output_text: str | None = None,
    ) -> NormalizedTelemetry:

        missing_evidence: list[str] = []
        source_events: list[str] = []

        workflow_payload = self._extract_event_payload(
            workflow_run_event,
            expected_type="WORKFLOW_RUN",
        )

        if not workflow_payload:
            raise ValueError(
                "A valid WORKFLOW_RUN event is required."
            )

        source_events.append("WORKFLOW_RUN")

        run = self._normalize_run_metadata(
            workflow_payload,
            missing_evidence,
        )

        steps = self._normalize_steps(
            workflow_payload,
        )

        document_accesses = self._extract_document_accesses(
            workflow_payload,
        )

        actions = self._normalize_actions(
            run_id=run.run_id,
            events=list(action_events or []),
            missing_evidence=missing_evidence,
        )

        if action_events:
            source_events.append("ACTION")

        mcp_usage = self._normalize_mcp_events(
            events=list(mcp_events or []),
            missing_evidence=missing_evidence,
        )

        if mcp_events:
            source_events.append("MCP_USAGE")

        normalized_protect = self._normalize_protect_findings(
            list(protect_findings or [])
        )

        if protect_findings:
            source_events.append("PROTECT_FINDING")

        output = self._normalize_output(
            final_output_text
        )

        if final_output_text is None:
            missing_evidence.append("FINAL_OUTPUT")

        counts = self._build_counts(
            steps=steps,
            document_accesses=document_accesses,
            actions=actions,
            mcp_usage=mcp_usage,
            protect_findings=normalized_protect,
        )

        missing_evidence = sorted(set(missing_evidence))

        return NormalizedTelemetry(
            run=run,
            steps=steps,
            document_accesses=document_accesses,
            actions=actions,
            mcp_usage=mcp_usage,
            protect_findings=normalized_protect,
            output=output,
            counts=counts,
            source_events=source_events,
            evidence_complete=len(missing_evidence) == 0,
            missing_evidence=missing_evidence,
        )

    # ------------------------------------------------------------------
    # WORKFLOW_RUN
    # ------------------------------------------------------------------

    def _normalize_run_metadata(
        self,
        payload: dict[str, Any],
        missing_evidence: list[str],
    ) -> RunMetadata:

        run_id = self._get(payload, "RunId", "runId", "run_id")

        if not run_id:
            raise ValueError(
                "WORKFLOW_RUN event does not contain RunId."
            )

        workflow_id = self._get(
            payload,
            "WorkflowId",
            "workflowId",
            "workflow_id",
        )

        initiator_raw = self._get(
            payload,
            "Initiator",
            "initiator",
        )

        platform = self._get(
            payload,
            "Platform",
            "platform",
        )

        duration_ms = self._safe_int(
            self._get(
                payload,
                "TotalExecutionLatency",
                "totalExecutionLatency",
                "total_execution_latency",
            )
        )

        started_at = self._get(
            payload,
            "WorkflowCreatedAt",
            "workflowCreatedAt",
            "workflow_created_at",
        )

        status = self._determine_workflow_status(payload)

        if not workflow_id:
            missing_evidence.append("WORKFLOW_ID")

        if not initiator_raw:
            missing_evidence.append("RUN_INITIATOR")

        return RunMetadata(
            run_id=str(run_id),
            workflow_id=str(workflow_id) if workflow_id else None,
            agent_name=self.baseline.agent_name,
            initiator=self._normalize_initiator(initiator_raw),
            platform=str(platform) if platform else None,
            status=status,
            started_at=str(started_at) if started_at else None,
            completed_at=None,
            duration_ms=duration_ms,
        )

    def _determine_workflow_status(
        self,
        payload: dict[str, Any],
    ) -> RunStatus:
        """
        WORKFLOW_RUN itself contains WorkflowExecutions.

        We conservatively aggregate their statuses.
        """

        executions = self._get(
            payload,
            "WorkflowExecutions",
            "workflowExecutions",
            "workflow_executions",
            default=[],
        )

        if not isinstance(executions, list) or not executions:
            return RunStatus.UNKNOWN

        statuses = {
            str(
                self._get(
                    execution,
                    "Status",
                    "status",
                    default="",
                )
            ).upper()
            for execution in executions
            if isinstance(execution, dict)
        }

        if "ERROR" in statuses or "FAILED" in statuses:
            return RunStatus.FAILED

        if "CANCELLED" in statuses or "CANCELED" in statuses:
            return RunStatus.CANCELLED

        if statuses and statuses <= {"SUCCESS"}:
            return RunStatus.SUCCESS

        return RunStatus.UNKNOWN

    # ------------------------------------------------------------------
    # Step executions
    # ------------------------------------------------------------------

    def _normalize_steps(
        self,
        payload: dict[str, Any],
    ) -> list[StepExecution]:

        raw_steps = self._get(
            payload,
            "StepExecutions",
            "stepExecutions",
            "step_executions",
            default=[],
        )

        if not isinstance(raw_steps, list):
            return []

        results: list[StepExecution] = []

        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue

            step_id = self._get(
                raw,
                "StepId",
                "stepId",
                "step_id",
            )

            raw_status = self._get(
                raw,
                "Status",
                "status",
            )

            error_type = self._get(
                raw,
                "ErrorType",
                "errorType",
                "error_type",
            )

            action_instance_id = self._get(
                raw,
                "ActionInstanceId",
                "actionInstanceId",
                "action_instance_id",
            )

            # Glean's exported StepExecution schema gives us StepId,
            # but the human-readable name may come from local workflow
            # metadata in a later enrichment stage.
            step_name = (
                str(step_id)
                if step_id
                else "UNKNOWN_STEP"
            )

            results.append(
                StepExecution(
                    step_id=str(step_id) if step_id else None,
                    step_name=step_name,
                    step_type=None,
                    status=self._normalize_step_status(raw_status),
                    tool_name=(
                        str(action_instance_id)
                        if action_instance_id
                        else None
                    ),
                    error_present=bool(error_type),
                    error_type=(
                        str(error_type)
                        if error_type
                        else None
                    ),
                    metadata={
                        "workflow_id": self._get(
                            raw,
                            "WorkflowId",
                            "workflowId",
                            "workflow_id",
                        ),
                        "raw_step_status": raw_status,
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Document/citation extraction
    # ------------------------------------------------------------------

    def _extract_document_accesses(
        self,
        payload: dict[str, Any],
    ) -> list[DocumentAccess]:

        raw_steps = self._get(
            payload,
            "StepExecutions",
            "stepExecutions",
            "step_executions",
            default=[],
        )

        accesses: list[DocumentAccess] = []
        seen: set[tuple[str | None, str | None]] = set()

        if not isinstance(raw_steps, list):
            return accesses

        for step in raw_steps:
            if not isinstance(step, dict):
                continue

            citations = self._get(
                step,
                "Citations",
                "citations",
                default=[],
            )

            if not isinstance(citations, list):
                continue

            for citation in citations:
                if not isinstance(citation, dict):
                    continue

                source_document = self._get(
                    citation,
                    "SourceDocument",
                    "sourceDocument",
                    "source_document",
                )

                source_file = self._get(
                    citation,
                    "SourceFile",
                    "sourceFile",
                    "source_file",
                )

                access = None

                if isinstance(source_document, dict):
                    document_id = self._get(
                        source_document,
                        "Id",
                        "id",
                    )

                    document_name = self._get(
                        source_document,
                        "Title",
                        "title",
                    )

                    source_system = self._get(
                        source_document,
                        "Datasource",
                        "datasource",
                        "ConnectorType",
                        "connectorType",
                    )

                    access = DocumentAccess(
                        document_id=(
                            str(document_id)
                            if document_id
                            else None
                        ),
                        document_name=(
                            str(document_name)
                            if document_name
                            else None
                        ),
                        source_system=(
                            str(source_system)
                            if source_system
                            else None
                        ),
                        access_type="READ",
                        user_selected=False,
                        citation_present=True,
                    )

                elif isinstance(source_file, dict):
                    document_id = self._get(
                        source_file,
                        "Id",
                        "id",
                    )

                    document_name = self._get(
                        source_file,
                        "Name",
                        "name",
                    )

                    access = DocumentAccess(
                        document_id=(
                            str(document_id)
                            if document_id
                            else None
                        ),
                        document_name=(
                            str(document_name)
                            if document_name
                            else None
                        ),
                        source_system="FILE",
                        access_type="READ",
                        user_selected=False,
                        citation_present=True,
                    )

                if access:
                    key = (
                        access.document_id,
                        access.document_name,
                    )

                    if key not in seen:
                        seen.add(key)
                        accesses.append(access)

        return accesses

    # ------------------------------------------------------------------
    # ACTION events
    # ------------------------------------------------------------------

    def _normalize_actions(
        self,
        run_id: str,
        events: list[dict[str, Any]],
        missing_evidence: list[str],
    ) -> list[ActionExecution]:

        actions: list[ActionExecution] = []

        for event in events:
            payload = self._extract_event_payload(
                event,
                expected_type="ACTION",
            )

            if not payload:
                continue

            event_run_id = self._get(
                payload,
                "WorkflowRunId",
                "workflowRunId",
                "workflow_run_id",
            )

            # ACTION events can exist outside workflow execution.
            # We only want actions belonging to this monitored run.
            if event_run_id and str(event_run_id) != run_id:
                continue

            if not event_run_id:
                continue

            action_name_raw = self._get(
                payload,
                "ActionName",
                "actionName",
                "action_name",
            )

            if not action_name_raw:
                missing_evidence.append(
                    "ACTION_WITHOUT_NAME"
                )
                continue

            action_name = str(action_name_raw)

            capability = self._lookup_capability(
                action_name,
                self.action_capabilities,
            )

            if capability is None:
                missing_evidence.append(
                    f"ACTION_CLASSIFICATION:{action_name}"
                )

                write_capable = False
                external = False

            else:
                write_capable = capability["write_capable"]
                external = capability["external"]

            action_id = self._get(
                payload,
                "ActionId",
                "actionId",
                "action_id",
            )

            execution_status = self._get(
                payload,
                "ExecutionStatus",
                "executionStatus",
                "execution_status",
            )

            error_type = self._get(
                payload,
                "ErrorType",
                "errorType",
                "error_type",
            )

            actions.append(
                ActionExecution(
                    action_id=(
                        str(action_id)
                        if action_id
                        else None
                    ),
                    action_name=action_name,
                    tool_name=None,
                    destination=None,
                    status=self._normalize_execution_status(
                        execution_status
                    ),
                    write_capable=write_capable,
                    external=external,
                    approval_required=False,
                    approved=None,
                    parameters={
                        "caller": self._get(
                            payload,
                            "Caller",
                            "caller",
                        ),
                        "execution_mode": self._get(
                            payload,
                            "ExecutionMode",
                            "executionMode",
                        ),
                        "is_dry_run": self._get(
                            payload,
                            "IsDryRun",
                            "isDryRun",
                            "is_dry_run",
                        ),
                        "error_type": error_type,
                    },
                )
            )

        return actions

    # ------------------------------------------------------------------
    # MCP events
    # ------------------------------------------------------------------

    def _normalize_mcp_events(
        self,
        events: list[dict[str, Any]],
        missing_evidence: list[str],
    ) -> list[MCPUsage]:

        results: list[MCPUsage] = []

        for event in events:
            payload = self._extract_event_payload(
                event,
                expected_type="MCP_USAGE",
            )

            if not payload:
                continue

            event_type = self._get(
                payload,
                "EventType",
                "eventType",
                "event_type",
            )

            # MCP_USAGE includes things other than actual tool calls,
            # such as tool listing.
            if event_type and str(event_type).upper() != "TOOL_CALL":
                continue

            tool_name_raw = self._get(
                payload,
                "ToolName",
                "toolName",
                "tool_name",
            )

            if not tool_name_raw:
                missing_evidence.append(
                    "MCP_TOOL_WITHOUT_NAME"
                )
                continue

            tool_name = str(tool_name_raw)

            capability = self._lookup_capability(
                tool_name,
                self.mcp_capabilities,
            )

            if capability is None:
                missing_evidence.append(
                    f"MCP_CLASSIFICATION:{tool_name}"
                )
                write_capable = False
            else:
                write_capable = capability["write_capable"]

            server_name = self._get(
                payload,
                "ExternalServerName",
                "externalServerName",
                "external_server_name",
                "Server",
                "server",
            )

            status = self._get(
                payload,
                "Status",
                "status",
            )

            results.append(
                MCPUsage(
                    server_name=(
                        str(server_name)
                        if server_name
                        else None
                    ),
                    tool_name=tool_name,
                    destination=(
                        str(server_name)
                        if server_name
                        else None
                    ),
                    status=self._normalize_execution_status(
                        status
                    ),
                    write_capable=write_capable,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Protect findings
    # ------------------------------------------------------------------

    def _normalize_protect_findings(
        self,
        findings: list[dict[str, Any]],
    ) -> list[ProtectFinding]:

        results: list[ProtectFinding] = []

        for finding in findings:
            policy_name = self._get(
                finding,
                "policy_name",
                "policyName",
                "PolicyName",
                "policy",
            )

            if not policy_name:
                continue

            severity = self._normalize_severity(
                self._get(
                    finding,
                    "severity",
                    "Severity",
                    default="INFO",
                )
            )

            action = self._normalize_protect_action(
                self._get(
                    finding,
                    "action",
                    "Action",
                    default="UNKNOWN",
                )
            )

            results.append(
                ProtectFinding(
                    finding_id=self._optional_string(
                        self._get(
                            finding,
                            "finding_id",
                            "findingId",
                            "FindingId",
                        )
                    ),
                    policy_name=str(policy_name),
                    severity=severity,
                    action=action,
                    source=self._optional_string(
                        self._get(
                            finding,
                            "source",
                            "Source",
                        )
                    ),
                    description=self._optional_string(
                        self._get(
                            finding,
                            "description",
                            "Description",
                        )
                    ),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Output observations
    # ------------------------------------------------------------------

    def _normalize_output(
        self,
        text: str | None,
    ) -> OutputObservation | None:

        if text is None:
            return None

        return OutputObservation(
            sentence_count=self._count_sentences(text),
            character_count=len(text),
            contains_url=bool(URL_RE.search(text)),
            contains_code=bool(CODE_FENCE_RE.search(text)),
            contains_possible_secret=any(
                pattern.search(text)
                for pattern in POSSIBLE_SECRET_PATTERNS
            ),
            contains_possible_pii=any(
                pattern.search(text)
                for pattern in POSSIBLE_PII_PATTERNS
            ),
            output_available=True,
        )

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def _build_counts(
        self,
        steps: list[StepExecution],
        document_accesses: list[DocumentAccess],
        actions: list[ActionExecution],
        mcp_usage: list[MCPUsage],
        protect_findings: list[ProtectFinding],
    ) -> TelemetryCounts:

        return TelemetryCounts(
            step_count=len(steps),

            document_read_count=sum(
                1
                for item in document_accesses
                if item.access_type.upper() == "READ"
            ),

            write_action_count=sum(
                1
                for action in actions
                if action.write_capable
            )
            + sum(
                1
                for usage in mcp_usage
                if usage.write_capable
            ),

            external_action_count=sum(
                1
                for action in actions
                if action.external
            ),

            mcp_call_count=len(mcp_usage),

            protect_finding_count=len(protect_findings),

            error_count=sum(
                1
                for step in steps
                if step.error_present
            )
            + sum(
                1
                for action in actions
                if action.status == RunStatus.FAILED
            )
            + sum(
                1
                for usage in mcp_usage
                if usage.status == RunStatus.FAILED
            ),
        )

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _extract_event_payload(
        self,
        event: dict[str, Any],
        expected_type: str,
    ) -> dict[str, Any] | None:
        """
        Supports either:

        {
            "jsonPayload": {
                "Type": "WORKFLOW_RUN",
                "WorkflowRun": {...}
            }
        }

        or already-unwrapped mock records:

        {
            "Type": "WORKFLOW_RUN",
            "WorkflowRun": {...}
        }

        or:

        {
            "type": "WORKFLOW_RUN",
            "workflow_run": {...}
        }
        """

        if not isinstance(event, dict):
            return None

        outer = self._get(
            event,
            "jsonPayload",
            "json_payload",
            default=event,
        )

        if not isinstance(outer, dict):
            return None

        event_type = self._get(
            outer,
            "Type",
            "type",
            default="",
        )

        if str(event_type).upper() != expected_type.upper():
            return None

        aliases = {
            "WORKFLOW_RUN": [
                "WorkflowRun",
                "workflowRun",
                "workflow_run",
            ],
            "ACTION": [
                "Action",
                "action",
            ],
            "MCP_USAGE": [
                "McpUsage",
                "mcpUsage",
                "mcp_usage",
            ],
        }

        for key in aliases.get(expected_type.upper(), []):
            value = self._get(
                outer,
                key,
            )

            if isinstance(value, dict):
                return value

        # Mock fixtures may place the fields directly on the event.
        return outer

    # ------------------------------------------------------------------
    # Status normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_step_status(
        value: Any,
    ) -> RunStatus:

        status = str(value or "").upper()

        if status in {
            "EXECUTED",
            "SUCCESS",
            "SUCCEEDED",
            "COMPLETE",
            "COMPLETED",
        }:
            return RunStatus.SUCCESS

        if status in {
            "ERROR",
            "FAILED",
            "FAILURE",
        }:
            return RunStatus.FAILED

        if status in {
            "CANCELLED",
            "CANCELED",
            "SKIPPED",
        }:
            return RunStatus.CANCELLED

        if status in {
            "RUNNING",
            "IN_PROGRESS",
        }:
            return RunStatus.RUNNING

        return RunStatus.UNKNOWN

    @staticmethod
    def _normalize_execution_status(
        value: Any,
    ) -> RunStatus:

        status = str(value or "").upper()

        if status in {
            "SUCCESS",
            "SUCCEEDED",
            "EXECUTED",
            "COMPLETE",
            "COMPLETED",
        }:
            return RunStatus.SUCCESS

        if status in {
            "ERROR",
            "FAILED",
            "FAILURE",
        }:
            return RunStatus.FAILED

        if status in {
            "CANCELLED",
            "CANCELED",
        }:
            return RunStatus.CANCELLED

        if status in {
            "RUNNING",
            "IN_PROGRESS",
        }:
            return RunStatus.RUNNING

        return RunStatus.UNKNOWN

    @staticmethod
    def _normalize_initiator(
        value: Any,
    ) -> InitiatorType:

        initiator = str(value or "").upper()

        if initiator == "USER":
            return InitiatorType.USER

        if initiator in {
            "AUTOMATION",
            "SCHEDULED",
        }:
            return InitiatorType.SCHEDULED

        if initiator in {
            "REST_API",
            "API",
        }:
            return InitiatorType.API

        if initiator in {
            "GLEAN",
            "EVAL",
            "EVAL_PTU",
            "PRODUCTION_PROBE",
            "LOAD_TEST",
            "HEALTH_CHECK",
            "HEALTH_AGENT_CHECK",
            "WORKFLOW_COMPILER",
        }:
            return InitiatorType.SYSTEM

        return InitiatorType.UNKNOWN

    # ------------------------------------------------------------------
    # Simple deterministic classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_severity(
        value: Any,
    ) -> Severity:

        text = str(value or "").upper()

        try:
            return Severity(text)
        except ValueError:
            return Severity.INFO

    @staticmethod
    def _normalize_protect_action(
        value: Any,
    ) -> ProtectAction:

        text = str(value or "").upper()

        try:
            return ProtectAction(text)
        except ValueError:
            return ProtectAction.UNKNOWN

    @staticmethod
    def _lookup_capability(
        name: str,
        registry: dict[str, dict[str, bool]],
    ) -> dict[str, bool] | None:

        normalized_name = (
            name.strip()
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return registry.get(normalized_name)

    # ------------------------------------------------------------------
    # Sentence counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_sentences(text: str) -> int:
        """
        Simple deterministic count suitable for this lab.

        This is intentionally not an LLM operation.

        It is not intended to be a perfect linguistic sentence segmenter.
        """

        cleaned = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not cleaned:
            return 0

        pieces = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        return len(
            [
                piece
                for piece in pieces
                if piece.strip()
            ]
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get(
        mapping: dict[str, Any],
        *names: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a field while tolerating emitter/casing differences.

        Glean documents that different event emitters can use slightly
        different field naming conventions.
        """

        if not isinstance(mapping, dict):
            return default

        for name in names:
            if name in mapping:
                return mapping[name]

        lowered = {
            str(key).lower(): value
            for key, value in mapping.items()
        }

        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]

        return default

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        text = str(value).strip()

        return text or None
