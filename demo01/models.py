"""
Guardian Telemetry Swarm
models.py

Shared Pydantic models for:

- normalized Glean telemetry
- monitored-agent behavioral baselines
- specialist Spy Agent findings
- Guardian correlation decisions

Design principles:

OBSERVATION != JUDGMENT
ANOMALY != COMPROMISE
DETECTION != ENFORCEMENT
MISSING EVIDENCE != SAFE
LLM OUTPUT != TRUSTED STATE

All telemetry should be normalized deterministically before being
presented to CrewAI agents or LangGraph.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnalysisStatus(StrEnum):
    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GuardianStatus(StrEnum):
    NORMAL = "NORMAL"
    REVIEW = "REVIEW"
    ALERT = "ALERT"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class InitiatorType(StrEnum):
    USER = "USER"
    SYSTEM = "SYSTEM"
    SCHEDULED = "SCHEDULED"
    API = "API"
    UNKNOWN = "UNKNOWN"


class ProtectAction(StrEnum):
    FLAG = "FLAG"
    BLOCK = "BLOCK"
    ALLOW = "ALLOW"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class GuardianBaseModel(BaseModel):
    """
    Strict default model.

    Security telemetry should not silently accept unexpected fields after
    normalization. Unknown fields should be handled by the normalizer first.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Agent baseline
# ---------------------------------------------------------------------------


class ExpectedBehavior(GuardianBaseModel):
    """
    Quantitative behavioral limits for the monitored agent.
    """

    document_reads_max: int = Field(default=1, ge=0)

    write_actions_max: int = Field(default=0, ge=0)

    mcp_calls_max: int = Field(default=0, ge=0)

    external_actions_max: int = Field(default=0, ge=0)

    expected_output_sentences: int | None = Field(
        default=None,
        ge=1,
    )


class AgentBaseline(GuardianBaseModel):
    """
    Defines what the monitored business agent is supposed to do.

    This is the security contract against which observed behavior is compared.
    """

    agent_name: str

    agent_id: str | None = None

    allowed_initiators: list[InitiatorType] = Field(
        default_factory=lambda: [InitiatorType.USER]
    )

    expected_behavior: ExpectedBehavior

    allowed_capabilities: list[str] = Field(default_factory=list)

    forbidden_capabilities: list[str] = Field(default_factory=list)

    expected_steps: list[str] = Field(default_factory=list)

    description: str | None = None


# ---------------------------------------------------------------------------
# Normalized Glean telemetry
# ---------------------------------------------------------------------------


class RunMetadata(GuardianBaseModel):
    """
    High-level metadata associated with one Glean workflow/agent run.
    """

    run_id: str

    workflow_id: str | None = None

    agent_name: str

    initiator: InitiatorType = InitiatorType.UNKNOWN

    initiator_id: str | None = None

    platform: str | None = None

    status: RunStatus = RunStatus.UNKNOWN

    started_at: str | None = None

    completed_at: str | None = None

    duration_ms: int | None = Field(default=None, ge=0)


class StepExecution(GuardianBaseModel):
    """
    One normalized workflow execution step.
    """

    step_id: str | None = None

    step_name: str

    step_type: str | None = None

    status: RunStatus = RunStatus.UNKNOWN

    tool_name: str | None = None

    started_at: str | None = None

    completed_at: str | None = None

    duration_ms: int | None = Field(default=None, ge=0)

    error_present: bool = False

    error_type: str | None = None

    # Never place secrets or raw sensitive values here.
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentAccess(GuardianBaseModel):
    """
    A document or content object touched during the run.
    """

    document_id: str | None = None

    document_name: str | None = None

    source_system: str | None = None

    access_type: str = "READ"

    user_selected: bool = False

    citation_present: bool = False


class ActionExecution(GuardianBaseModel):
    """
    Represents an observed action or write-capable operation.
    """

    action_id: str | None = None

    action_name: str

    tool_name: str | None = None

    destination: str | None = None

    status: RunStatus = RunStatus.UNKNOWN

    write_capable: bool = False

    external: bool = False

    approval_required: bool = False

    approved: bool | None = None

    # Sanitized parameters only.
    parameters: dict[str, Any] = Field(default_factory=dict)


class MCPUsage(GuardianBaseModel):
    """
    Observed MCP-server/tool invocation.
    """

    server_name: str | None = None

    tool_name: str

    destination: str | None = None

    status: RunStatus = RunStatus.UNKNOWN

    write_capable: bool = False


class ProtectFinding(GuardianBaseModel):
    """
    Normalized Glean Protect / AI security finding.
    """

    finding_id: str | None = None

    policy_name: str

    severity: Severity

    action: ProtectAction = ProtectAction.UNKNOWN

    source: str | None = None

    description: str | None = None


class OutputObservation(GuardianBaseModel):
    """
    Deterministically derived properties of the agent response.

    The raw output does not need to be passed to every security agent.
    """

    sentence_count: int | None = Field(default=None, ge=0)

    character_count: int | None = Field(default=None, ge=0)

    contains_url: bool = False

    contains_code: bool = False

    contains_possible_secret: bool = False

    contains_possible_pii: bool = False

    output_available: bool = True


class TelemetryCounts(GuardianBaseModel):
    """
    Precalculated counts.

    These should be produced by deterministic Python rather than LLMs.
    """

    step_count: int = Field(default=0, ge=0)

    document_read_count: int = Field(default=0, ge=0)

    write_action_count: int = Field(default=0, ge=0)

    external_action_count: int = Field(default=0, ge=0)

    mcp_call_count: int = Field(default=0, ge=0)

    protect_finding_count: int = Field(default=0, ge=0)

    error_count: int = Field(default=0, ge=0)


class NormalizedTelemetry(GuardianBaseModel):
    """
    Canonical evidence object consumed by the Guardian swarm.

    Glean's raw schemas should NOT be sent directly to the agents.

    Raw Glean telemetry
          |
          v
    telemetry_normalizer.py
          |
          v
    NormalizedTelemetry
          |
          v
    CrewAI Spy Agents
    """

    schema_version: str = "1.0"

    run: RunMetadata

    steps: list[StepExecution] = Field(default_factory=list)

    document_accesses: list[DocumentAccess] = Field(
        default_factory=list
    )

    actions: list[ActionExecution] = Field(default_factory=list)

    mcp_usage: list[MCPUsage] = Field(default_factory=list)

    protect_findings: list[ProtectFinding] = Field(
        default_factory=list
    )

    output: OutputObservation | None = None

    counts: TelemetryCounts

    source_events: list[str] = Field(default_factory=list)

    evidence_complete: bool = True

    missing_evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual evidence findings
# ---------------------------------------------------------------------------


class EvidenceReference(GuardianBaseModel):
    """
    Pointer back to evidence that produced a finding.

    Avoid copying raw sensitive data into LLM-generated findings.
    """

    evidence_type: str

    reference: str

    description: str | None = None


class SpyFinding(GuardianBaseModel):
    """
    One discrete security observation.
    """

    finding_code: str

    title: str

    severity: Severity

    description: str

    evidence: list[EvidenceReference] = Field(default_factory=list)

    recommendation: str | None = None


# ---------------------------------------------------------------------------
# Generic specialist-agent result
# ---------------------------------------------------------------------------


class SpyAnalysis(GuardianBaseModel):
    """
    Shared output contract used by the specialist CrewAI agents.

    Each Spy can use a more specialized subclass later if necessary.
    """

    run_id: str

    analyst: str

    status: AnalysisStatus

    severity: Severity

    confidence: float = Field(ge=0.0, le=1.0)

    summary: str

    findings: list[SpyFinding] = Field(default_factory=list)

    evidence_reviewed: list[str] = Field(default_factory=list)

    missing_evidence: list[str] = Field(default_factory=list)

    requires_guardian_attention: bool = False


# ---------------------------------------------------------------------------
# Specialized Spy output types
# ---------------------------------------------------------------------------


class RunBehaviorAnalysis(SpyAnalysis):
    analyst: str = "Run Behavior Analyst"

    unexpected_steps: list[str] = Field(default_factory=list)

    unexpected_nested_workflows: list[str] = Field(
        default_factory=list
    )

    unusual_execution_pattern: bool = False


class ToolBehaviorAnalysis(SpyAnalysis):
    analyst: str = "Tool & Action Analyst"

    unexpected_tools: list[str] = Field(default_factory=list)

    unexpected_write_actions: list[str] = Field(
        default_factory=list
    )

    unexpected_mcp_tools: list[str] = Field(default_factory=list)


class DataAccessAnalysis(SpyAnalysis):
    analyst: str = "Data Access Analyst"

    expected_documents: list[str] = Field(default_factory=list)

    observed_documents: list[str] = Field(default_factory=list)

    unexpected_documents: list[str] = Field(default_factory=list)


class OutputBehaviorAnalysis(SpyAnalysis):
    analyst: str = "Output Behavior Analyst"

    expected_sentence_count: int | None = None

    observed_sentence_count: int | None = None

    output_contract_violated: bool = False


class ProtectFindingsAnalysis(SpyAnalysis):
    analyst: str = "Protect Findings Analyst"

    protect_findings_present: bool = False

    protect_block_present: bool = False

    highest_protect_severity: Severity = Severity.INFO


# ---------------------------------------------------------------------------
# Final Guardian correlation result
# ---------------------------------------------------------------------------


class GuardianDecision(GuardianBaseModel):
    """
    Final correlated monitoring decision.

    Demo #2 is observational.

    Therefore this intentionally does NOT contain ALLOW/BLOCK.
    Enforcement belongs to Demo #3.
    """

    run_id: str

    monitored_agent: str

    guardian_status: GuardianStatus

    severity: Severity

    confidence: float = Field(ge=0.0, le=1.0)

    summary: str

    reason_codes: list[str] = Field(default_factory=list)

    findings: list[SpyFinding] = Field(default_factory=list)

    recommended_actions: list[str] = Field(default_factory=list)

    security_review_required: bool = False

    evidence_complete: bool = True

    missing_evidence: list[str] = Field(default_factory=list)

    # Hard security boundary for Demo #2.
    enforcement_performed: bool = False
