"""
Guardian Telemetry Swarm
guardian_correlator.py

Correlation/supervisor layer for the Guardian Telemetry Swarm.

This module does NOT execute the five specialist agents.

LangGraph owns specialist execution.

This module receives their completed results and answers:

    "Taken together, what does the available evidence mean?"

Responsibilities:

- aggregate specialist findings
- identify conflicts
- identify incomplete evidence
- calculate deterministic severity/disposition floors
- prevent unsafe downgrade to NORMAL
- invoke one CrewAI supervisor for cross-domain interpretation
- return a validated GuardianDecision

Final monitoring states:

    NORMAL
    REVIEW
    ALERT

This module performs NO enforcement.

Architecture:

Run Spy --------\
Tool Spy --------\
Data Spy ---------+--> deterministic correlation
Output Spy -------/             |
Protect Spy -----/              v
                         guardrail floor
                                |
                                v
                       CrewAI Correlator
                                |
                                v
                        GuardianDecision
"""

from __future__ import annotations

import json
import os
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import ValidationError

from graph_state import (
    GuardianGraphState,
    collect_specialist_errors,
)
from models import (
    AnalysisStatus,
    GuardianDecision,
    GuardianStatus,
    NormalizedTelemetry,
    ProtectAction,
    Severity,
    SpyAnalysis,
)


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------


SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


GUARDIAN_STATUS_ORDER = {
    GuardianStatus.NORMAL: 0,
    GuardianStatus.REVIEW: 1,
    GuardianStatus.ALERT: 2,
}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_llm() -> LLM:
    """
    Construct the CrewAI LLM used by the Guardian Correlator.
    """

    model_name = os.getenv(
        "CREWAI_MODEL",
        "openai/gpt-4.1-mini",
    )

    return LLM(
        model=model_name,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Correlator agent
# ---------------------------------------------------------------------------


def build_guardian_correlator_agent() -> Agent:
    """
    Build the single supervisor/correlation agent.
    """

    return Agent(
        role="Guardian Correlator",

        goal=(
            "Correlate independent telemetry-security assessments and "
            "produce a defensible monitoring disposition of NORMAL, "
            "REVIEW, or ALERT without overriding deterministic safety "
            "guardrails."
        ),

        backstory=(
            "You are the supervising security analyst for an enterprise "
            "AI-agent monitoring system. Five specialist analysts examine "
            "execution behavior, tool usage, data access, output behavior, "
            "and Glean-native security findings. Your job is to reconcile "
            "their evidence, identify agreement or conflict, account for "
            "missing evidence, and produce a concise security monitoring "
            "decision. You never perform enforcement and you never treat "
            "majority voting as a substitute for security reasoning."
        ),

        llm=build_llm(),

        verbose=_env_bool(
            "CREWAI_VERBOSE",
            default=False,
        ),

        allow_delegation=False,

        allow_code_execution=False,
    )


# ---------------------------------------------------------------------------
# Deterministic correlation
# ---------------------------------------------------------------------------


def build_correlation_evidence(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Convert the five specialist analyses into deterministic correlation
    facts.

    Python determines:

    - number of completed specialists
    - number of anomalous specialists
    - number of insufficient-evidence specialists
    - highest severity
    - Protect block presence
    - high/critical analyst count
    - specialist failures
    - overall evidence completeness
    - disagreement between specialists
    """

    analyses = _get_specialist_analyses(
        state
    )

    telemetry = state.get(
        "telemetry"
    )

    errors = collect_specialist_errors(
        state
    )

    anomalous = [
        analysis
        for analysis in analyses
        if analysis.status == AnalysisStatus.ANOMALOUS
    ]

    insufficient = [
        analysis
        for analysis in analyses
        if (
            analysis.status
            == AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
    ]

    normal = [
        analysis
        for analysis in analyses
        if analysis.status == AnalysisStatus.NORMAL
    ]

    high_or_critical = [
        analysis
        for analysis in analyses
        if (
            SEVERITY_ORDER[
                analysis.severity
            ]
            >= SEVERITY_ORDER[
                Severity.HIGH
            ]
        )
    ]

    highest_severity = _highest_severity(
        [
            analysis.severity
            for analysis in analyses
        ]
    )

    protect_analysis = state.get(
        "protect_findings_analysis"
    )

    protect_block_present = False

    protect_high_or_critical = False

    if protect_analysis is not None:

        protect_block_present = (
            protect_analysis.protect_block_present
        )

        protect_high_or_critical = (
            SEVERITY_ORDER[
                protect_analysis.highest_protect_severity
            ]
            >= SEVERITY_ORDER[
                Severity.HIGH
            ]
        )

    telemetry_complete = (
        telemetry.evidence_complete
        if isinstance(
            telemetry,
            NormalizedTelemetry,
        )
        else False
    )

    specialist_attention_count = sum(
        1
        for analysis in analyses
        if analysis.requires_guardian_attention
    )

    statuses = {
        analysis.status
        for analysis in analyses
    }

    specialist_disagreement = (
        len(statuses) > 1
    )

    return {
        "specialists_expected": 5,

        "specialists_completed": (
            len(analyses)
        ),

        "specialist_error_count": (
            len(errors)
        ),

        "specialist_errors": (
            errors
        ),

        "normal_count": (
            len(normal)
        ),

        "anomalous_count": (
            len(anomalous)
        ),

        "insufficient_evidence_count": (
            len(insufficient)
        ),

        "guardian_attention_count": (
            specialist_attention_count
        ),

        "high_or_critical_count": (
            len(high_or_critical)
        ),

        "highest_specialist_severity": (
            highest_severity.value
        ),

        "protect_block_present": (
            protect_block_present
        ),

        "protect_high_or_critical": (
            protect_high_or_critical
        ),

        "telemetry_evidence_complete": (
            telemetry_complete
        ),

        "specialist_disagreement": (
            specialist_disagreement
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic disposition floor
# ---------------------------------------------------------------------------


def determine_minimum_disposition(
    state: GuardianGraphState,
    correlation: dict[str, Any],
) -> dict[str, Any]:
    """
    Establish the LOWEST disposition the CrewAI supervisor is permitted
    to return.

    This is deliberately asymmetric.

    One decisive security fact can prevent NORMAL.

    NORMAL requires broad coherent evidence.
    """

    reason_codes: list[str] = []

    minimum_status = GuardianStatus.NORMAL

    minimum_severity = Severity.INFO


    # --------------------------------------------------------------
    # Missing specialist execution
    # --------------------------------------------------------------

    if correlation["specialist_error_count"] > 0:

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.REVIEW,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.MEDIUM,
        )

        reason_codes.append(
            "SPECIALIST_EXECUTION_FAILURE"
        )


    # --------------------------------------------------------------
    # Incomplete telemetry
    # --------------------------------------------------------------

    if not correlation[
        "telemetry_evidence_complete"
    ]:

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.REVIEW,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.LOW,
        )

        reason_codes.append(
            "TELEMETRY_INCOMPLETE"
        )


    # --------------------------------------------------------------
    # Specialist insufficient evidence
    # --------------------------------------------------------------

    if (
        correlation[
            "insufficient_evidence_count"
        ]
        > 0
    ):

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.REVIEW,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.MEDIUM,
        )

        reason_codes.append(
            "SPECIALIST_EVIDENCE_INCOMPLETE"
        )


    # --------------------------------------------------------------
    # Protect BLOCK
    # --------------------------------------------------------------

    if correlation[
        "protect_block_present"
    ]:

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.ALERT,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.HIGH,
        )

        reason_codes.append(
            "PROTECT_BLOCK"
        )


    # --------------------------------------------------------------
    # Any CRITICAL specialist
    # --------------------------------------------------------------

    if (
        correlation[
            "highest_specialist_severity"
        ]
        == Severity.CRITICAL.value
    ):

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.ALERT,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.CRITICAL,
        )

        reason_codes.append(
            "CRITICAL_SPECIALIST_FINDING"
        )


    # --------------------------------------------------------------
    # Multiple HIGH/CRITICAL specialists
    # --------------------------------------------------------------

    if (
        correlation[
            "high_or_critical_count"
        ]
        >= 2
    ):

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.ALERT,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.HIGH,
        )

        reason_codes.append(
            "MULTIPLE_HIGH_SEVERITY_SIGNALS"
        )


    # --------------------------------------------------------------
    # At least one HIGH anomaly
    # --------------------------------------------------------------

    if _contains_high_anomaly(
        state
    ):

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.REVIEW,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.HIGH,
        )

        reason_codes.append(
            "HIGH_SEVERITY_ANOMALY"
        )


    # --------------------------------------------------------------
    # Explicit write-capability anomaly
    # --------------------------------------------------------------

    tool_analysis = state.get(
        "tool_behavior_analysis"
    )

    if tool_analysis is not None:

        if tool_analysis.unexpected_write_actions:

            minimum_status = _max_guardian_status(
                minimum_status,
                GuardianStatus.ALERT,
            )

            minimum_severity = _max_severity(
                minimum_severity,
                Severity.HIGH,
            )

            reason_codes.append(
                "UNEXPECTED_WRITE_CAPABILITY"
            )


        if tool_analysis.unexpected_mcp_tools:

            minimum_status = _max_guardian_status(
                minimum_status,
                GuardianStatus.REVIEW,
            )

            minimum_severity = _max_severity(
                minimum_severity,
                Severity.MEDIUM,
            )

            reason_codes.append(
                "UNEXPECTED_MCP_USAGE"
            )


    # --------------------------------------------------------------
    # Cross-domain anomaly
    # --------------------------------------------------------------

    if (
        correlation[
            "anomalous_count"
        ]
        >= 3
    ):

        minimum_status = _max_guardian_status(
            minimum_status,
            GuardianStatus.ALERT,
        )

        minimum_severity = _max_severity(
            minimum_severity,
            Severity.HIGH,
        )

        reason_codes.append(
            "CROSS_DOMAIN_BEHAVIORAL_DEVIATION"
        )


    return {
        "minimum_guardian_status": (
            minimum_status.value
        ),

        "minimum_severity": (
            minimum_severity.value
        ),

        "mandatory_reason_codes": sorted(
            set(
                reason_codes
            )
        ),
    }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def build_guardian_correlation_task(
    agent: Agent,
    state: GuardianGraphState,
) -> Task:
    """
    Build the CrewAI supervisor task.
    """

    correlation = build_correlation_evidence(
        state
    )

    guardrails = determine_minimum_disposition(
        state=state,
        correlation=correlation,
    )

    specialist_payload = (
        _serialize_specialist_results(
            state
        )
    )

    telemetry = state.get(
        "telemetry"
    )

    run_id = state.get(
        "run_id"
    )

    monitored_agent = state.get(
        "monitored_agent"
    )

    correlation_json = json.dumps(
        correlation,
        indent=2,
    )

    guardrail_json = json.dumps(
        guardrails,
        indent=2,
    )

    specialists_json = json.dumps(
        specialist_payload,
        indent=2,
    )

    missing_telemetry = []

    if isinstance(
        telemetry,
        NormalizedTelemetry,
    ):
        missing_telemetry = (
            telemetry.missing_evidence
        )

    description = f"""
You are the Guardian Correlator.

You are supervising five independent security-observability analysts.

Your task is to determine the overall MONITORING disposition for one
Glean agent run.

Final allowed states:

    NORMAL
    REVIEW
    ALERT

You perform NO enforcement.

============================================================
RUN
============================================================

run_id:

    {run_id}

monitored agent:

    {monitored_agent}

============================================================
SPECIALIST RESULTS
============================================================

{specialists_json}

============================================================
DETERMINISTIC CORRELATION
============================================================

{correlation_json}

============================================================
DETERMINISTIC GUARDRAILS
============================================================

{guardrail_json}

These guardrails were produced by deterministic Python.

They are mandatory.

You MAY escalate above the minimum.

You MUST NOT downgrade below the minimum.

For example:

    minimum_guardian_status = REVIEW

means:

    REVIEW or ALERT are allowed

but:

    NORMAL is forbidden.

Likewise:

    minimum_guardian_status = ALERT

means:

    final guardian_status MUST be ALERT.

============================================================
CORRELATION OBJECTIVE
============================================================

Answer:

    "Taken together, what does the available evidence mean?"

You must correlate evidence across:

1. Run Behavior
2. Tool & Action Behavior
3. Data Access
4. Output Behavior
5. Glean Protect Findings

Do not simply count votes.

SECURITY IS NOT MAJORITY VOTING.

Example:

    Run Spy       NORMAL
    Data Spy      NORMAL
    Output Spy    NORMAL
    Protect Spy   NORMAL
    Tool Spy      HIGH / unexpected SEND_EMAIL

The result must NOT become NORMAL merely because four analysts were
normal.

One decisive fact can dominate a security decision.

============================================================
NORMAL
============================================================

Use NORMAL only when:

- no material anomalies exist
- no specialist errors exist
- no important evidence is missing
- no unexpected write activity exists
- no meaningful Protect finding exists
- specialist results are coherent
- evidence is sufficiently complete

NORMAL requires broad positive confidence.

============================================================
REVIEW
============================================================

Use REVIEW when:

- evidence is incomplete
- a specialist failed
- analysts materially disagree
- anomaly exists but impact is uncertain
- unexpected MCP/read behavior requires investigation
- moderate or isolated high-risk behavior exists
- missing provenance prevents confident NORMAL classification

REVIEW means:

    "Security should inspect this run."

It does NOT mean:

    "An attack definitely occurred."

============================================================
ALERT
============================================================

Use ALERT when:

- major behavioral deviation occurred
- prohibited write capability was invoked
- multiple independent HIGH signals agree
- Glean Protect blocked important malicious or unsafe behavior
- several monitoring domains show correlated anomalies
- compelling evidence warrants immediate security attention

ALERT still does NOT prove compromise.

============================================================
PROTECT BLOCK RULE
============================================================

If Protect reports:

    action = BLOCK

preserve this distinction:

    dangerous behavior detected
    +
    native security control intervened

Do NOT say:

    "The control failed"

unless independent evidence actually demonstrates failure.

A blocked malicious action may demonstrate successful control operation.

============================================================
EVIDENCE QUALITY
============================================================

Telemetry missing evidence:

{json.dumps(missing_telemetry, indent=2)}

Specialist errors:

{json.dumps(correlation["specialist_errors"], indent=2)}

Missing evidence must affect confidence.

Never turn missing evidence into NORMAL by assumption.

============================================================
CONFLICT HANDLING
============================================================

Specialists can legitimately disagree because they evaluate different
domains.

Example:

    Run Spy:
        NORMAL

    Tool Spy:
        ANOMALOUS / HIGH

This is not necessarily a contradiction.

The workflow shape may look normal while one tool action is dangerous.

True conflict exists when two analysts interpret the SAME evidence in
materially incompatible ways.

Preserve nuance.

============================================================
SEVERITY
============================================================

Final severity should represent the combined security significance.

It does not have to equal the highest individual severity, but must not
casually downgrade strong evidence.

Examples:

Multiple independent HIGH anomalies:

    generally HIGH

One minor sentence-count deviation:

    generally LOW

Critical specialist finding:

    cannot become INFO or LOW

============================================================
REASON CODES
============================================================

Include concise machine-friendly reason codes.

Examples:

    UNEXPECTED_WRITE_CAPABILITY

    UNEXPECTED_MCP_USAGE

    DATA_SCOPE_EXPANSION

    OUTPUT_CONTRACT_VIOLATION

    PROTECT_BLOCK

    PROMPT_INJECTION_FINDING

    TELEMETRY_INCOMPLETE

    SPECIALIST_EXECUTION_FAILURE

    CROSS_DOMAIN_BEHAVIORAL_DEVIATION

Preserve all mandatory reason codes from the deterministic guardrails.

============================================================
RECOMMENDED ACTIONS
============================================================

Recommended actions may include:

    SECURITY_REVIEW

    INSPECT_WORKFLOW_TRACE

    INSPECT_TOOL_EXECUTION

    REVIEW_DATA_ACCESS

    REVIEW_PROTECT_FINDING

    VALIDATE_AGENT_CONFIGURATION

    VALIDATE_TELEMETRY_PIPELINE

Do NOT recommend destructive actions automatically.

This Demo #2 is monitoring only.

============================================================
IMPORTANT PRINCIPLES
============================================================

ANOMALY != COMPROMISE

DETECTION != ENFORCEMENT

PROTECT BLOCK != CONTROL FAILURE

MISSING EVIDENCE != SAFE

NORMAL REQUIRES COHERENT EVIDENCE

ONE DECISIVE SECURITY FACT CAN OUTWEIGH MULTIPLE NORMAL OBSERVATIONS

============================================================
OUTPUT
============================================================

Return a GuardianDecision.

run_id MUST equal:

    {run_id}

monitored_agent MUST equal:

    {monitored_agent}

enforcement_performed MUST be:

    false

Populate:

- guardian_status
- severity
- confidence
- summary
- reason_codes
- findings
- recommended_actions
- security_review_required
- evidence_complete
- missing_evidence

The final guardian_status must NOT be lower than:

    {guardrails["minimum_guardian_status"]}

The final severity must NOT be lower than:

    {guardrails["minimum_severity"]}
"""

    return Task(
        description=description,

        expected_output=(
            "A validated GuardianDecision correlating all available "
            "specialist security evidence for the monitored Glean run."
        ),

        agent=agent,

        output_pydantic=GuardianDecision,
    )


# ---------------------------------------------------------------------------
# Main correlation execution
# ---------------------------------------------------------------------------


def correlate_guardian_findings(
    state: GuardianGraphState,
) -> GuardianDecision:
    """
    Run the Guardian Correlator and enforce deterministic post-validation.
    """

    agent = build_guardian_correlator_agent()

    task = build_guardian_correlation_task(
        agent=agent,
        state=state,
    )

    correlation = build_correlation_evidence(
        state
    )

    guardrails = determine_minimum_disposition(
        state=state,
        correlation=correlation,
    )

    crew = Crew(
        agents=[
            agent,
        ],

        tasks=[
            task,
        ],

        process=Process.sequential,

        verbose=_env_bool(
            "CREWAI_VERBOSE",
            default=False,
        ),
    )

    try:

        result = crew.kickoff()

    except Exception as exc:

        raise RuntimeError(
            "Guardian Correlator CrewAI execution failed."
        ) from exc


    decision = _extract_pydantic_result(
        result=result,
        task=task,
        expected_type=GuardianDecision,
    )


    # --------------------------------------------------------------
    # Hard post-validation
    #
    # Never assume the LLM followed disposition floors correctly.
    # --------------------------------------------------------------

    decision = enforce_guardrails(
        decision=decision,
        guardrails=guardrails,
        state=state,
    )


    return decision


# ---------------------------------------------------------------------------
# Deterministic post-processing
# ---------------------------------------------------------------------------


def enforce_guardrails(
    decision: GuardianDecision,
    guardrails: dict[str, Any],
    state: GuardianGraphState,
) -> GuardianDecision:
    """
    Ensure the LLM cannot downgrade below deterministic policy.

    This is intentionally deterministic.
    """

    minimum_status = GuardianStatus(
        guardrails[
            "minimum_guardian_status"
        ]
    )

    minimum_severity = Severity(
        guardrails[
            "minimum_severity"
        ]
    )


    # --------------------------------------------------------------
    # Status floor
    # --------------------------------------------------------------

    if (
        GUARDIAN_STATUS_ORDER[
            decision.guardian_status
        ]
        <
        GUARDIAN_STATUS_ORDER[
            minimum_status
        ]
    ):
        decision.guardian_status = (
            minimum_status
        )


    # --------------------------------------------------------------
    # Severity floor
    # --------------------------------------------------------------

    if (
        SEVERITY_ORDER[
            decision.severity
        ]
        <
        SEVERITY_ORDER[
            minimum_severity
        ]
    ):
        decision.severity = (
            minimum_severity
        )


    # --------------------------------------------------------------
    # Mandatory reason codes
    # --------------------------------------------------------------

    reason_codes = set(
        decision.reason_codes
    )

    reason_codes.update(
        guardrails[
            "mandatory_reason_codes"
        ]
    )

    decision.reason_codes = sorted(
        reason_codes
    )


    # --------------------------------------------------------------
    # Demo #2 cannot enforce
    # --------------------------------------------------------------

    decision.enforcement_performed = False


    # --------------------------------------------------------------
    # REVIEW / ALERT implies human/security attention
    # --------------------------------------------------------------

    decision.security_review_required = (
        decision.guardian_status
        in {
            GuardianStatus.REVIEW,
            GuardianStatus.ALERT,
        }
    )


    # --------------------------------------------------------------
    # Evidence completeness
    # --------------------------------------------------------------

    telemetry = state.get(
        "telemetry"
    )

    specialist_errors = (
        collect_specialist_errors(
            state
        )
    )

    telemetry_complete = (
        telemetry.evidence_complete
        if isinstance(
            telemetry,
            NormalizedTelemetry,
        )
        else False
    )

    decision.evidence_complete = (
        telemetry_complete
        and not specialist_errors
    )


    # --------------------------------------------------------------
    # Missing evidence
    # --------------------------------------------------------------

    missing = set(
        decision.missing_evidence
    )

    if isinstance(
        telemetry,
        NormalizedTelemetry,
    ):
        missing.update(
            telemetry.missing_evidence
        )

    for specialist_name in specialist_errors:
        missing.add(
            f"SPECIALIST_ERROR:{specialist_name}"
        )

    decision.missing_evidence = sorted(
        missing
    )


    return decision


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def guardian_correlator_node(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    LangGraph-compatible Guardian correlation node.

    Reads all specialist results.

    Writes:

        guardian_decision

        phase
    """

    try:

        decision = correlate_guardian_findings(
            state
        )

        return {
            "guardian_decision": decision,
        }

    except Exception as exc:

        # Correlator failure itself is a graph-level problem.
        #
        # guardian_graph.py can decide how to finalize this state.
        return {
            "guardian_decision": _build_correlator_failure_decision(
                state=state,
                error=str(
                    exc
                ),
            )
        }


# ---------------------------------------------------------------------------
# Correlator failure fallback
# ---------------------------------------------------------------------------


def _build_correlator_failure_decision(
    state: GuardianGraphState,
    error: str,
) -> GuardianDecision:
    """
    Safe deterministic fallback if the CrewAI correlator fails.

    A correlation failure can NEVER become NORMAL.
    """

    run_id = str(
        state.get(
            "run_id",
            "UNKNOWN_RUN",
        )
    )

    monitored_agent = str(
        state.get(
            "monitored_agent",
            "UNKNOWN_AGENT",
        )
    )

    return GuardianDecision(
        run_id=run_id,

        monitored_agent=monitored_agent,

        guardian_status=GuardianStatus.REVIEW,

        severity=Severity.MEDIUM,

        confidence=1.0,

        summary=(
            "Guardian specialist evidence could not be fully correlated "
            "because the correlation layer failed. Security review is "
            "required before this run can be considered normal."
        ),

        reason_codes=[
            "GUARDIAN_CORRELATION_FAILURE"
        ],

        findings=[],

        recommended_actions=[
            "SECURITY_REVIEW",
            "VALIDATE_GUARDIAN_CORRELATOR",
        ],

        security_review_required=True,

        evidence_complete=False,

        missing_evidence=[
            f"CORRELATOR_ERROR:{error}"
        ],

        enforcement_performed=False,
    )


# ---------------------------------------------------------------------------
# Specialist serialization
# ---------------------------------------------------------------------------


def _serialize_specialist_results(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Produce a clean prompt payload for the Guardian Correlator.
    """

    mapping = {
        "run_behavior": (
            state.get(
                "run_behavior_analysis"
            )
        ),

        "tool_behavior": (
            state.get(
                "tool_behavior_analysis"
            )
        ),

        "data_access": (
            state.get(
                "data_access_analysis"
            )
        ),

        "output_behavior": (
            state.get(
                "output_behavior_analysis"
            )
        ),

        "protect_findings": (
            state.get(
                "protect_findings_analysis"
            )
        ),
    }


    serialized: dict[
        str,
        Any
    ] = {}


    for name, analysis in mapping.items():

        if analysis is None:

            serialized[
                name
            ] = None

            continue


        serialized[
            name
        ] = analysis.model_dump(
            mode="json",
            exclude_none=True,
        )


    return serialized


# ---------------------------------------------------------------------------
# Specialist extraction
# ---------------------------------------------------------------------------


def _get_specialist_analyses(
    state: GuardianGraphState,
) -> list[SpyAnalysis]:
    """
    Return all successfully completed specialist analyses.
    """

    candidates = [
        state.get(
            "run_behavior_analysis"
        ),

        state.get(
            "tool_behavior_analysis"
        ),

        state.get(
            "data_access_analysis"
        ),

        state.get(
            "output_behavior_analysis"
        ),

        state.get(
            "protect_findings_analysis"
        ),
    ]


    return [
        analysis
        for analysis in candidates
        if isinstance(
            analysis,
            SpyAnalysis,
        )
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains_high_anomaly(
    state: GuardianGraphState,
) -> bool:
    """
    Determine whether any specialist produced an anomalous HIGH or
    CRITICAL result.
    """

    for analysis in _get_specialist_analyses(
        state
    ):

        if (
            analysis.status
            == AnalysisStatus.ANOMALOUS
            and
            SEVERITY_ORDER[
                analysis.severity
            ]
            >= SEVERITY_ORDER[
                Severity.HIGH
            ]
        ):
            return True

    return False


def _highest_severity(
    severities: list[Severity],
) -> Severity:
    """
    Return highest specialist severity.
    """

    if not severities:
        return Severity.INFO

    return max(
        severities,
        key=lambda severity: (
            SEVERITY_ORDER[
                severity
            ]
        ),
    )


def _max_severity(
    left: Severity,
    right: Severity,
) -> Severity:

    if (
        SEVERITY_ORDER[
            right
        ]
        >
        SEVERITY_ORDER[
            left
        ]
    ):
        return right

    return left


def _max_guardian_status(
    left: GuardianStatus,
    right: GuardianStatus,
) -> GuardianStatus:

    if (
        GUARDIAN_STATUS_ORDER[
            right
        ]
        >
        GUARDIAN_STATUS_ORDER[
            left
        ]
    ):
        return right

    return left


# ---------------------------------------------------------------------------
# Structured output extraction
# ---------------------------------------------------------------------------


def _extract_pydantic_result(
    result: Any,
    task: Task,
    expected_type: type[GuardianDecision],
) -> GuardianDecision:
    """
    Accept only validated CrewAI output.
    """

    crew_pydantic = getattr(
        result,
        "pydantic",
        None,
    )

    if isinstance(
        crew_pydantic,
        expected_type,
    ):
        return crew_pydantic


    task_output = getattr(
        task,
        "output",
        None,
    )

    if task_output is not None:

        task_pydantic = getattr(
            task_output,
            "pydantic",
            None,
        )

        if isinstance(
            task_pydantic,
            expected_type,
        ):
            return task_pydantic


    json_dict = getattr(
        result,
        "json_dict",
        None,
    )

    if isinstance(
        json_dict,
        dict,
    ):

        try:

            return expected_type.model_validate(
                json_dict
            )

        except ValidationError:
            pass


    raise RuntimeError(
        "Guardian Correlator did not return a valid "
        "GuardianDecision Pydantic object."
    )


# ---------------------------------------------------------------------------
# Environment helper
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
