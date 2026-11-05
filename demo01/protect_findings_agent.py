"""
Guardian Telemetry Swarm
protect_findings_agent.py

CrewAI specialist responsible for interpreting security findings produced
by Glean Protect or another Glean-native AI security control.

Primary question:

    "Did Glean's own security controls identify anything relevant to
     this monitored agent run?"

This analyst evaluates:

- Protect findings associated with the run
- Protect policy names
- Protect severity
- Protect actions such as FLAG or BLOCK
- finding source/origin
- multiple correlated Protect findings
- whether native Glean security evidence materially increases concern

It does NOT:

- independently re-detect prompt injection
- independently scan for malicious code
- re-run DLP analysis
- judge document access scope
- judge tool authorization
- make the final Guardian NORMAL / REVIEW / ALERT decision
- claim that a Protect finding proves compromise

Architecture:

Glean Protect
      |
      v
ProtectFinding[]
      |
      v
Deterministic finding summary
      |
      v
Protect Findings Analyst
      |
      v
ProtectFindingsAnalysis
      |
      v
LangGraph
"""

from __future__ import annotations

import json
import os
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import ValidationError

from models import (
    AgentBaseline,
    NormalizedTelemetry,
    ProtectAction,
    ProtectFindingsAnalysis,
    Severity,
)


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------


def build_llm() -> LLM:
    """
    Construct the LLM used by the Protect Findings Analyst.
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
# Agent
# ---------------------------------------------------------------------------


def build_protect_findings_agent() -> Agent:
    """
    Construct the CrewAI Protect Findings Analyst.
    """

    return Agent(
        role="Protect Findings Analyst",

        goal=(
            "Interpret Glean-native AI security findings associated with "
            "a monitored agent run and determine whether those findings "
            "represent meaningful independent security evidence."
        ),

        backstory=(
            "You are a security observability analyst specializing in "
            "enterprise AI platform security findings. You do not attempt "
            "to replace Glean Protect. You consume Protect findings as an "
            "independent source of evidence, preserve their stated severity "
            "and action, and explain how much Guardian attention they "
            "deserve without overstating what the findings prove."
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
# Deterministic evidence preparation
# ---------------------------------------------------------------------------


def build_protect_evidence(
    baseline: AgentBaseline,
    telemetry: NormalizedTelemetry,
) -> dict[str, Any]:
    """
    Deterministically summarize Glean Protect findings.

    Python determines:

    - how many findings exist
    - highest reported severity
    - whether Glean BLOCKed anything
    - whether Glean FLAGged anything
    - which policies fired
    - which sources produced findings

    CrewAI interprets the significance.

    IMPORTANT:

    Glean Protect findings remain evidence.

    They are NOT automatically treated as proof of compromise.
    """

    findings = telemetry.protect_findings

    protect_findings_present = bool(
        findings
    )

    severity_values = [
        finding.severity
        for finding in findings
    ]

    highest_severity = _highest_severity(
        severity_values
    )

    blocked_findings = [
        finding
        for finding in findings
        if finding.action == ProtectAction.BLOCK
    ]

    flagged_findings = [
        finding
        for finding in findings
        if finding.action == ProtectAction.FLAG
    ]

    allowed_findings = [
        finding
        for finding in findings
        if finding.action == ProtectAction.ALLOW
    ]

    unknown_action_findings = [
        finding
        for finding in findings
        if finding.action == ProtectAction.UNKNOWN
    ]

    policy_names = sorted(
        {
            finding.policy_name
            for finding in findings
        }
    )

    sources = sorted(
        {
            finding.source
            for finding in findings
            if finding.source
        }
    )

    descriptions_present = sum(
        1
        for finding in findings
        if finding.description
    )

    protect_specific_missing = sorted(
        {
            item
            for item in telemetry.missing_evidence
            if (
                "PROTECT" in item.upper()
                or "SECURITY_FINDING" in item.upper()
            )
        }
    )

    # This assumes the retrieval/normalization layer marks Protect
    # telemetry unavailable when it was not collected.
    #
    # An empty findings list is therefore allowed to mean "no findings"
    # only when there is no explicit Protect evidence gap.
    protect_evidence_available = (
        len(protect_specific_missing) == 0
    )

    return {
        "protect_evidence_available": (
            protect_evidence_available
        ),

        "protect_findings_present": (
            protect_findings_present
        ),

        "finding_count": (
            len(findings)
        ),

        "highest_severity": (
            highest_severity.value
        ),

        "block_present": (
            bool(blocked_findings)
        ),

        "block_count": (
            len(blocked_findings)
        ),

        "flag_present": (
            bool(flagged_findings)
        ),

        "flag_count": (
            len(flagged_findings)
        ),

        "allow_count": (
            len(allowed_findings)
        ),

        "unknown_action_count": (
            len(unknown_action_findings)
        ),

        "policy_names": (
            policy_names
        ),

        "finding_sources": (
            sources
        ),

        "descriptions_present": (
            descriptions_present
        ),

        "missing_protect_evidence": (
            protect_specific_missing
        ),
    }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def build_protect_findings_task(
    agent: Agent,
    baseline: AgentBaseline,
    telemetry: NormalizedTelemetry,
) -> Task:
    """
    Create the CrewAI Protect Findings analysis task.
    """

    protect_evidence = build_protect_evidence(
        baseline=baseline,
        telemetry=telemetry,
    )

    baseline_json = baseline.model_dump_json(
        indent=2,
        exclude_none=True,
    )

    telemetry_json = telemetry.model_dump_json(
        indent=2,
        exclude_none=True,
    )

    evidence_json = json.dumps(
        protect_evidence,
        indent=2,
    )

    description = f"""
Analyze Glean Protect security findings for one monitored agent run.

You are the Protect Findings Analyst.

Your purpose is to interpret NATIVE GLEAN SECURITY FINDINGS.

You are not attempting to replace Glean Protect.

You MUST NOT independently re-detect:

- prompt injection
- malicious code
- data loss
- tool misalignment
- unsafe content

Those detections belong to Glean Protect or other specialist systems.

Your task is to interpret the findings that already exist.

============================================================
MONITORED AGENT BASELINE
============================================================

{baseline_json}

============================================================
NORMALIZED TELEMETRY
============================================================

{telemetry_json}

============================================================
DETERMINISTIC PROTECT SUMMARY
============================================================

{evidence_json}

The deterministic summary above was produced by Python.

Treat these counts and classifications as authoritative.

============================================================
ANALYSIS OBJECTIVE
============================================================

Determine whether Glean-native security findings provide independent
evidence that should increase Guardian attention for this run.

Evaluate the following areas.

------------------------------------------------------------
1. NO PROTECT FINDINGS
------------------------------------------------------------

If Protect telemetry is available and:

    finding_count = 0

then there is no native Protect finding associated with the available
evidence.

This does NOT mean:

    the run is definitely safe

It means only:

    no Protect finding was observed.

Other Guardian analysts may still identify anomalous behavior.

Do not override another specialist merely because Protect found nothing.

------------------------------------------------------------
2. FLAGGED FINDINGS
------------------------------------------------------------

A finding with:

    action = FLAG

means Glean's native security layer identified behavior worth recording
or reviewing.

Preserve:

- policy name
- severity
- source
- stated action

Do not rewrite a FLAG as a BLOCK.

Do not state that the flagged behavior definitely succeeded.

------------------------------------------------------------
3. BLOCKED FINDINGS
------------------------------------------------------------

A finding with:

    action = BLOCK

is especially important.

It indicates Glean's security control took preventive action against
the behavior represented by the finding.

However:

    BLOCKED ATTACK
        !=
    CONTROL FAILURE

In fact, successful blocking may demonstrate that a control worked.

Your analysis must distinguish:

    dangerous behavior observed

from:

    dangerous behavior successfully executed

and:

    native control successfully blocked behavior

when the evidence supports those distinctions.

------------------------------------------------------------
4. SEVERITY
------------------------------------------------------------

Respect the severity reported by Protect.

Do not casually downgrade:

    HIGH

to:

    LOW

and do not automatically promote:

    MEDIUM

to:

    CRITICAL.

The highest Protect severity is supplied deterministically.

Use it as strong input to your own assessment.

------------------------------------------------------------
5. MULTIPLE FINDINGS
------------------------------------------------------------

Multiple independent Protect findings in one run may increase concern.

Example:

    Prompt Injection      HIGH
    Tool Misalignment     HIGH
    Unsafe Content        MEDIUM

is materially different from one isolated LOW finding.

Do not merely count findings.

Consider whether the findings represent related security behavior.

------------------------------------------------------------
6. FINDING SOURCE
------------------------------------------------------------

A finding may identify a source such as:

    user prompt
    retrieved content
    tool call
    model response
    external content

Source matters.

Example:

    Prompt injection detected in retrieved content

does NOT necessarily mean:

    the user attempted prompt injection.

Preserve the stated provenance.

------------------------------------------------------------
7. PROTECT + GUARDIAN INDEPENDENCE
------------------------------------------------------------

Glean Protect and Guardian are separate signals.

This analyst must NOT say:

    "Guardian detected..."

unless the supplied Protect evidence actually came from Guardian.

Likewise, Protect having no findings does not invalidate:

    Run Behavior findings
    Tool Behavior findings
    Data Access findings
    Output Behavior findings

The final Guardian Correlator handles agreement and disagreement.

------------------------------------------------------------
8. MISSING PROTECT EVIDENCE
------------------------------------------------------------

If Protect telemetry was not collected, is unavailable, or cannot be
associated with this run:

    do NOT conclude NORMAL.

Use:

    INSUFFICIENT_EVIDENCE

when Protect evidence availability cannot be established.

Important distinction:

    no findings
        !=
    findings telemetry unavailable

============================================================
STATUS DEFINITIONS
============================================================

NORMAL

Use NORMAL when:

- Protect evidence is available
- no meaningful Protect findings are present

A NORMAL result means only that this independent security source did
not report a meaningful finding.

It does not certify the entire run as safe.

------------------------------------------------------------

ANOMALOUS

Use ANOMALOUS when:

- one or more meaningful Protect findings exist
- Protect flagged or blocked behavior relevant to the run
- native security controls identified significant abnormal behavior

------------------------------------------------------------

INSUFFICIENT_EVIDENCE

Use when:

- Protect evidence is unavailable
- telemetry association is incomplete
- evidence cannot establish whether Protect evaluated the run

============================================================
SEVERITY GUIDANCE
============================================================

INFO

Protect evidence available and no meaningful findings.

------------------------------------------------------------

LOW

Low-severity Protect finding with limited security significance.

------------------------------------------------------------

MEDIUM

Meaningful Protect finding requiring investigation.

------------------------------------------------------------

HIGH

High-severity Protect finding or significant native security alert.

------------------------------------------------------------

CRITICAL

Use only when Protect reports compelling critical-severity evidence
or equivalent extraordinary behavior.

Do not manufacture CRITICAL severity.

============================================================
CONTROL-SUCCESS PRINCIPLE
============================================================

This distinction is mandatory.

Example:

    Protect finding:
        Prompt Injection

    severity:
        HIGH

    action:
        BLOCK

Correct interpretation:

    "High-severity prompt injection behavior was identified and
     Glean Protect blocked the affected operation."

Incorrect interpretation:

    "The attacker successfully compromised the agent."

Another incorrect interpretation:

    "The security system failed."

A block can represent successful defensive control behavior.

============================================================
IMPORTANT SECURITY PRINCIPLES
============================================================

PROTECT FINDING != CONFIRMED COMPROMISE

PROTECT BLOCK != CONTROL FAILURE

NO PROTECT FINDING != SAFE RUN

NO TELEMETRY != NO FINDINGS

FLAG != BLOCK

DETECTION != EXECUTION

SOURCE != ACTOR

NATIVE SECURITY SIGNAL != FINAL GUARDIAN DECISION

============================================================
CONFIDENCE
============================================================

Confidence must be between:

    0.0 and 1.0

It represents confidence in your interpretation of the Protect evidence.

It does not represent probability of compromise.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return a ProtectFindingsAnalysis object.

The run_id MUST be:

    {telemetry.run.run_id}

The analyst field must remain:

    Protect Findings Analyst

Populate:

- status
- severity
- confidence
- summary
- findings
- evidence_reviewed
- missing_evidence
- requires_guardian_attention
- protect_findings_present
- protect_block_present
- highest_protect_severity

protect_findings_present must reflect whether actual Protect findings
were observed.

protect_block_present must reflect whether any finding has:

    action = BLOCK

highest_protect_severity must reflect the deterministic evidence.

Do not invent Protect policies.

Do not invent findings.

Do not claim a finding came from Glean Protect unless it exists in the
provided telemetry.
"""

    return Task(
        description=description,

        expected_output=(
            "A validated ProtectFindingsAnalysis interpreting native "
            "Glean Protect findings associated with the monitored run."
        ),

        agent=agent,

        output_pydantic=ProtectFindingsAnalysis,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def analyze_protect_findings(
    baseline: AgentBaseline,
    telemetry: NormalizedTelemetry,
) -> ProtectFindingsAnalysis:
    """
    Execute the Protect Findings Analyst.
    """

    agent = build_protect_findings_agent()

    task = build_protect_findings_task(
        agent=agent,
        baseline=baseline,
        telemetry=telemetry,
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
            "Protect Findings Analyst CrewAI execution failed."
        ) from exc

    return _extract_pydantic_result(
        result=result,
        task=task,
        expected_type=ProtectFindingsAnalysis,
    )


# ---------------------------------------------------------------------------
# Structured output extraction
# ---------------------------------------------------------------------------


def _extract_pydantic_result(
    result: Any,
    task: Task,
    expected_type: type[ProtectFindingsAnalysis],
) -> ProtectFindingsAnalysis:
    """
    Accept only validated structured CrewAI output.
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
        "Protect Findings Analyst did not return a valid "
        "ProtectFindingsAnalysis Pydantic object."
    )


# ---------------------------------------------------------------------------
# LangGraph node adapter
# ---------------------------------------------------------------------------


def protect_findings_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    LangGraph-compatible Protect Findings Spy node.

    Reads:

        baseline
        telemetry

    Writes ONLY:

        protect_findings_analysis

    OR:

        protect_findings_error
    """

    baseline = state.get(
        "baseline"
    )

    telemetry = state.get(
        "telemetry"
    )

    if not isinstance(
        baseline,
        AgentBaseline,
    ):
        return {
            "protect_findings_error": (
                "Guardian graph state does not contain "
                "a valid AgentBaseline."
            )
        }

    if not isinstance(
        telemetry,
        NormalizedTelemetry,
    ):
        return {
            "protect_findings_error": (
                "Guardian graph state does not contain "
                "valid NormalizedTelemetry."
            )
        }

    try:

        analysis = analyze_protect_findings(
            baseline=baseline,
            telemetry=telemetry,
        )

        return {
            "protect_findings_analysis": analysis
        }

    except Exception as exc:

        return {
            "protect_findings_error": str(
                exc
            )
        }


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def _highest_severity(
    severities: list[Severity],
) -> Severity:
    """
    Return the highest observed Protect severity.

    If there are no findings, INFO is appropriate because there is
    nothing to rank.
    """

    if not severities:
        return Severity.INFO

    return max(
        severities,
        key=lambda severity: SEVERITY_ORDER[
            severity
        ],
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
