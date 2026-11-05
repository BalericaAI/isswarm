"""
Guardian Telemetry Swarm
run_behavior_agent.py

CrewAI specialist responsible for analyzing the execution shape of a
monitored Glean agent run.

This analyst answers:

    "Did this agent execute in the way its behavioral contract says
     it should?"

It evaluates:

- initiator
- workflow execution shape
- expected vs observed steps
- unexpected workflow steps
- missing expected steps
- errors
- retries / unusual execution patterns
- unexpected nested workflow behavior
- basic execution drift

It does NOT evaluate:

- document sensitivity
- whether accessed data was appropriate
- destination risk
- policy compliance
- whether a specific tool is inherently dangerous
- Glean Protect findings
- final NORMAL / REVIEW / ALERT Guardian disposition

Those responsibilities belong to other specialists and the Guardian
Correlator.
"""

from __future__ import annotations

import os
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import ValidationError

from models import (
    AgentBaseline,
    AnalysisStatus,
    NormalizedTelemetry,
    RunBehaviorAnalysis,
    Severity,
)


# ---------------------------------------------------------------------------
# CrewAI configuration
# ---------------------------------------------------------------------------


def build_llm() -> LLM:
    """
    Create the LLM used by the Run Behavior Analyst.

    The model can be changed without modifying application code:

        CREWAI_MODEL=openai/gpt-4.1-mini

    or another CrewAI-supported model/provider.
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


def build_run_behavior_agent() -> Agent:
    """
    Construct the CrewAI Run Behavior Analyst.
    """

    return Agent(
        role="Run Behavior Analyst",

        goal=(
            "Determine whether a monitored Glean agent executed according "
            "to its declared behavioral baseline, using only the supplied "
            "normalized telemetry and baseline evidence."
        ),

        backstory=(
            "You are a security observability analyst specializing in "
            "enterprise autonomous-agent execution behavior. "
            "You inspect workflow structure, execution state, initiators, "
            "step sequences, failures, retries, and execution drift. "
            "You do not speculate about compromise and you do not make "
            "final enforcement decisions."
        ),

        llm=build_llm(),

        verbose=_env_bool(
            "CREWAI_VERBOSE",
            default=False,
        ),

        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def build_run_behavior_task(
    agent: Agent,
    baseline: AgentBaseline,
    telemetry: NormalizedTelemetry,
) -> Task:
    """
    Create the CrewAI task using already validated Pydantic evidence.
    """

    baseline_json = baseline.model_dump_json(
        indent=2,
        exclude_none=True,
    )

    telemetry_json = telemetry.model_dump_json(
        indent=2,
        exclude_none=True,
    )

    description = f"""
Analyze the execution behavior of one monitored Glean agent run.

You MUST evaluate execution behavior only.

Do not perform tool-risk analysis.
Do not perform data-sensitivity analysis.
Do not perform destination-risk analysis.
Do not evaluate company policy.
Do not issue a final Guardian NORMAL / REVIEW / ALERT decision.

============================================================
MONITORED AGENT BASELINE
============================================================

{baseline_json}

============================================================
NORMALIZED TELEMETRY
============================================================

{telemetry_json}

============================================================
ANALYSIS OBJECTIVE
============================================================

Determine whether the observed execution shape conforms to the
declared behavioral baseline.

Evaluate the following:

1. INITIATOR

Compare:

    telemetry.run.initiator

against:

    baseline.allowed_initiators

If the initiator is unexpected, record that as execution drift.

Do not automatically call the run malicious.

------------------------------------------------------------

2. EXPECTED WORKFLOW STEPS

Compare the expected workflow steps from:

    baseline.expected_steps

against the observed:

    telemetry.steps

Identify:

- expected steps that occurred
- expected steps that did not occur
- unexpected steps
- suspiciously expanded workflow execution
- evidence of unexpected nested or additional workflow behavior

Do not assume that different internal step IDs automatically indicate
malicious behavior.

Distinguish semantic deviation from harmless naming differences.

------------------------------------------------------------

3. EXECUTION STATUS

Review:

- overall run status
- individual step status
- failed steps
- cancelled steps
- errors
- repeated/unusual execution behavior if visible

A failed step is not automatically a security incident.

However, unusual execution patterns should be noted when they materially
depart from the baseline.

------------------------------------------------------------

4. EXECUTION COMPLEXITY

The Five Sentence Document Reviewer is intentionally simple.

Its expected logical execution is approximately:

    Input Form
        ->
    Read Document
        ->
    Respond

Unexpected expansion such as:

    Input
        ->
    Read Document
        ->
    Search
        ->
    Another Workflow
        ->
    Additional Processing
        ->
    Respond

may represent behavioral drift.

Do NOT decide whether the tools themselves are dangerous.
The Tool & Action Analyst handles that.

------------------------------------------------------------

5. EVIDENCE QUALITY

If required telemetry is unavailable, do not guess.

Use:

    INSUFFICIENT_EVIDENCE

when the evidence does not support a defensible behavioral conclusion.

Missing telemetry is not proof that the run behaved normally.

============================================================
STATUS DEFINITIONS
============================================================

NORMAL

Use NORMAL when the execution behavior is materially consistent with
the declared baseline and there is sufficient evidence to make that
determination.

ANOMALOUS

Use ANOMALOUS when observed execution behavior materially deviates from
the declared baseline.

Examples:

- unauthorized initiator
- unexpected workflow stages
- unexplained nested workflow activity
- significant execution expansion
- substantial failure/retry behavior inconsistent with the baseline

INSUFFICIENT_EVIDENCE

Use INSUFFICIENT_EVIDENCE when telemetry is too incomplete to determine
whether the run conformed to the baseline.

============================================================
SEVERITY GUIDANCE
============================================================

INFO

Normal execution with no meaningful deviations.

LOW

Minor execution irregularity with little apparent security significance.

MEDIUM

Meaningful workflow deviation requiring investigation.

HIGH

Major execution drift, unexpected workflow expansion, or behavior strongly
inconsistent with the declared agent function.

CRITICAL

Reserve this for extraordinary execution behavior with compelling evidence
of major control failure.

Do not use CRITICAL merely because something looks unusual.

============================================================
CONFIDENCE
============================================================

Confidence must be between:

    0.0 and 1.0

Confidence represents confidence in your assessment based on available
evidence.

It is not a probability that the agent has been compromised.

============================================================
IMPORTANT SECURITY RULES
============================================================

OBSERVATION != JUDGMENT

ANOMALY != COMPROMISE

FAILED STEP != SECURITY INCIDENT

MISSING EVIDENCE != NORMAL

UNEXPECTED STEP != MALICIOUS STEP

You are identifying execution deviation.

You are NOT deciding whether an attack occurred.

============================================================
OUTPUT
============================================================

Return a RunBehaviorAnalysis object.

The run_id MUST match:

    {telemetry.run.run_id}

The analyst field must remain:

    Run Behavior Analyst

Populate:

- status
- severity
- confidence
- summary
- findings
- evidence_reviewed
- missing_evidence
- requires_guardian_attention
- unexpected_steps
- unexpected_nested_workflows
- unusual_execution_pattern

Do not include raw secrets or sensitive source-document contents in
findings.

Use evidence references whenever possible.
"""

    return Task(
        description=description,

        expected_output=(
            "A validated RunBehaviorAnalysis describing whether the "
            "observed Glean workflow execution materially deviated from "
            "the monitored agent's declared baseline."
        ),

        agent=agent,

        output_pydantic=RunBehaviorAnalysis,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def analyze_run_behavior(
    baseline: AgentBaseline,
    telemetry: NormalizedTelemetry,
) -> RunBehaviorAnalysis:
    """
    Run the CrewAI specialist and return validated Pydantic output.

    This is the function LangGraph will eventually call from the
    Run Behavior node.
    """

    agent = build_run_behavior_agent()

    task = build_run_behavior_task(
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
            "Run Behavior Analyst CrewAI execution failed."
        ) from exc

    return _extract_pydantic_result(
        result=result,
        task=task,
        expected_type=RunBehaviorAnalysis,
    )


# ---------------------------------------------------------------------------
# CrewAI output extraction
# ---------------------------------------------------------------------------


def _extract_pydantic_result(
    result: Any,
    task: Task,
    expected_type: type[RunBehaviorAnalysis],
) -> RunBehaviorAnalysis:
    """
    Safely extract the structured CrewAI result.

    CrewAI versions/providers may expose structured task output slightly
    differently, so this function checks the validated paths rather than
    trusting arbitrary text.
    """

    # --------------------------------------------------------------
    # Preferred path:
    # CrewOutput.pydantic
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Task output path
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Structured dictionary fallback
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Do NOT silently trust free-form output.
    # --------------------------------------------------------------

    raise RuntimeError(
        "Run Behavior Analyst did not return a valid "
        "RunBehaviorAnalysis Pydantic object."
    )


# ---------------------------------------------------------------------------
# LangGraph node adapter
# ---------------------------------------------------------------------------


def run_behavior_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    LangGraph-compatible node.

    Reads:

        state["baseline"]
        state["telemetry"]

    Writes ONLY:

        run_behavior_analysis

    OR:

        run_behavior_error

    This unique-key write pattern allows the node to execute safely
    in parallel with the other Spy agents.
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
            "run_behavior_error": (
                "Guardian graph state does not contain "
                "a valid AgentBaseline."
            )
        }

    if not isinstance(
        telemetry,
        NormalizedTelemetry,
    ):
        return {
            "run_behavior_error": (
                "Guardian graph state does not contain "
                "valid NormalizedTelemetry."
            )
        }

    try:
        analysis = analyze_run_behavior(
            baseline=baseline,
            telemetry=telemetry,
        )

        return {
            "run_behavior_analysis": analysis
        }

    except Exception as exc:
        return {
            "run_behavior_error": str(exc)
        }


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
