"""
Guardian Telemetry Swarm
graph_state.py

Shared LangGraph state for the Guardian Telemetry Swarm.

Architecture:

Validated Pydantic evidence
        |
        v
GuardianGraphState
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
   Run Behavior         Tool Behavior       Data Access
      Agent                Agent               Agent
        |                   |                   |
        +-------------------+-------------------+
                            |
                  +---------+---------+
                  |                   |
                  v                   v
             Output Agent        Protect Agent
                  |                   |
                  +---------+---------+
                            |
                            v
                   Guardian Correlator
                            |
                            v
                     GuardianDecision


DESIGN PRINCIPLES

1. Pydantic validates evidence entering and leaving the graph.
2. TypedDict provides lightweight LangGraph shared state.
3. Each parallel specialist writes to a unique state key.
4. Specialists do not overwrite telemetry or baseline data.
5. Specialists do not make enforcement decisions.
6. The Guardian Correlator owns the final NORMAL / REVIEW / ALERT result.
7. Errors from one specialist remain isolated from the others.

IMPORTANT:

This file contains NO CrewAI execution.
This file contains NO routing logic.
This file contains NO LLM calls.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from models import (
    AgentBaseline,
    DataAccessAnalysis,
    GuardianDecision,
    NormalizedTelemetry,
    OutputBehaviorAnalysis,
    ProtectFindingsAnalysis,
    RunBehaviorAnalysis,
    ToolBehaviorAnalysis,
)


# ---------------------------------------------------------------------------
# Graph lifecycle
# ---------------------------------------------------------------------------


class GraphPhase(StrEnum):
    """
    High-level orchestration phase.

    Specialist nodes should generally NOT modify this field individually
    while executing in parallel.

    The orchestration graph changes phase at synchronization boundaries.
    """

    INITIALIZED = "INITIALIZED"

    ANALYZING = "ANALYZING"

    CORRELATING = "CORRELATING"

    COMPLETE = "COMPLETE"

    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Boundary models
# ---------------------------------------------------------------------------


class GraphBoundaryModel(BaseModel):
    """
    Strict Pydantic base model used at graph boundaries.

    LangGraph uses GuardianGraphState internally.

    Pydantic models validate:
        - graph input
        - graph output
        - individual CrewAI results
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class GuardianGraphInput(GraphBoundaryModel):
    """
    Everything required to begin a Guardian telemetry analysis.
    """

    baseline: AgentBaseline

    telemetry: NormalizedTelemetry


class GuardianGraphResult(GraphBoundaryModel):
    """
    Validated result returned after the graph completes.

    Individual specialist results are preserved so that Security can see
    exactly how the Guardian reached its conclusion.
    """

    run_id: str

    monitored_agent: str

    run_behavior_analysis: RunBehaviorAnalysis | None = None

    tool_behavior_analysis: ToolBehaviorAnalysis | None = None

    data_access_analysis: DataAccessAnalysis | None = None

    output_behavior_analysis: OutputBehaviorAnalysis | None = None

    protect_findings_analysis: ProtectFindingsAnalysis | None = None

    guardian_decision: GuardianDecision | None = None

    phase: GraphPhase

    specialist_errors: dict[str, str] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Internal LangGraph state
# ---------------------------------------------------------------------------


class GuardianGraphState(TypedDict, total=False):
    """
    Shared state carried through LangGraph.

    total=False is intentional.

    LangGraph nodes return PARTIAL state updates rather than rebuilding
    the entire state object after every node.

    For example:

        return {
            "tool_behavior_analysis": result
        }

    rather than returning every field in the graph.
    """

    # ------------------------------------------------------------------
    # Immutable-ish source evidence
    # ------------------------------------------------------------------

    baseline: AgentBaseline

    telemetry: NormalizedTelemetry

    run_id: str

    monitored_agent: str

    # ------------------------------------------------------------------
    # Orchestration metadata
    # ------------------------------------------------------------------

    phase: GraphPhase

    # ------------------------------------------------------------------
    # CrewAI specialist results
    #
    # IMPORTANT:
    #
    # Every parallel specialist gets its OWN state key.
    #
    # This avoids concurrent-update collisions such as:
    #
    #     Run Spy  ----\
    #     Tool Spy ----+--> findings[]
    #     Data Spy ----/
    #
    # Instead:
    #
    #     Run Spy  ----> run_behavior_analysis
    #     Tool Spy ----> tool_behavior_analysis
    #     Data Spy ----> data_access_analysis
    #
    # The Guardian Correlator combines them later.
    # ------------------------------------------------------------------

    run_behavior_analysis: RunBehaviorAnalysis

    tool_behavior_analysis: ToolBehaviorAnalysis

    data_access_analysis: DataAccessAnalysis

    output_behavior_analysis: OutputBehaviorAnalysis

    protect_findings_analysis: ProtectFindingsAnalysis

    # ------------------------------------------------------------------
    # Specialist-specific errors
    #
    # Again, use separate keys because Spy agents may execute in parallel.
    # ------------------------------------------------------------------

    run_behavior_error: str

    tool_behavior_error: str

    data_access_error: str

    output_behavior_error: str

    protect_findings_error: str

    # ------------------------------------------------------------------
    # Final correlated result
    # ------------------------------------------------------------------

    guardian_decision: GuardianDecision


# ---------------------------------------------------------------------------
# Initial-state builder
# ---------------------------------------------------------------------------


def build_initial_graph_state(
    graph_input: GuardianGraphInput,
) -> GuardianGraphState:
    """
    Convert validated graph input into initial LangGraph state.

    This is the ONLY recommended way to construct initial state.

    Example:

        validated_input = GuardianGraphInput(
            baseline=baseline,
            telemetry=telemetry,
        )

        state = build_initial_graph_state(
            validated_input
        )
    """

    return GuardianGraphState(
        baseline=graph_input.baseline,

        telemetry=graph_input.telemetry,

        run_id=graph_input.telemetry.run.run_id,

        monitored_agent=graph_input.baseline.agent_name,

        phase=GraphPhase.INITIALIZED,
    )


# ---------------------------------------------------------------------------
# Final-state validator
# ---------------------------------------------------------------------------


def build_graph_result(
    state: GuardianGraphState,
) -> GuardianGraphResult:
    """
    Convert completed LangGraph state back into a strict Pydantic model.

    This gives us validation at the EXIT boundary too.

    If the graph somehow loses required orchestration fields, Pydantic
    catches that here.
    """

    errors = collect_specialist_errors(
        state
    )

    return GuardianGraphResult(
        run_id=_require_state_value(
            state,
            "run_id",
        ),

        monitored_agent=_require_state_value(
            state,
            "monitored_agent",
        ),

        run_behavior_analysis=state.get(
            "run_behavior_analysis"
        ),

        tool_behavior_analysis=state.get(
            "tool_behavior_analysis"
        ),

        data_access_analysis=state.get(
            "data_access_analysis"
        ),

        output_behavior_analysis=state.get(
            "output_behavior_analysis"
        ),

        protect_findings_analysis=state.get(
            "protect_findings_analysis"
        ),

        guardian_decision=state.get(
            "guardian_decision"
        ),

        phase=state.get(
            "phase",
            GraphPhase.ERROR,
        ),

        specialist_errors=errors,
    )


# ---------------------------------------------------------------------------
# State inspection helpers
# ---------------------------------------------------------------------------


def collect_specialist_errors(
    state: GuardianGraphState,
) -> dict[str, str]:
    """
    Collect specialist errors after parallel execution.

    We intentionally avoid having all specialists append to one shared
    list because parallel writes to the same state key would require a
    reducer.

    Separate error fields make concurrency behavior explicit.
    """

    error_fields = {
        "run_behavior": "run_behavior_error",
        "tool_behavior": "tool_behavior_error",
        "data_access": "data_access_error",
        "output_behavior": "output_behavior_error",
        "protect_findings": "protect_findings_error",
    }

    results: dict[str, str] = {}

    for specialist, field_name in error_fields.items():

        value = state.get(
            field_name  # type: ignore[literal-required]
        )

        if value:
            results[specialist] = str(
                value
            )

    return results


def specialist_result_count(
    state: GuardianGraphState,
) -> int:
    """
    Count successfully populated specialist-analysis fields.

    This is deterministic orchestration information.

    Do not ask an LLM:
        "Did all five agents run?"
    """

    result_fields = (
        "run_behavior_analysis",
        "tool_behavior_analysis",
        "data_access_analysis",
        "output_behavior_analysis",
        "protect_findings_analysis",
    )

    return sum(
        1
        for field_name in result_fields
        if state.get(
            field_name  # type: ignore[literal-required]
        )
        is not None
    )


def all_specialists_completed(
    state: GuardianGraphState,
) -> bool:
    """
    Return True only when all five specialist outputs are present.

    A specialist error does NOT count as successful completion.
    """

    return specialist_result_count(
        state
    ) == 5


def has_specialist_errors(
    state: GuardianGraphState,
) -> bool:
    """
    Determine whether any specialist failed.
    """

    return bool(
        collect_specialist_errors(
            state
        )
    )


# ---------------------------------------------------------------------------
# Correlation readiness
# ---------------------------------------------------------------------------


def ready_for_correlation(
    state: GuardianGraphState,
) -> bool:
    """
    Determine whether enough processing has occurred for the Guardian
    Correlator to run.

    IMPORTANT:

    The Correlator MAY still run if one specialist failed.

    Why?

        Specialist failure
            !=
        abandon monitoring

    Instead the correlator should receive the available evidence and
    treat the missing specialist output as incomplete evidence.

    Therefore correlation becomes possible when:

        specialist results
        +
        specialist failures
        ==
        five expected specialists
    """

    result_fields = (
        "run_behavior_analysis",
        "tool_behavior_analysis",
        "data_access_analysis",
        "output_behavior_analysis",
        "protect_findings_analysis",
    )

    error_fields = (
        "run_behavior_error",
        "tool_behavior_error",
        "data_access_error",
        "output_behavior_error",
        "protect_findings_error",
    )

    completed_slots = 0

    for result_field, error_field in zip(
        result_fields,
        error_fields,
        strict=True,
    ):

        result_present = (
            state.get(
                result_field  # type: ignore[literal-required]
            )
            is not None
        )

        error_present = bool(
            state.get(
                error_field  # type: ignore[literal-required]
            )
        )

        if result_present or error_present:
            completed_slots += 1

    return completed_slots == 5


# ---------------------------------------------------------------------------
# Graph-state summary
# ---------------------------------------------------------------------------


def summarize_graph_state(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Produce a compact deterministic view useful for logging/debugging.

    This deliberately avoids dumping full telemetry or sensitive content.
    """

    telemetry = state.get(
        "telemetry"
    )

    decision = state.get(
        "guardian_decision"
    )

    return {
        "run_id": state.get(
            "run_id"
        ),

        "monitored_agent": state.get(
            "monitored_agent"
        ),

        "phase": state.get(
            "phase"
        ),

        "specialists_completed": (
            specialist_result_count(
                state
            )
        ),

        "specialist_errors": list(
            collect_specialist_errors(
                state
            ).keys()
        ),

        "evidence_complete": (
            telemetry.evidence_complete
            if telemetry
            else False
        ),

        "guardian_status": (
            decision.guardian_status
            if decision
            else None
        ),

        "guardian_severity": (
            decision.severity
            if decision
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_state_value(
    state: GuardianGraphState,
    key: str,
) -> Any:
    """
    Fetch a required orchestration value.

    Raises immediately if LangGraph somehow reaches finalization without
    the required state.
    """

    value = state.get(
        key  # type: ignore[literal-required]
    )

    if value is None:
        raise ValueError(
            f"Required graph state value is missing: {key}"
        )

    return value
