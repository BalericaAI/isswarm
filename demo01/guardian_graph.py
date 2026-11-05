"""
Guardian Telemetry Swarm
guardian_graph.py

LangGraph execution orchestrator for the Guardian Telemetry Swarm.

Responsibilities:

- validate graph input
- initialize shared state
- fan five CrewAI Spy agents out in parallel
- wait for ALL specialist branches
- repair missing specialist result/error slots defensively
- invoke Guardian Correlator
- finalize graph state
- return validated GuardianGraphResult

IMPORTANT ARCHITECTURE:

guardian_graph.py
    =
EXECUTION orchestration

guardian_correlator.py
    =
EVIDENCE / DECISION orchestration

The graph does NOT perform security reasoning itself.

Graph topology:

                         START
                           |
                           v
                       initialize
                           |
          +----------------+----------------+
          |        |       |       |        |
          v        v       v       v        v
       Run Spy  Tool Spy Data Spy Output  Protect
          |        |       |       |        |
          +----------------+----------------+
                           |
                           v
                       synchronize
                           |
                           v
                        correlate
                           |
                           v
                         finalize
                           |
                           v
                          END
"""

from __future__ import annotations

from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph

from data_access_agent import data_access_node
from graph_state import (
    GraphPhase,
    GuardianGraphInput,
    GuardianGraphResult,
    GuardianGraphState,
    build_graph_result,
    build_initial_graph_state,
    ready_for_correlation,
)
from guardian_correlator import guardian_correlator_node
from output_behavior_agent import output_behavior_node
from protect_findings_agent import protect_findings_node
from run_behavior_agent import run_behavior_node
from tool_behavior_agent import tool_behavior_node


# ---------------------------------------------------------------------------
# Node names
# ---------------------------------------------------------------------------


NODE_INITIALIZE = "initialize"

NODE_RUN_BEHAVIOR = "run_behavior"

NODE_TOOL_BEHAVIOR = "tool_behavior"

NODE_DATA_ACCESS = "data_access"

NODE_OUTPUT_BEHAVIOR = "output_behavior"

NODE_PROTECT_FINDINGS = "protect_findings"

NODE_SYNCHRONIZE = "synchronize"

NODE_CORRELATE = "correlate"

NODE_FINALIZE = "finalize"


SPY_NODES = [
    NODE_RUN_BEHAVIOR,
    NODE_TOOL_BEHAVIOR,
    NODE_DATA_ACCESS,
    NODE_OUTPUT_BEHAVIOR,
    NODE_PROTECT_FINDINGS,
]


# ---------------------------------------------------------------------------
# Initialization node
# ---------------------------------------------------------------------------


def initialize_node(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Validate the basic relationship between baseline and telemetry.

    This node performs NO security analysis.

    It establishes the graph execution phase.

    Expected initial state already contains:

        baseline
        telemetry
        run_id
        monitored_agent
        phase

    because build_initial_graph_state() created it.
    """

    baseline = state.get(
        "baseline"
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


    if baseline is None:
        raise ValueError(
            "Guardian graph initialization failed: "
            "baseline is missing."
        )


    if telemetry is None:
        raise ValueError(
            "Guardian graph initialization failed: "
            "telemetry is missing."
        )


    if not run_id:
        raise ValueError(
            "Guardian graph initialization failed: "
            "run_id is missing."
        )


    if not monitored_agent:
        raise ValueError(
            "Guardian graph initialization failed: "
            "monitored_agent is missing."
        )


    # --------------------------------------------------------------
    # Defensive consistency check
    # --------------------------------------------------------------

    if telemetry.run.run_id != run_id:
        raise ValueError(
            "Guardian graph initialization failed: "
            "state run_id does not match telemetry run_id."
        )


    if baseline.agent_name != monitored_agent:
        raise ValueError(
            "Guardian graph initialization failed: "
            "state monitored_agent does not match baseline agent_name."
        )


    return {
        "phase": GraphPhase.ANALYZING
    }


# ---------------------------------------------------------------------------
# Synchronization node
# ---------------------------------------------------------------------------


def synchronize_spies_node(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Execute after ALL five Spy branches have completed.

    LangGraph itself provides the synchronization barrier through:

        graph.add_edge(
            SPY_NODES,
            NODE_SYNCHRONIZE
        )

    This node performs defensive validation.

    Every specialist should have produced EITHER:

        <specialist>_analysis

    OR:

        <specialist>_error

    If a specialist somehow returned neither, create an explicit error
    rather than silently treating the missing result as NORMAL.
    """

    updates: dict[str, Any] = {
        "phase": GraphPhase.CORRELATING
    }


    specialist_slots = [
        (
            "run_behavior_analysis",
            "run_behavior_error",
            "Run Behavior Analyst",
        ),

        (
            "tool_behavior_analysis",
            "tool_behavior_error",
            "Tool & Action Analyst",
        ),

        (
            "data_access_analysis",
            "data_access_error",
            "Data Access Analyst",
        ),

        (
            "output_behavior_analysis",
            "output_behavior_error",
            "Output Behavior Analyst",
        ),

        (
            "protect_findings_analysis",
            "protect_findings_error",
            "Protect Findings Analyst",
        ),
    ]


    for (
        result_key,
        error_key,
        analyst_name,
    ) in specialist_slots:

        result_present = (
            state.get(
                result_key  # type: ignore[literal-required]
            )
            is not None
        )

        error_present = bool(
            state.get(
                error_key  # type: ignore[literal-required]
            )
        )


        if not result_present and not error_present:

            updates[
                error_key
            ] = (
                f"{analyst_name} completed its graph branch "
                "without returning either a validated analysis "
                "or an explicit error."
            )


    return updates


# ---------------------------------------------------------------------------
# Correlator wrapper
# ---------------------------------------------------------------------------


def correlate_node(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Invoke the Guardian Correlator.

    This wrapper exists so guardian_graph.py owns orchestration while
    guardian_correlator.py remains focused on evidence synthesis.
    """

    # --------------------------------------------------------------
    # Defensive check
    #
    # synchronize_spies_node should have guaranteed this.
    # --------------------------------------------------------------

    if not ready_for_correlation(
        state
    ):
        return {
            "guardian_decision": None,
            "phase": GraphPhase.ERROR,
        }


    result = guardian_correlator_node(
        state
    )


    # Correlator returns:

    # {
    #     "guardian_decision": GuardianDecision(...)
    # }

    return result


# ---------------------------------------------------------------------------
# Finalization node
# ---------------------------------------------------------------------------


def finalize_node(
    state: GuardianGraphState,
) -> dict[str, Any]:
    """
    Mark the graph complete.

    The graph is COMPLETE only if a GuardianDecision exists.

    Otherwise mark execution ERROR.
    """

    decision = state.get(
        "guardian_decision"
    )


    if decision is None:

        return {
            "phase": GraphPhase.ERROR
        }


    return {
        "phase": GraphPhase.COMPLETE
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_guardian_graph(
    checkpointer: Any | None = None,
):
    """
    Build and compile the Guardian Telemetry Swarm LangGraph.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer.

        For the first PoC this may be None.

        Later examples:

            InMemorySaver
            SqliteSaver
            PostgresSaver

    Returns
    -------
    Compiled LangGraph application.
    """

    builder = StateGraph(
        GuardianGraphState
    )


    # ==============================================================
    # Nodes
    # ==============================================================

    builder.add_node(
        NODE_INITIALIZE,
        initialize_node,
    )


    # --------------------------------------------------------------
    # Five independent CrewAI Spy nodes
    # --------------------------------------------------------------

    builder.add_node(
        NODE_RUN_BEHAVIOR,
        run_behavior_node,
    )

    builder.add_node(
        NODE_TOOL_BEHAVIOR,
        tool_behavior_node,
    )

    builder.add_node(
        NODE_DATA_ACCESS,
        data_access_node,
    )

    builder.add_node(
        NODE_OUTPUT_BEHAVIOR,
        output_behavior_node,
    )

    builder.add_node(
        NODE_PROTECT_FINDINGS,
        protect_findings_node,
    )


    # --------------------------------------------------------------
    # Synchronization / correlation / finalization
    # --------------------------------------------------------------

    builder.add_node(
        NODE_SYNCHRONIZE,
        synchronize_spies_node,
    )

    builder.add_node(
        NODE_CORRELATE,
        correlate_node,
    )

    builder.add_node(
        NODE_FINALIZE,
        finalize_node,
    )


    # ==============================================================
    # Entry
    # ==============================================================

    builder.add_edge(
        START,
        NODE_INITIALIZE,
    )


    # ==============================================================
    # FAN-OUT
    #
    # One node -> five outgoing edges.
    #
    # LangGraph runs these destinations in parallel in the next
    # super-step.
    # ==============================================================

    builder.add_edge(
        NODE_INITIALIZE,
        NODE_RUN_BEHAVIOR,
    )

    builder.add_edge(
        NODE_INITIALIZE,
        NODE_TOOL_BEHAVIOR,
    )

    builder.add_edge(
        NODE_INITIALIZE,
        NODE_DATA_ACCESS,
    )

    builder.add_edge(
        NODE_INITIALIZE,
        NODE_OUTPUT_BEHAVIOR,
    )

    builder.add_edge(
        NODE_INITIALIZE,
        NODE_PROTECT_FINDINGS,
    )


    # ==============================================================
    # FAN-IN / BARRIER
    #
    # This is the important LangGraph operation.
    #
    # Providing a LIST of start nodes causes LangGraph to wait for
    # ALL listed nodes before executing NODE_SYNCHRONIZE.
    #
    # We therefore do NOT need:
    #
    #     counters
    #     polling
    #     sleep()
    #     manual futures
    #     "are all agents done?" LLM logic
    # ==============================================================

    builder.add_edge(
        SPY_NODES,
        NODE_SYNCHRONIZE,
    )


    # ==============================================================
    # Correlation
    # ==============================================================

    builder.add_edge(
        NODE_SYNCHRONIZE,
        NODE_CORRELATE,
    )


    # ==============================================================
    # Finalization
    # ==============================================================

    builder.add_edge(
        NODE_CORRELATE,
        NODE_FINALIZE,
    )

    builder.add_edge(
        NODE_FINALIZE,
        END,
    )


    # ==============================================================
    # Compile
    # ==============================================================

    if checkpointer is not None:

        return builder.compile(
            checkpointer=checkpointer
        )


    return builder.compile()


# ---------------------------------------------------------------------------
# High-level execution helper
# ---------------------------------------------------------------------------


def run_guardian_graph(
    graph_input: GuardianGraphInput,
    graph: Any | None = None,
    thread_id: str | None = None,
) -> GuardianGraphResult:
    """
    Execute the complete Guardian telemetry analysis.

    Flow:

        GuardianGraphInput
                |
                v
        build_initial_graph_state()
                |
                v
          LangGraph.invoke()
                |
                v
        GuardianGraphState
                |
                v
        build_graph_result()
                |
                v
        GuardianGraphResult
    """

    if graph is None:
        graph = build_guardian_graph()


    initial_state = build_initial_graph_state(
        graph_input
    )


    # --------------------------------------------------------------
    # Optional thread configuration
    #
    # Required when using checkpointers.
    # --------------------------------------------------------------

    config = None

    if thread_id:

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }


    if config is not None:

        final_state = graph.invoke(
            initial_state,
            config=config,
        )

    else:

        final_state = graph.invoke(
            initial_state
        )


    return build_graph_result(
        final_state
    )


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def stream_guardian_graph(
    graph_input: GuardianGraphInput,
    graph: Any | None = None,
    thread_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Stream graph updates.

    This will be especially useful for the demo because the student can
    show each Spy completing independently before the Guardian correlates
    the results.

    Example conceptual output:

        initialize
            ↓
        run_behavior
        tool_behavior
        data_access
        output_behavior
        protect_findings
            ↓
        synchronize
            ↓
        correlate
            ↓
        finalize
    """

    if graph is None:
        graph = build_guardian_graph()


    initial_state = build_initial_graph_state(
        graph_input
    )


    config = None

    if thread_id:

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }


    kwargs: dict[str, Any] = {
        "stream_mode": "updates"
    }


    if config is not None:
        kwargs["config"] = config


    for update in graph.stream(
        initial_state,
        **kwargs,
    ):

        yield update


# ---------------------------------------------------------------------------
# Optional development graph
# ---------------------------------------------------------------------------


def build_development_graph():
    """
    Development convenience helper using an in-memory checkpointer.

    This is useful for:

    - LangGraph Studio
    - debugging
    - state inspection
    - demonstrating checkpoints

    It is NOT durable production storage.
    """

    from langgraph.checkpoint.memory import (
        InMemorySaver,
    )


    checkpointer = InMemorySaver()


    return build_guardian_graph(
        checkpointer=checkpointer
    )
