"""
Guardian Telemetry Swarm
main.py

Application entry point.

Responsibilities:

1. Load environment configuration.
2. Load the monitored-agent behavioral baseline.
3. Retrieve raw telemetry.
4. Normalize telemetry.
5. Create validated GuardianGraphInput.
6. Execute the LangGraph Guardian swarm.
7. Print and optionally save the final result.

main.py contains NO:

- security reasoning
- CrewAI agent definitions
- correlation rules
- anomaly detection rules
- LangGraph topology definitions

Those responsibilities belong elsewhere.

Pipeline:

.env
 |
 v
baseline JSON
 |
 v
TelemetryClient
 |
 v
RawTelemetryBundle
 |
 v
TelemetryNormalizer
 |
 v
NormalizedTelemetry
 |
 v
GuardianGraphInput
 |
 v
Guardian LangGraph
 |
 v
GuardianGraphResult
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from graph_state import (
    GuardianGraphInput,
    GuardianGraphResult,
)
from guardian_graph import (
    build_development_graph,
    build_guardian_graph,
    run_guardian_graph,
    stream_guardian_graph,
)
from models import AgentBaseline
from telemetry_client import (
    TelemetryClient,
    TelemetryClientError,
)
from telemetry_normalizer import (
    TelemetryNormalizer,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


DEFAULT_BASELINE_FILE = (
    "baseline/five_sentence_reviewer.json"
)

DEFAULT_OUTPUT_DIR = "output"

DEFAULT_RUN_ID = "run-normal-001"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Execute one Guardian telemetry analysis.

    Returns:

        0 = successful execution
        1 = configuration/input failure
        2 = telemetry retrieval failure
        3 = validation failure
        4 = graph execution failure
    """

    load_dotenv()

    print_banner()

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    baseline_file = os.getenv(
        "BASELINE_FILE",
        DEFAULT_BASELINE_FILE,
    )

    run_id = os.getenv(
        "RUN_ID",
        DEFAULT_RUN_ID,
    )

    output_dir = os.getenv(
        "OUTPUT_DIR",
        DEFAULT_OUTPUT_DIR,
    )

    save_output = env_bool(
        "SAVE_OUTPUT",
        default=True,
    )

    stream_output = env_bool(
        "STREAM_GRAPH",
        default=True,
    )

    use_checkpointer = env_bool(
        "USE_CHECKPOINTER",
        default=False,
    )

    thread_id = os.getenv(
        "THREAD_ID",
        run_id,
    )

    # --------------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------------

    try:

        baseline = load_baseline(
            baseline_file
        )

    except Exception as exc:

        print_error(
            "Could not load agent baseline.",
            exc,
        )

        return 1


    print(
        f"Monitored Agent : {baseline.agent_name}"
    )

    print(
        f"Run ID          : {run_id}"
    )

    print(
        f"Baseline        : {baseline_file}"
    )

    print()


    # --------------------------------------------------------------
    # Retrieve telemetry
    # --------------------------------------------------------------

    try:

        telemetry_client = (
            TelemetryClient.from_environment()
        )

        raw_bundle = (
            telemetry_client.get_run_telemetry(
                run_id=run_id
            )
        )

    except TelemetryClientError as exc:

        print_error(
            "Telemetry retrieval failed.",
            exc,
        )

        return 2

    except Exception as exc:

        print_error(
            "Unexpected telemetry error.",
            exc,
        )

        return 2


    print(
        f"Telemetry Source: {raw_bundle.source}"
    )

    if raw_bundle.retrieval_warnings:

        print(
            "Retrieval Warnings:"
        )

        for warning in (
            raw_bundle.retrieval_warnings
        ):
            print(
                f"  - {warning}"
            )

    print()


    # --------------------------------------------------------------
    # Normalize telemetry
    # --------------------------------------------------------------

    try:

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

    except ValidationError as exc:

        print_error(
            "Normalized telemetry failed "
            "Pydantic validation.",
            exc,
        )

        return 3

    except Exception as exc:

        print_error(
            "Telemetry normalization failed.",
            exc,
        )

        return 3


    print_telemetry_summary(
        telemetry
    )


    # --------------------------------------------------------------
    # Build validated graph input
    # --------------------------------------------------------------

    try:

        graph_input = GuardianGraphInput(
            baseline=baseline,
            telemetry=telemetry,
        )

    except ValidationError as exc:

        print_error(
            "Guardian graph input failed validation.",
            exc,
        )

        return 3


    # --------------------------------------------------------------
    # Build LangGraph
    # --------------------------------------------------------------

    try:

        if use_checkpointer:

            graph = build_development_graph()

        else:

            graph = build_guardian_graph()

    except Exception as exc:

        print_error(
            "Could not construct Guardian LangGraph.",
            exc,
        )

        return 4


    # --------------------------------------------------------------
    # Execute
    # --------------------------------------------------------------

    try:

        if stream_output:

            print(
                "=" * 70
            )

            print(
                "GUARDIAN SWARM EXECUTION"
            )

            print(
                "=" * 70
            )

            print()

            display_graph_stream(
                graph_input=graph_input,
                graph=graph,
                thread_id=(
                    thread_id
                    if use_checkpointer
                    else None
                ),
            )

            # Run once more for the complete validated object only when
            # not using a checkpointer.
            #
            # In production we would normally consume final state from
            # the same stream execution instead of executing twice.
            #
            # For this educational PoC, use non-stream mode for the final
            # saved result by default.
            result = run_guardian_graph(
                graph_input=graph_input,
                graph=graph,
                thread_id=(
                    thread_id
                    if use_checkpointer
                    else None
                ),
            )

        else:

            result = run_guardian_graph(
                graph_input=graph_input,
                graph=graph,
                thread_id=(
                    thread_id
                    if use_checkpointer
                    else None
                ),
            )

    except Exception as exc:

        print_error(
            "Guardian graph execution failed.",
            exc,
        )

        return 4


    # --------------------------------------------------------------
    # Display final result
    # --------------------------------------------------------------

    print_final_result(
        result
    )


    # --------------------------------------------------------------
    # Save output
    # --------------------------------------------------------------

    if save_output:

        try:

            output_path = save_result(
                result=result,
                output_dir=output_dir,
            )

            print()
            print(
                f"Saved result    : {output_path}"
            )

        except Exception as exc:

            # Saving output should not invalidate an otherwise
            # successful security analysis.
            print()
            print(
                "WARNING: Result completed but could "
                "not be saved."
            )

            print(
                f"Reason: {exc}"
            )


    return 0


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------


def load_baseline(
    path: str,
) -> AgentBaseline:
    """
    Load and validate the monitored-agent behavioral baseline.
    """

    baseline_path = Path(
        path
    )

    if not baseline_path.exists():

        raise FileNotFoundError(
            f"Baseline file not found: "
            f"{baseline_path}"
        )


    with baseline_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )


    return AgentBaseline.model_validate(
        data
    )


# ---------------------------------------------------------------------------
# Streaming display
# ---------------------------------------------------------------------------


def display_graph_stream(
    graph_input: GuardianGraphInput,
    graph,
    thread_id: str | None = None,
) -> None:
    """
    Display LangGraph node updates as the swarm executes.

    This is intentionally presentation-friendly for the PoC.
    """

    for update in stream_guardian_graph(
        graph_input=graph_input,
        graph=graph,
        thread_id=thread_id,
    ):

        if not isinstance(
            update,
            dict,
        ):
            continue


        for node_name, node_update in (
            update.items()
        ):

            print(
                f"[{node_name.upper()}]"
            )

            if not isinstance(
                node_update,
                dict,
            ):
                print(
                    node_update
                )

                print()
                continue


            display_node_update(
                node_name=node_name,
                node_update=node_update,
            )

            print()


# ---------------------------------------------------------------------------
# Node update display
# ---------------------------------------------------------------------------


def display_node_update(
    node_name: str,
    node_update: dict,
) -> None:
    """
    Print compact human-readable node results.

    Avoid dumping entire telemetry blobs.
    """

    analysis_keys = (
        "run_behavior_analysis",
        "tool_behavior_analysis",
        "data_access_analysis",
        "output_behavior_analysis",
        "protect_findings_analysis",
    )


    # --------------------------------------------------------------
    # Specialist result
    # --------------------------------------------------------------

    for key in analysis_keys:

        analysis = node_update.get(
            key
        )

        if analysis is None:
            continue

        print(
            f"  Analyst    : {analysis.analyst}"
        )

        print(
            f"  Status     : {analysis.status}"
        )

        print(
            f"  Severity   : {analysis.severity}"
        )

        print(
            f"  Confidence : {analysis.confidence:.2f}"
        )

        print(
            f"  Summary    : {analysis.summary}"
        )

        return


    # --------------------------------------------------------------
    # Specialist error
    # --------------------------------------------------------------

    error_keys = (
        "run_behavior_error",
        "tool_behavior_error",
        "data_access_error",
        "output_behavior_error",
        "protect_findings_error",
    )


    for key in error_keys:

        error = node_update.get(
            key
        )

        if error:

            print(
                f"  ERROR: {error}"
            )

            return


    # --------------------------------------------------------------
    # Guardian decision
    # --------------------------------------------------------------

    decision = node_update.get(
        "guardian_decision"
    )

    if decision is not None:

        print(
            f"  Status     : "
            f"{decision.guardian_status}"
        )

        print(
            f"  Severity   : "
            f"{decision.severity}"
        )

        print(
            f"  Confidence : "
            f"{decision.confidence:.2f}"
        )

        print(
            f"  Summary    : "
            f"{decision.summary}"
        )

        return


    # --------------------------------------------------------------
    # Phase transition
    # --------------------------------------------------------------

    phase = node_update.get(
        "phase"
    )

    if phase is not None:

        print(
            f"  Phase      : {phase}"
        )

        return


    print(
        "  Completed."
    )


# ---------------------------------------------------------------------------
# Telemetry summary
# ---------------------------------------------------------------------------


def print_telemetry_summary(
    telemetry,
) -> None:
    """
    Display deterministic telemetry facts before AI analysis begins.
    """

    print(
        "=" * 70
    )

    print(
        "NORMALIZED TELEMETRY"
    )

    print(
        "=" * 70
    )

    print(
        f"Run Status       : "
        f"{telemetry.run.status}"
    )

    print(
        f"Initiator        : "
        f"{telemetry.run.initiator}"
    )

    print(
        f"Steps            : "
        f"{telemetry.counts.step_count}"
    )

    print(
        f"Document Reads   : "
        f"{telemetry.counts.document_read_count}"
    )

    print(
        f"Write Actions    : "
        f"{telemetry.counts.write_action_count}"
    )

    print(
        f"External Actions : "
        f"{telemetry.counts.external_action_count}"
    )

    print(
        f"MCP Calls        : "
        f"{telemetry.counts.mcp_call_count}"
    )

    print(
        f"Protect Findings : "
        f"{telemetry.counts.protect_finding_count}"
    )

    print(
        f"Evidence Complete: "
        f"{telemetry.evidence_complete}"
    )


    if telemetry.missing_evidence:

        print(
            "Missing Evidence  :"
        )

        for item in (
            telemetry.missing_evidence
        ):

            print(
                f"  - {item}"
            )


    print()


# ---------------------------------------------------------------------------
# Final result display
# ---------------------------------------------------------------------------


def print_final_result(
    result: GuardianGraphResult,
) -> None:
    """
    Print the final Guardian disposition.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "GUARDIAN FINAL RESULT"
    )

    print(
        "=" * 70
    )


    decision = (
        result.guardian_decision
    )


    print(
        f"Run ID           : {result.run_id}"
    )

    print(
        f"Monitored Agent  : "
        f"{result.monitored_agent}"
    )

    print(
        f"Graph Phase      : "
        f"{result.phase}"
    )


    if decision is None:

        print(
            "Guardian Status  : UNAVAILABLE"
        )

        return


    print(
        f"Guardian Status  : "
        f"{decision.guardian_status}"
    )

    print(
        f"Severity         : "
        f"{decision.severity}"
    )

    print(
        f"Confidence       : "
        f"{decision.confidence:.2f}"
    )

    print(
        f"Security Review  : "
        f"{decision.security_review_required}"
    )

    print(
        f"Evidence Complete: "
        f"{decision.evidence_complete}"
    )

    print(
        f"Enforcement      : "
        f"{decision.enforcement_performed}"
    )

    print()

    print(
        "Summary:"
    )

    print(
        f"  {decision.summary}"
    )


    if decision.reason_codes:

        print()

        print(
            "Reason Codes:"
        )

        for reason in (
            decision.reason_codes
        ):

            print(
                f"  - {reason}"
            )


    if decision.recommended_actions:

        print()

        print(
            "Recommended Actions:"
        )

        for action in (
            decision.recommended_actions
        ):

            print(
                f"  - {action}"
            )


    if decision.missing_evidence:

        print()

        print(
            "Missing Evidence:"
        )

        for item in (
            decision.missing_evidence
        ):

            print(
                f"  - {item}"
            )


    if result.specialist_errors:

        print()

        print(
            "Specialist Errors:"
        )

        for (
            specialist,
            error,
        ) in result.specialist_errors.items():

            print(
                f"  - {specialist}: {error}"
            )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_result(
    result: GuardianGraphResult,
    output_dir: str,
) -> Path:
    """
    Save validated graph result as JSON.
    """

    directory = Path(
        output_dir
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    safe_run_id = sanitize_filename(
        result.run_id
    )


    output_path = (
        directory
        / f"{safe_run_id}_guardian_result.json"
    )


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result.model_dump(
                mode="json",
                exclude_none=True,
            ),
            file,
            indent=2,
        )


    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(
    value: str,
) -> str:
    """
    Allow conservative characters in generated filenames.
    """

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_"
    )

    cleaned = "".join(
        character
        if character in allowed
        else "_"
        for character in value
    )

    return (
        cleaned
        or "guardian_run"
    )


def env_bool(
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


def print_banner() -> None:

    print()

    print(
        "=" * 70
    )

    print(
        "GUARDIAN TELEMETRY SWARM"
    )

    print(
        "Glean Agent Security Monitoring PoC"
    )

    print(
        "=" * 70
    )

    print()


def print_error(
    message: str,
    error: Exception,
) -> None:

    print(
        file=sys.stderr
    )

    print(
        "=" * 70,
        file=sys.stderr,
    )

    print(
        "ERROR",
        file=sys.stderr,
    )

    print(
        "=" * 70,
        file=sys.stderr,
    )

    print(
        message,
        file=sys.stderr,
    )

    print(
        str(error),
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
