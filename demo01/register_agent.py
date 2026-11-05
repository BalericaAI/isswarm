"""
Guardian Telemetry Swarm
register_agent.py

Interactive onboarding interface for new Glean applications.

Purpose:

    Allow an operator to register a Glean application for Guardian
    monitoring using:

        1. Glean Application ID
        2. Owner
        3. Plain-English description of intended behavior

The onboarding LLM converts plain-English intent into a DRAFT
behavioral baseline.

The draft MUST be:

    - Pydantic validated
    - deterministically sanity checked
    - displayed to the operator
    - explicitly approved

before it is written to the Guardian registry.

The LLM DOES NOT authorize permissions.

Architecture:

Human
  |
  | App ID
  | Owner
  | Plain-English intended behavior
  v
Onboarding Draft Agent
  |
  v
Pydantic Draft
  |
  v
Deterministic Validation
  |
  v
Human Preview / Approval
  |
  +--> baselines/<agent>.json
  |
  +--> registry/monitored_agents.json
  |
  v
Guardian Live Monitor
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from agent_registry import (
    AgentRegistry,
    AgentRiskTier,
    BaselineVersion,
    MonitoredAgentRegistration,
    RegistryDocument,
)
from models import (
    AgentBaseline,
    ExpectedBehavior,
)


# ---------------------------------------------------------------------------
# Capability vocabulary
# ---------------------------------------------------------------------------


KNOWN_CAPABILITIES = {
    "INPUT_FORM",
    "READ_DOCUMENT",
    "SEARCH",
    "EMPLOYEE_SEARCH",
    "RESPOND",
    "SEND_EMAIL",
    "SEND_MESSAGE",
    "WRITE_DOCUMENT",
    "SHARE_DOCUMENT",
    "DELETE_DOCUMENT",
    "MOVE_DOCUMENT",
    "CHANGE_PERMISSIONS",
    "CREATE_RECORD",
    "DELETE_RECORD",
    "EXTERNAL_API",
    "MCP_TOOL",
    "EXECUTE_CODE",
    "SHELL_COMMAND",
}


WRITE_CAPABILITIES = {
    "SEND_EMAIL",
    "SEND_MESSAGE",
    "WRITE_DOCUMENT",
    "SHARE_DOCUMENT",
    "DELETE_DOCUMENT",
    "MOVE_DOCUMENT",
    "CHANGE_PERMISSIONS",
    "CREATE_RECORD",
    "DELETE_RECORD",
}


HIGH_RISK_CAPABILITIES = {
    "DELETE_DOCUMENT",
    "CHANGE_PERMISSIONS",
    "EXECUTE_CODE",
    "SHELL_COMMAND",
    "EXTERNAL_API",
    "MCP_TOOL",
}


# ---------------------------------------------------------------------------
# Draft models
# ---------------------------------------------------------------------------


class OnboardingBaseModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class DraftExpectedBehavior(
    OnboardingBaseModel
):

    document_reads_max: int = Field(
        default=0,
        ge=0,
    )

    write_actions_max: int = Field(
        default=0,
        ge=0,
    )

    mcp_calls_max: int = Field(
        default=0,
        ge=0,
    )

    external_actions_max: int = Field(
        default=0,
        ge=0,
    )

    expected_output_sentences: int | None = Field(
        default=None,
        ge=1,
    )


class AgentOnboardingDraft(
    OnboardingBaseModel
):
    """
    AI-generated DRAFT only.

    This object is not yet a Guardian authorization.
    """

    agent_name: str

    description: str

    risk_tier: AgentRiskTier

    expected_behavior: DraftExpectedBehavior

    allowed_capabilities: list[str]

    forbidden_capabilities: list[str]

    expected_steps: list[str]

    assumptions: list[str] = Field(
        default_factory=list
    )

    questions_for_operator: list[str] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_llm() -> LLM:

    return LLM(
        model=os.getenv(
            "CREWAI_MODEL",
            "openai/gpt-4.1-mini",
        ),
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Onboarding drafting agent
# ---------------------------------------------------------------------------


def build_onboarding_agent() -> Agent:

    return Agent(
        role=(
            "Guardian Agent Onboarding Architect"
        ),

        goal=(
            "Convert an operator's description of a Glean agent into "
            "a conservative draft behavioral security contract."
        ),

        backstory=(
            "You design behavioral baselines for enterprise AI agents. "
            "Your job is to describe what an agent is expected to do, "
            "not to grant unnecessary privileges. When intent is "
            "ambiguous you choose the more restrictive interpretation "
            "and clearly record assumptions."
        ),

        llm=build_llm(),

        verbose=False,

        allow_delegation=False,

        allow_code_execution=False,
    )


# ---------------------------------------------------------------------------
# Drafting task
# ---------------------------------------------------------------------------


def build_onboarding_task(
    agent: Agent,
    app_id: str,
    owner: str,
    description: str,
) -> Task:

    capability_list = "\n".join(
        f"- {item}"
        for item in sorted(
            KNOWN_CAPABILITIES
        )
    )


    prompt = f"""
Create a DRAFT behavioral security baseline for a Glean application.

This is configuration drafting.

It is NOT security authorization.

============================================================
APPLICATION
============================================================

Glean Application ID:

    {app_id}

Owner:

    {owner}

Operator description:

    {description}

============================================================
CAPABILITY VOCABULARY
============================================================

Use ONLY these capability names:

{capability_list}

============================================================
RULES
============================================================

1. Be conservative.

2. Do NOT invent capabilities that are not reasonably implied by the
   operator description.

3. Reading and responding do NOT imply permission to write.

4. Search does NOT imply permission to access arbitrary sensitive data.

5. MCP_TOOL should be allowed only when MCP usage is explicitly stated
   or clearly required.

6. EXTERNAL_API should be allowed only when explicitly stated.

7. EXECUTE_CODE and SHELL_COMMAND should normally remain forbidden.

8. DELETE_DOCUMENT and CHANGE_PERMISSIONS should normally remain
   forbidden unless explicitly required.

9. write_actions_max represents the maximum number of expected
   write-capable actions in one normal run.

10. document_reads_max represents the expected upper bound for a normal
    run, not the largest theoretically possible run.

11. If an exact output sentence count is not part of the contract,
    expected_output_sentences must be null.

12. Risk tier guidance:

    LOW:
        read-only or very constrained informational agent

    MODERATE:
        ordinary business automation with limited actions

    HIGH:
        agent can write business records, communicate externally,
        access sensitive business data, or use important MCP tools

    CRITICAL:
        privileged administration, destructive control, major financial
        authority, security administration, or similarly consequential
        behavior

13. expected_steps should describe conceptual business steps rather than
    internal implementation details that were not provided.

14. Put uncertainty into assumptions.

15. Put material unresolved questions into questions_for_operator.

============================================================
IMPORTANT
============================================================

The output is a DRAFT.

A human operator will review it before Guardian accepts it.

Return AgentOnboardingDraft only.
"""


    return Task(
        description=prompt,

        expected_output=(
            "A conservative Pydantic-valid AgentOnboardingDraft."
        ),

        agent=agent,

        output_pydantic=AgentOnboardingDraft,
    )


# ---------------------------------------------------------------------------
# Draft execution
# ---------------------------------------------------------------------------


def create_draft(
    app_id: str,
    owner: str,
    description: str,
) -> AgentOnboardingDraft:

    agent = build_onboarding_agent()

    task = build_onboarding_task(
        agent=agent,
        app_id=app_id,
        owner=owner,
        description=description,
    )


    crew = Crew(
        agents=[
            agent
        ],

        tasks=[
            task
        ],

        process=Process.sequential,

        verbose=False,
    )


    result = crew.kickoff()


    pydantic_result = getattr(
        result,
        "pydantic",
        None,
    )


    if isinstance(
        pydantic_result,
        AgentOnboardingDraft,
    ):

        return pydantic_result


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
            AgentOnboardingDraft,
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

        return (
            AgentOnboardingDraft
            .model_validate(
                json_dict
            )
        )


    raise RuntimeError(
        "Onboarding agent did not return a valid "
        "AgentOnboardingDraft."
    )


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------


def validate_draft(
    draft: AgentOnboardingDraft,
) -> list[str]:
    """
    Return validation problems.

    Empty list means the draft passes deterministic checks.
    """

    problems: list[str] = []


    allowed = {
        capability.upper()
        for capability
        in draft.allowed_capabilities
    }


    forbidden = {
        capability.upper()
        for capability
        in draft.forbidden_capabilities
    }


    # --------------------------------------------------------------
    # Unknown capabilities
    # --------------------------------------------------------------

    unknown = (
        allowed
        | forbidden
    ) - KNOWN_CAPABILITIES


    if unknown:

        problems.append(
            "Unknown capabilities: "
            + ", ".join(
                sorted(
                    unknown
                )
            )
        )


    # --------------------------------------------------------------
    # Capability cannot be allowed AND forbidden
    # --------------------------------------------------------------

    overlap = (
        allowed
        & forbidden
    )


    if overlap:

        problems.append(
            "Capabilities appear in both allowed and forbidden lists: "
            + ", ".join(
                sorted(
                    overlap
                )
            )
        )


    # --------------------------------------------------------------
    # Write-count consistency
    # --------------------------------------------------------------

    allowed_write = (
        allowed
        & WRITE_CAPABILITIES
    )


    if (
        draft.expected_behavior.write_actions_max
        == 0
        and allowed_write
    ):

        problems.append(
            "Write-capable actions are allowed but "
            "write_actions_max is 0."
        )


    if (
        draft.expected_behavior.write_actions_max
        > 0
        and not allowed_write
    ):

        problems.append(
            "write_actions_max is greater than 0 but "
            "no write-capable action is allowed."
        )


    # --------------------------------------------------------------
    # MCP consistency
    # --------------------------------------------------------------

    if (
        draft.expected_behavior.mcp_calls_max
        > 0
        and "MCP_TOOL"
        not in allowed
    ):

        problems.append(
            "mcp_calls_max is greater than 0 but "
            "MCP_TOOL is not allowed."
        )


    if (
        draft.expected_behavior.mcp_calls_max
        == 0
        and "MCP_TOOL"
        in allowed
    ):

        problems.append(
            "MCP_TOOL is allowed but mcp_calls_max is 0."
        )


    # --------------------------------------------------------------
    # External API consistency
    # --------------------------------------------------------------

    if (
        draft.expected_behavior.external_actions_max
        > 0
        and "EXTERNAL_API"
        not in allowed
        and "SEND_EMAIL"
        not in allowed
        and "SEND_MESSAGE"
        not in allowed
    ):

        problems.append(
            "external_actions_max is greater than 0 but "
            "no external-action capability is allowed."
        )


    # --------------------------------------------------------------
    # Basic read consistency
    # --------------------------------------------------------------

    if (
        draft.expected_behavior.document_reads_max
        > 0
        and "READ_DOCUMENT"
        not in allowed
        and "SEARCH"
        not in allowed
    ):

        problems.append(
            "document_reads_max is greater than 0 but "
            "no document-reading/search capability is allowed."
        )


    return problems


# ---------------------------------------------------------------------------
# Convert draft into Guardian baseline
# ---------------------------------------------------------------------------


def build_baseline(
    draft: AgentOnboardingDraft,
    agent_id: str,
) -> AgentBaseline:

    return AgentBaseline(
        agent_name=draft.agent_name,

        agent_id=agent_id,

        allowed_initiators=[
            "USER"
        ],

        expected_behavior=ExpectedBehavior(
            document_reads_max=(
                draft.expected_behavior
                .document_reads_max
            ),

            write_actions_max=(
                draft.expected_behavior
                .write_actions_max
            ),

            mcp_calls_max=(
                draft.expected_behavior
                .mcp_calls_max
            ),

            external_actions_max=(
                draft.expected_behavior
                .external_actions_max
            ),

            expected_output_sentences=(
                draft.expected_behavior
                .expected_output_sentences
            ),
        ),

        allowed_capabilities=sorted(
            {
                capability.upper()
                for capability
                in draft.allowed_capabilities
            }
        ),

        forbidden_capabilities=sorted(
            {
                capability.upper()
                for capability
                in draft.forbidden_capabilities
            }
        ),

        expected_steps=(
            draft.expected_steps
        ),

        description=(
            draft.description
        ),
    )


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def display_draft(
    app_id: str,
    owner: str,
    draft: AgentOnboardingDraft,
) -> None:

    print()

    print(
        "=" * 70
    )

    print(
        "GUARDIAN BEHAVIORAL CONTRACT - DRAFT"
    )

    print(
        "=" * 70
    )

    print(
        f"Application ID  : {app_id}"
    )

    print(
        f"Agent           : {draft.agent_name}"
    )

    print(
        f"Owner           : {owner}"
    )

    print(
        f"Risk Tier       : {draft.risk_tier}"
    )

    print()


    behavior = (
        draft.expected_behavior
    )


    print(
        "Expected Limits"
    )

    print(
        f"  Document Reads   : "
        f"{behavior.document_reads_max}"
    )

    print(
        f"  Write Actions    : "
        f"{behavior.write_actions_max}"
    )

    print(
        f"  MCP Calls        : "
        f"{behavior.mcp_calls_max}"
    )

    print(
        f"  External Actions : "
        f"{behavior.external_actions_max}"
    )

    print(
        f"  Output Sentences : "
        f"{behavior.expected_output_sentences}"
    )

    print()


    print(
        "Allowed Capabilities"
    )

    for item in sorted(
        draft.allowed_capabilities
    ):

        print(
            f"  + {item}"
        )


    print()

    print(
        "Forbidden Capabilities"
    )

    for item in sorted(
        draft.forbidden_capabilities
    ):

        print(
            f"  - {item}"
        )


    print()

    print(
        "Expected Steps"
    )

    for index, step in enumerate(
        draft.expected_steps,
        start=1,
    ):

        print(
            f"  {index}. {step}"
        )


    if draft.assumptions:

        print()

        print(
            "Assumptions"
        )

        for assumption in (
            draft.assumptions
        ):

            print(
                f"  ? {assumption}"
            )


    if draft.questions_for_operator:

        print()

        print(
            "Questions / Uncertainty"
        )

        for question in (
            draft.questions_for_operator
        ):

            print(
                f"  ? {question}"
            )


    print()

    print(
        "Description"
    )

    print(
        f"  {draft.description}"
    )

    print()


# ---------------------------------------------------------------------------
# Registration persistence
# ---------------------------------------------------------------------------


def register_agent(
    registry_path: Path,
    baseline_dir: Path,
    app_id: str,
    owner: str,
    draft: AgentOnboardingDraft,
) -> tuple[Path, str]:
    """
    Persist approved onboarding draft.

    Writes:

        baselines/<agent-id>.json

        registry/monitored_agents.json
    """

    registry = AgentRegistry(
        str(
            registry_path
        )
    )


    # --------------------------------------------------------------
    # App ID must not already belong to another registered agent.
    # --------------------------------------------------------------

    existing = registry.find_registration(
        application_id=app_id
    )


    if existing is not None:

        raise RuntimeError(
            "Application ID is already registered to: "
            f"{existing.agent_name}"
        )


    agent_id = _make_unique_agent_id(
        draft.agent_name,
        registry,
    )


    baseline = build_baseline(
        draft=draft,
        agent_id=agent_id,
    )


    baseline_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    baseline_path = (
        baseline_dir
        / f"{agent_id}.json"
    )


    # Store the repo-relative location in registry.
    registry_baseline_path = (
        baseline_path
        .as_posix()
    )


    now = datetime.now(
        timezone.utc
    )


    registration = MonitoredAgentRegistration(
        agent_id=agent_id,

        agent_name=draft.agent_name,

        enabled=True,

        owner=owner,

        risk_tier=draft.risk_tier,

        workflow_ids=[],

        application_ids=[
            app_id
        ],

        namespaces=[],

        baselines=[
            BaselineVersion(
                version="1.0",

                file=(
                    registry_baseline_path
                ),

                effective_from=now,

                effective_until=None,

                enabled=True,

                notes=(
                    "Initial baseline generated through "
                    "Guardian interactive onboarding."
                ),
            )
        ],

        description=(
            draft.description
        ),

        tags=[
            "glean",
            "guardian-monitored",
            "interactive-onboarding",
        ],
    )


    # --------------------------------------------------------------
    # Build updated registry document in memory FIRST.
    # --------------------------------------------------------------

    updated_document = RegistryDocument(
        registry_version=(
            registry.document.registry_version
        ),

        agents=[
            *registry.document.agents,
            registration,
        ],
    )


    # --------------------------------------------------------------
    # Write baseline first.
    #
    # An orphan baseline is less dangerous than a registry entry
    # pointing to a nonexistent baseline.
    # --------------------------------------------------------------

    _atomic_json_write(
        baseline_path,
        baseline.model_dump(
            mode="json",
            exclude_none=False,
        ),
    )


    # --------------------------------------------------------------
    # Backup current registry.
    # --------------------------------------------------------------

    backup_path = (
        registry_path
        .with_suffix(
            registry_path.suffix
            + ".bak"
        )
    )


    shutil.copy2(
        registry_path,
        backup_path,
    )


    # --------------------------------------------------------------
    # Write updated registry atomically.
    # --------------------------------------------------------------

    try:

        _atomic_json_write(
            registry_path,
            updated_document.model_dump(
                mode="json",
                exclude_none=False,
            ),
        )


        # ----------------------------------------------------------
        # Reload complete registry as final validation.
        # ----------------------------------------------------------

        validation_registry = (
            AgentRegistry(
                str(
                    registry_path
                )
            )
        )


        resolved = (
            validation_registry.resolve(
                application_id=app_id,
                run_timestamp=now,
            )
        )


        if (
            resolved.registration.agent_id
            != agent_id
        ):

            raise RuntimeError(
                "Post-write registry validation failed."
            )


    except Exception:

        # Restore registry if something went wrong.
        shutil.copy2(
            backup_path,
            registry_path,
        )

        raise


    return (
        baseline_path,
        agent_id,
    )


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------


def _atomic_json_write(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )


    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )


    temporary.replace(
        path
    )


# ---------------------------------------------------------------------------
# Agent ID
# ---------------------------------------------------------------------------


def _slugify(
    value: str,
) -> str:

    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    value = value.strip(
        "-"
    )


    return (
        value
        or "glean-agent"
    )


def _make_unique_agent_id(
    name: str,
    registry: AgentRegistry,
) -> str:

    base = _slugify(
        name
    )


    candidate = base

    counter = 2


    while registry.find_registration(
        agent_id=candidate
    ) is not None:

        candidate = (
            f"{base}-{counter}"
        )

        counter += 1


    return candidate


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def ask_yes_no(
    prompt: str,
) -> bool:

    answer = input(
        prompt
    ).strip().lower()


    return answer in {
        "y",
        "yes",
    }


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Register a Glean application with "
            "Guardian monitoring."
        )
    )


    parser.add_argument(
        "--app-id"
    )

    parser.add_argument(
        "--owner"
    )

    parser.add_argument(
        "--description"
    )


    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:

    load_dotenv()


    args = parse_args()


    print()

    print(
        "=" * 70
    )

    print(
        "GUARDIAN AGENT ONBOARDING"
    )

    print(
        "=" * 70
    )

    print()


    # --------------------------------------------------------------
    # Application ID
    # --------------------------------------------------------------

    app_id = (
        args.app_id
        or input(
            "Glean Application ID:\n> "
        ).strip()
    )


    if not app_id:

        print(
            "Application ID is required."
        )

        return 1


    # --------------------------------------------------------------
    # Check existing registration early.
    # --------------------------------------------------------------

    registry_path = Path(
        os.getenv(
            "AGENT_REGISTRY_FILE",
            "registry/monitored_agents.json",
        )
    )


    registry = AgentRegistry(
        str(
            registry_path
        )
    )


    existing = registry.find_registration(
        application_id=app_id
    )


    if existing:

        print()

        print(
            "This application is already registered."
        )

        print(
            f"Agent      : {existing.agent_name}"
        )

        print(
            f"Owner      : {existing.owner}"
        )

        print(
            f"Risk Tier  : {existing.risk_tier}"
        )

        print(
            f"Monitoring : "
            f"{'ENABLED' if existing.enabled else 'DISABLED'}"
        )

        return 0


    # --------------------------------------------------------------
    # Owner
    # --------------------------------------------------------------

    default_owner = os.getenv(
        "REGISTRATION_DEFAULT_OWNER",
        "Demo Team",
    )


    if args.owner:

        owner = args.owner

    else:

        entered_owner = input(
            f"Owner [{default_owner}]:\n> "
        ).strip()

        owner = (
            entered_owner
            or default_owner
        )


    # --------------------------------------------------------------
    # Plain-English intended behavior
    # --------------------------------------------------------------

    if args.description:

        description = (
            args.description
        )

    else:

        print()

        print(
            "Describe what this agent is supposed to do."
        )

        print(
            "Include important reads, writes, external actions, "
            "MCP usage, and anything it must NEVER do."
        )

        print()

        description = input(
            "> "
        ).strip()


    if not description:

        print(
            "Behavior description is required."
        )

        return 1


    # --------------------------------------------------------------
    # Draft
    # --------------------------------------------------------------

    print()

    print(
        "Guardian is drafting a behavioral contract..."
    )


    try:

        draft = create_draft(
            app_id=app_id,
            owner=owner,
            description=description,
        )

    except Exception as exc:

        print(
            f"Could not create onboarding draft: {exc}"
        )

        return 2


    # --------------------------------------------------------------
    # Deterministic validation
    # --------------------------------------------------------------

    problems = validate_draft(
        draft
    )


    display_draft(
        app_id=app_id,
        owner=owner,
        draft=draft,
    )


    if problems:

        print(
            "=" * 70
        )

        print(
            "DRAFT VALIDATION FAILED"
        )

        print(
            "=" * 70
        )


        for problem in problems:

            print(
                f"- {problem}"
            )


        print()

        print(
            "The draft was NOT registered."
        )

        return 3


    # --------------------------------------------------------------
    # Explicit human approval
    # --------------------------------------------------------------

    approved = ask_yes_no(
        "Register this agent with Guardian? [y/N]: "
    )


    if not approved:

        print()

        print(
            "Registration cancelled."
        )

        return 0


    # --------------------------------------------------------------
    # Persist
    # --------------------------------------------------------------

    baseline_dir = Path(
        os.getenv(
            "BASELINE_DIR",
            "baselines",
        )
    )


    try:

        (
            baseline_path,
            agent_id,
        ) = register_agent(
            registry_path=registry_path,

            baseline_dir=baseline_dir,

            app_id=app_id,

            owner=owner,

            draft=draft,
        )

    except Exception as exc:

        print()

        print(
            f"Registration failed: {exc}"
        )

        return 4


    # --------------------------------------------------------------
    # Success
    # --------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "REGISTERED"
    )

    print(
        "=" * 70
    )

    print(
        f"Agent       : {draft.agent_name}"
    )

    print(
        f"Agent ID    : {agent_id}"
    )

    print(
        f"Application : {app_id}"
    )

    print(
        f"Owner       : {owner}"
    )

    print(
        f"Risk Tier   : {draft.risk_tier}"
    )

    print(
        f"Baseline    : {baseline_path}"
    )

    print(
        "Version     : 1.0"
    )

    print(
        "Monitoring  : ENABLED"
    )

    print()

    print(
        "The live monitor will load this registration "
        "on its next registry refresh."
    )

    print()


    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
