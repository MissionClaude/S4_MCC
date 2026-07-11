"""CoordinatorAgent — the hub of the hub-and-spoke topology.

The Coordinator decomposes user tasks, dispatches subagents via the
Task tool, and synthesises their findings. It does **not** read files,
search code, or write code itself — those jobs belong to the Explore,
Generate, and Automate subagents.

## How Task spawning works

The Coordinator registers a :class:`TaskTool` alongside the built-in
tools. The Task tool takes three arguments:

* ``subagent_type`` — one of ``"explore"``, ``"generate"``, ``"automate"``
* ``prompt`` — the full prompt to pass to the subagent, with all
  necessary context (the subagent does NOT inherit the coordinator's
  conversation history)
* ``description`` — a one-line description of what the subagent will do

When the coordinator's model emits a ``Task`` tool call, the agentic
loop routes it to :class:`TaskTool.execute`, which:

1. Creates a fresh subagent instance via the factory (one per call, so
   the loop state stays clean)
2. Calls :meth:`BaseAgent.run` with the explicit prompt
3. Returns the subagent's final text as a :class:`ToolResult`

Context isolation is enforced by design: the subagent constructor takes
its own ``messages`` list starting from a single ``UserMessage`` with
the prompt. There is no path by which the coordinator's history leaks
into the subagent's conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.models.tools import (
    AgentConfig,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from scenario_4_dev_productivity.prompts.coordinator import COORDINATOR_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.base import BUILTIN_TOOLS, default_registry
from scenario_4_dev_productivity.tools.registry import ToolRegistry

# Subagent type literal — used in the Task tool schema and as the
# dispatch key in :class:`CoordinatorAgent`.
SUBAGENT_TYPES: tuple[str, ...] = ("explore", "generate", "automate")


class TaskTool:
    """Spawn a subagent with isolated context and return its summary.

    The Task tool is the **only** way the coordinator dispatches work.
    The model is told (in the coordinator system prompt) to use it
    instead of doing the work itself.

    The factory pattern is intentional: each call creates a fresh
    subagent so two consecutive Task calls don't share conversation
    state. The factory also keeps the test surface small — tests can
    inject a fake factory.
    """

    name: str = "Task"

    def __init__(
        self,
        subagent_factory: Callable[[str], BaseAgent],
    ) -> None:
        self._factory = subagent_factory

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Spawn a subagent with isolated context and return its "
                "summary.\n\n"
                "Use this to dispatch work to a specialized subagent. "
                "The subagent does NOT inherit your conversation history "
                "— the prompt you pass must contain all context the "
                "subagent needs (paths, constraints, prior findings).\n\n"
                "Input examples:\n"
                '  {"subagent_type": "explore", '
                '"prompt": "Find every place we use AnthropicClient and '
                'summarise the retry policy.", '
                '"description": "Audit retry policy"}\n'
                '  {"subagent_type": "generate", '
                '"prompt": "Write tests/test_foo.py with three cases: ...", '
                '"description": "Write tests for foo"}\n\n'
                "Boundary conditions:\n"
                "- The subagent runs in its own conversation; do not assume "
                "it can see any of your prior tool results.\n"
                "- Run independent sub-tasks in parallel by emitting "
                "multiple Task calls in the same response.\n"
                f"- Valid subagent_type values: {', '.join(SUBAGENT_TYPES)}"
            ),
            parameters=ToolParameterSchema(
                properties={
                    "subagent_type": {
                        "type": "string",
                        "enum": list(SUBAGENT_TYPES),
                        "description": (
                            "Which subagent to spawn. 'explore' for read-only "
                            "investigation, 'generate' for writing files, "
                            "'automate' for running shell commands."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Full prompt for the subagent. Must include every "
                            "piece of context the subagent needs (paths, "
                            "constraints, prior findings)."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One-line description of what the subagent will "
                            "do. Used in the coordinator's transcript."
                        ),
                    },
                },
                required=["subagent_type", "prompt"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        subagent_type = arguments.get("subagent_type")
        prompt = arguments.get("prompt", "")

        if not isinstance(subagent_type, str) or subagent_type not in SUBAGENT_TYPES:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"Task requires a valid 'subagent_type' "
                    f"(one of {', '.join(SUBAGENT_TYPES)}). "
                    f"Got: {subagent_type!r}."
                ),
                category="validation",
                retryable=False,
            )
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Task requires a non-empty 'prompt' string.",
                category="validation",
                retryable=False,
            )

        try:
            subagent = self._factory(subagent_type)
        except Exception as exc:  # noqa: BLE001 — factory failure must not crash the loop
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Could not spawn {subagent_type} subagent: {exc}",
                category="transient",
                retryable=False,
            )

        try:
            response = subagent.run(prompt)
        except Exception as exc:  # noqa: BLE001 — subagent failure must not crash the loop
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"{subagent_type} subagent failed: {type(exc).__name__}: {exc}"
                ),
                category="transient",
                retryable=False,
            )
        return ToolResult.success(tool_use_id=tool_use_id, content=response.text)


def _default_subagent_factory(registry: ToolRegistry, client: AnthropicClient) -> Callable[[str], BaseAgent]:
    """Build the default subagent factory used by :class:`CoordinatorAgent`.

    Imports are deferred to avoid a circular import (the agents import
    this module, and the factory imports the concrete agent classes).
    """
    from scenario_4_dev_productivity.agents.automate_agent import AutomateAgent
    from scenario_4_dev_productivity.agents.explore_agent import ExploreAgent
    from scenario_4_dev_productivity.agents.generate_agent import GenerateAgent

    def _factory(subagent_type: str) -> BaseAgent:
        if subagent_type == "explore":
            return ExploreAgent(registry=registry, client=client)
        if subagent_type == "generate":
            return GenerateAgent(registry=registry, client=client)
        if subagent_type == "automate":
            return AutomateAgent(registry=registry, client=client)
        raise ValueError(f"Unknown subagent_type: {subagent_type!r}")

    return _factory


class CoordinatorAgent(BaseAgent):
    """The hub of the hub-and-spoke topology.

    The coordinator registers a :class:`TaskTool` alongside the
    built-in tools so it can both dispatch subagents **and** fall back
    to direct tool use when the work is trivial.
    """

    DESCRIPTION: str = (
        "Orchestrator of the multi-agent developer productivity system. "
        "Decomposes tasks and dispatches them to specialized subagents "
        "(explore, generate, automate) via the Task tool. Has all "
        "built-in tools as a fallback for trivial work."
    )

    #: Tools the coordinator has access to: Task for delegation, plus
    #: every built-in tool for direct fallback. Names are matched
    #: against the registry at construction time.
    DEFAULT_TOOLS: tuple[str, ...] = (
        "Task",
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
    )

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        client: AnthropicClient | None = None,
        *,
        model: str | None = None,
        max_turns: int = 15,
        subagent_factory: Callable[[str], BaseAgent] | None = None,
    ) -> None:
        # The coordinator uses a *separate* registry from the shared
        # default one — we need the Task tool, which is not part of
        # BUILTIN_TOOLS. Start from the default registry (which has the
        # built-in tools) and add the Task tool.
        if registry is None:
            registry = default_registry()
        elif not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry or None")
        if client is None:
            raise ValueError(
                "CoordinatorAgent requires an AnthropicClient — pass one explicitly"
            )

        subagent_factory = subagent_factory or _default_subagent_factory(registry, client)
        task_tool = TaskTool(subagent_factory=subagent_factory)

        # Idempotent registration: if the caller already added a Task
        # tool to their registry, replace it with ours so the factory
        # closure stays in sync.
        registry.unregister("Task")
        registry.register(task_tool)

        agent_config = AgentConfig(
            name="coordinator",
            description=self.DESCRIPTION,
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            allowed_tools=list(self.DEFAULT_TOOLS),
            model=model or config.coordinator_model,
        )
        super().__init__(
            config=agent_config,
            registry=registry,
            client=client,
            max_turns=max_turns,
        )


__all__ = ["CoordinatorAgent", "TaskTool", "SUBAGENT_TYPES", "BUILTIN_TOOLS"]
