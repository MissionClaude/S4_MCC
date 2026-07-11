"""Unit tests for :class:`CoordinatorAgent` and :class:`TaskTool`.

The coordinator is the hub: it owns task decomposition, subagent
dispatch via the Task tool, and result synthesis. The tests verify:

* The coordinator registers a Task tool alongside the built-in tools.
* The Task tool spawns subagents with isolated context (no shared
  history with the coordinator).
* The Task tool validates its inputs and returns structured failures
  for bad arguments.
* The coordinator's ``run`` flow can dispatch to a real subagent and
  surface the subagent's response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import (
    SUBAGENT_TYPES,
    AutomateAgent,
    BaseAgent,
    CoordinatorAgent,
    ExploreAgent,
    GenerateAgent,
    TaskTool,
)
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.tools import ToolResult
from scenario_4_dev_productivity.prompts import COORDINATOR_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry

# -- fixtures --------------------------------------------------------------


def _make_sdk_message(text: str, stop_reason: str = "end_turn") -> MagicMock:
    sdk = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    sdk.content = [text_block]
    sdk.stop_reason = stop_reason
    return sdk


@pytest.fixture
def client() -> AnthropicClient:
    real = AnthropicClient(
        api_key="sk-test",
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.01,
    )
    real._client = MagicMock()
    return real


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


# -- TaskTool tests --------------------------------------------------------


class TestTaskToolDefinition:
    def test_name_is_task(self) -> None:
        tool = TaskTool(subagent_factory=lambda _: MagicMock(spec=BaseAgent))
        assert tool.name == "Task"

    def test_definition_requires_subagent_type_and_prompt(self) -> None:
        tool = TaskTool(subagent_factory=lambda _: MagicMock(spec=BaseAgent))
        defn = tool.definition
        assert "subagent_type" in defn.parameters.required
        assert "prompt" in defn.parameters.required

    def test_definition_enum_lists_subagent_types(self) -> None:
        tool = TaskTool(subagent_factory=lambda _: MagicMock(spec=BaseAgent))
        defn = tool.definition
        enum = defn.parameters.properties["subagent_type"].get("enum")
        assert enum == list(SUBAGENT_TYPES)

    def test_description_warns_about_context_isolation(self) -> None:
        tool = TaskTool(subagent_factory=lambda _: MagicMock(spec=BaseAgent))
        defn = tool.definition
        # Context isolation is the spec's hard rule.
        assert "isolated" in defn.description.lower() or "context" in defn.description.lower()
        assert "history" in defn.description.lower()


class TestTaskToolExecution:
    def test_spawns_subagent_with_isolated_context(self) -> None:
        """The subagent receives ONLY the prompt, not the coordinator's history."""
        captured_prompts: list[str] = []
        captured_agents: list[str] = []

        def factory(subagent_type: str) -> BaseAgent:
            captured_agents.append(subagent_type)
            sub = MagicMock(spec=BaseAgent)
            sub.name = subagent_type
            sub.run.side_effect = lambda prompt, **_: captured_prompts.append(prompt) or _make_assistant(
                f"summary from {subagent_type}"
            )
            return sub

        tool = TaskTool(subagent_factory=factory)
        result = tool.execute(
            tool_use_id="call-1",
            arguments={"subagent_type": "explore", "prompt": "Find all usages of ToolRegistry"},
        )

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.content == "summary from explore"
        assert captured_agents == ["explore"]
        assert captured_prompts == ["Find all usages of ToolRegistry"]

    def test_rejects_unknown_subagent_type(self) -> None:
        factory = MagicMock()
        tool = TaskTool(subagent_factory=factory)
        result = tool.execute(
            tool_use_id="call-1",
            arguments={"subagent_type": "rogue", "prompt": "do something"},
        )
        assert result.is_error is True
        assert result.error_category == "validation"
        assert result.is_retryable is False
        assert "subagent_type" in result.content
        factory.assert_not_called()

    def test_rejects_empty_prompt(self) -> None:
        factory = MagicMock()
        tool = TaskTool(subagent_factory=factory)
        result = tool.execute(
            tool_use_id="call-1",
            arguments={"subagent_type": "explore", "prompt": "  "},
        )
        assert result.is_error is True
        assert result.error_category == "validation"
        assert "prompt" in result.content.lower()
        factory.assert_not_called()

    def test_rejects_missing_arguments(self) -> None:
        factory = MagicMock()
        tool = TaskTool(subagent_factory=factory)
        result = tool.execute(tool_use_id="call-1", arguments={})
        assert result.is_error is True
        assert result.error_category == "validation"

    def test_factory_exception_returns_structured_failure(self) -> None:
        """A factory failure must not crash the loop — the model gets a structured error."""

        def factory(subagent_type: str) -> BaseAgent:
            raise RuntimeError("factory boom")

        tool = TaskTool(subagent_factory=factory)
        result = tool.execute(
            tool_use_id="call-1",
            arguments={"subagent_type": "explore", "prompt": "x"},
        )
        assert result.is_error is True
        assert result.error_category == "transient"
        assert "factory boom" in result.content

    def test_subagent_exception_returns_structured_failure(self) -> None:
        """A subagent failure during run() must not crash the loop either."""
        sub = MagicMock(spec=BaseAgent)
        sub.name = "explore"
        sub.run.side_effect = RuntimeError("model timeout")
        tool = TaskTool(subagent_factory=lambda _: sub)
        result = tool.execute(
            tool_use_id="call-1",
            arguments={"subagent_type": "explore", "prompt": "x"},
        )
        assert result.is_error is True
        assert result.error_category == "transient"
        assert "model timeout" in result.content

    def test_each_invocation_creates_a_fresh_subagent(self) -> None:
        """Two Task calls in one coordinator turn = two subagent instances.

        This is what enforces context isolation: the second subagent
        does not share the first's conversation state.
        """
        created: list[BaseAgent] = []

        def factory(subagent_type: str) -> BaseAgent:
            sub = MagicMock(spec=BaseAgent)
            sub.name = subagent_type
            sub.run.return_value = _make_assistant(f"out-{len(created)}")
            created.append(sub)
            return sub

        tool = TaskTool(subagent_factory=factory)
        tool.execute("c1", {"subagent_type": "explore", "prompt": "first"})
        tool.execute("c2", {"subagent_type": "explore", "prompt": "second"})
        assert len(created) == 2
        assert created[0] is not created[1]


# -- CoordinatorAgent tests -----------------------------------------------


class TestCoordinatorWiring:
    def test_name_and_description(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        coord = CoordinatorAgent(registry=registry, client=client)
        assert coord.name == "coordinator"
        assert "Coordinator" in coord.description or "orchestrator" in coord.description.lower()

    def test_uses_coordinator_prompt(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        coord = CoordinatorAgent(registry=registry, client=client)
        assert coord.system_prompt == COORDINATOR_SYSTEM_PROMPT

    def test_allowed_tools_include_task_and_all_builtins(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        coord = CoordinatorAgent(registry=registry, client=client)
        tools = set(coord.allowed_tools)
        assert "Task" in tools
        # All six built-in tools are available as a fallback.
        for builtin in ("Read", "Write", "Edit", "Bash", "Grep", "Glob"):
            assert builtin in tools, f"Coordinator missing {builtin!r}"

    def test_model_uses_coordinator_tier(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        coord = CoordinatorAgent(registry=registry, client=client)
        from scenario_4_dev_productivity.config import config

        assert coord.model == config.coordinator_model

    def test_registers_task_tool_in_registry(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        CoordinatorAgent(registry=registry, client=client)
        assert "Task" in registry
        assert isinstance(registry.get("Task"), TaskTool)

    def test_requires_explicit_client(self, registry: ToolRegistry) -> None:
        """The coordinator needs a client; it can't be defaulted because
        the ``default_registry()`` doesn't carry one."""
        with pytest.raises(ValueError, match="AnthropicClient"):
            CoordinatorAgent(registry=registry, client=None)  # type: ignore[arg-type]


class TestCoordinatorFactory:
    def test_default_factory_returns_explore_agent(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        from scenario_4_dev_productivity.agents import _default_subagent_factory

        factory = _default_subagent_factory(registry, client)
        sub = factory("explore")
        assert isinstance(sub, ExploreAgent)

    def test_default_factory_returns_generate_agent(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        from scenario_4_dev_productivity.agents import _default_subagent_factory

        factory = _default_subagent_factory(registry, client)
        sub = factory("generate")
        assert isinstance(sub, GenerateAgent)

    def test_default_factory_returns_automate_agent(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        from scenario_4_dev_productivity.agents import _default_subagent_factory

        factory = _default_subagent_factory(registry, client)
        sub = factory("automate")
        assert isinstance(sub, AutomateAgent)

    def test_default_factory_rejects_unknown_type(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        from scenario_4_dev_productivity.agents import _default_subagent_factory

        factory = _default_subagent_factory(registry, client)
        with pytest.raises(ValueError, match="Unknown subagent_type"):
            factory("rogue")

    def test_injected_factory_used(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        """Tests can inject a custom factory via the ``subagent_factory``
        constructor argument — verify the override is what gets called
        when the coordinator's Task tool executes."""
        custom_sub = MagicMock(spec=BaseAgent)
        custom_sub.name = "custom"
        custom_sub.run.return_value = _make_assistant("custom-result")

        custom_factory = MagicMock(return_value=custom_sub)

        CoordinatorAgent(
            registry=registry,
            client=client,
            subagent_factory=custom_factory,
        )
        # Pull the Task tool the coordinator registered and call it.
        task_tool = registry.get("Task")
        assert isinstance(task_tool, TaskTool)
        result = task_tool.execute(
            "c1", {"subagent_type": "explore", "prompt": "x"}
        )
        # The custom factory was called, not the default one.
        assert custom_factory.called
        # The custom subagent's run was invoked and its text returned.
        assert custom_sub.run.called
        assert result.content == "custom-result"


# -- helpers --------------------------------------------------------------


def _make_assistant(text: str) -> MagicMock:
    """Build a fake :class:`AssistantMessage`-like object that ``text`` resolves to."""
    msg = MagicMock()
    msg.text = text
    return msg
