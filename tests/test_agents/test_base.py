"""Unit tests for :class:`BaseAgent` and :func:`build_agent`.

The base class is a thin declarative wrapper. We test:

* Construction from an :class:`AgentConfig` wires model, system prompt,
  and tool allowlist correctly.
* The ``with_*`` builders return a new instance without mutating the
  original.
* :meth:`build_loop` returns a configured :class:`AgenticLoop`.
* :meth:`run` dispatches to the loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import BaseAgent, build_agent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.tools import AgentConfig
from scenario_4_dev_productivity.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    return reg


@pytest.fixture
def client() -> AnthropicClient:
    """An :class:`AnthropicClient` with a mock SDK that returns a single
    end-turn response saying 'hello'."""
    sdk = MagicMock()
    sdk_message = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "hello"
    sdk_message.content = [text_block]
    sdk_message.stop_reason = "end_turn"
    sdk.messages.create.return_value = sdk_message

    real = AnthropicClient(
        api_key="sk-test",
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.01,
    )
    real._client = sdk
    return real


def _config(**overrides: str | list[str]) -> AgentConfig:
    base: dict[str, str | list[str]] = {
        "name": "Explore",
        "description": "Explore the codebase",
        "system_prompt": "You are an ExploreAgent.",
        "allowed_tools": ["Read", "Grep"],
        "model": "claude-haiku-4-5",
    }
    base.update(overrides)
    return AgentConfig(**base)  # type: ignore[arg-type]


class TestConstruction:
    def test_basic_fields(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = BaseAgent(_config(), registry, client)
        assert agent.name == "Explore"
        assert agent.description == "Explore the codebase"
        assert agent.system_prompt == "You are an ExploreAgent."
        assert agent.allowed_tools == ("Read", "Grep")
        assert agent.model == "claude-haiku-4-5"

    def test_repr_does_not_leak_sensitive_data(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = BaseAgent(_config(), registry, client)
        text = repr(agent)
        assert "Explore" in text
        assert "claude-haiku-4-5" in text
        # No system prompt body — keeps the repr concise.
        assert "You are an ExploreAgent" not in text


class TestBuilders:
    def test_with_model_returns_new_instance(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        original = BaseAgent(_config(), registry, client)
        bumped = original.with_model("claude-sonnet-4-6")
        assert bumped is not original
        assert original.model == "claude-haiku-4-5"
        assert bumped.model == "claude-sonnet-4-6"

    def test_with_prompt(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        original = BaseAgent(_config(), registry, client)
        customised = original.with_prompt("New prompt")
        assert customised.system_prompt == "New prompt"
        assert original.system_prompt == "You are an ExploreAgent."

    def test_with_tools(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        original = BaseAgent(_config(), registry, client)
        narrowed = original.with_tools(["Read"])
        assert narrowed.allowed_tools == ("Read",)
        assert original.allowed_tools == ("Read", "Grep")

    def test_with_max_turns(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        original = BaseAgent(_config(model="claude-haiku-4-5"), registry, client, max_turns=15)
        assert original._max_turns == 15
        bumped = original.with_max_turns(50)
        assert bumped._max_turns == 50
        assert original._max_turns == 15


class TestBuildLoop:
    def test_loop_uses_agent_config(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = BaseAgent(_config(model="claude-sonnet-4-6"), registry, client, max_turns=7)
        loop = agent.build_loop()
        assert loop._model == "claude-sonnet-4-6"
        assert loop._max_turns == 7
        assert loop._system_prompt == "You are an ExploreAgent."

    def test_each_build_loop_is_fresh(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        """A new loop per call means concurrent runs don't share state."""
        agent = BaseAgent(_config(), registry, client)
        assert agent.build_loop() is not agent.build_loop()


class TestRun:
    def test_run_delegates_to_loop(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = BaseAgent(_config(), registry, client)
        result = agent.run("hi")
        assert result.text == "hello"
        # The client was hit exactly once (mock returns end_turn immediately).
        sdk_client = client._client
        assert sdk_client is not None
        assert sdk_client.messages.create.call_count == 1  # type: ignore[attr-defined]

    def test_run_passes_overrides_through(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = BaseAgent(_config(), registry, client)
        agent.run("hi", max_turns=1)
        loop = agent.build_loop()
        # The override was applied per-call, not persisted to the agent.
        assert loop._max_turns == 15


def test_build_agent_factory(registry: ToolRegistry, client: AnthropicClient) -> None:
    """The factory returns a :class:`BaseAgent` wired the same way."""
    agent = build_agent(_config(), registry, client, max_turns=3)
    assert isinstance(agent, BaseAgent)
    assert agent._max_turns == 3
    assert agent.registry is registry
    assert agent.client is client
