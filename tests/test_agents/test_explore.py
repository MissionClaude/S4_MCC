"""Unit tests for :class:`ExploreAgent`.

The agent is a thin subclass of :class:`BaseAgent`: it pins the
read-only tool allowlist, the explore model, and the explore system
prompt. We test the wiring (the public contract), not the prompt body.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import ExploreAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.prompts import EXPLORE_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def client() -> AnthropicClient:
    """An :class:`AnthropicClient` with a stub SDK."""
    real = AnthropicClient(
        api_key="sk-test",
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.01,
    )
    real._client = MagicMock()
    return real


class TestExploreAgentWiring:
    def test_name_is_explore(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = ExploreAgent(registry, client)
        assert agent.name == "explore"

    def test_allowed_tools_are_read_only(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        assert agent.allowed_tools == ("Read", "Grep", "Glob")

    def test_does_not_have_write_or_bash(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        assert "Write" not in agent.allowed_tools
        assert "Bash" not in agent.allowed_tools
        assert "Edit" not in agent.allowed_tools

    def test_uses_explore_system_prompt(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        assert agent.system_prompt == EXPLORE_SYSTEM_PROMPT

    def test_description_mentions_read_only(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        # Description surfaces in the Coordinator's Task tool list, so
        # it has to be specific about scope.
        assert "Read" in agent.description
        assert "Grep" in agent.description
        assert "Glob" in agent.description
        assert "no Bash" in agent.description or "no Write" in agent.description

    def test_model_uses_explore_tier(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        # From config.explore_model — the lean tier (claude-haiku).
        from scenario_4_dev_productivity.config import config

        assert agent.model == config.explore_model

    def test_model_override_is_respected(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client, model="claude-sonnet-4-6")
        assert agent.model == "claude-sonnet-4-6"

    def test_max_turns_override(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client, max_turns=5)
        assert agent._max_turns == 5

    def test_shares_registry_and_client(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client)
        # The hub-and-spoke topology means all subagents share the
        # same registry and client instance.
        assert agent.registry is registry
        assert agent.client is client


class TestExploreAgentBehavior:
    def test_run_invokes_loop_and_returns_text(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        """End-to-end: the agent's ``run`` dispatches to the loop and
        surfaces the model's text. The mock returns ``end_turn`` with
        a known text, so we can verify the wiring."""
        sdk = client._client
        sdk_message = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "explore-result"
        sdk_message.content = [text_block]
        sdk_message.stop_reason = "end_turn"
        sdk.messages.create.return_value = sdk_message

        agent = ExploreAgent(registry, client)
        result = agent.run("find usages of ToolRegistry")
        assert result.text == "explore-result"
        # The SDK was called exactly once (no tool_use dispatch).
        assert sdk.messages.create.call_count == 1

    def test_build_loop_uses_explore_config(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = ExploreAgent(registry, client, model="claude-sonnet-4-6", max_turns=3)
        loop = agent.build_loop()
        assert loop._model == "claude-sonnet-4-6"
        assert loop._max_turns == 3
        assert loop._system_prompt == EXPLORE_SYSTEM_PROMPT
