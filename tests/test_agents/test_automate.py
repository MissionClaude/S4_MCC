"""Unit tests for :class:`AutomateAgent`.

Automate has a Bash-only allowlist: the agent runs shell commands but
cannot edit files or read code in-loop. The safety rails are in the
system prompt; this file only verifies the wiring.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import AutomateAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.prompts import AUTOMATE_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def client() -> AnthropicClient:
    real = AnthropicClient(
        api_key="sk-test",
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.01,
    )
    real._client = MagicMock()
    return real


class TestAutomateAgentWiring:
    def test_name_is_automate(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = AutomateAgent(registry, client)
        assert agent.name == "automate"

    def test_allowed_tools_is_bash_only(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = AutomateAgent(registry, client)
        assert agent.allowed_tools == ("Bash",)

    def test_does_not_have_write_read_grep_glob_or_edit(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = AutomateAgent(registry, client)
        for forbidden in ("Write", "Read", "Grep", "Glob", "Edit"):
            assert forbidden not in agent.allowed_tools

    def test_uses_automate_system_prompt(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = AutomateAgent(registry, client)
        assert agent.system_prompt == AUTOMATE_SYSTEM_PROMPT

    def test_model_uses_automate_tier(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = AutomateAgent(registry, client)
        from scenario_4_dev_productivity.config import config

        # Lean tier (claude-haiku) — shell automation is a routine task.
        assert agent.model == config.automate_model

    def test_description_mentions_safety(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = AutomateAgent(registry, client)
        assert "Bash" in agent.description
        # Safety rails are part of the design — surface them so the
        # coordinator knows what the agent can and cannot do.
        assert "safety" in agent.description.lower() or "no" in agent.description.lower()


class TestAutomateAgentBehavior:
    def test_run_dispatches_to_loop(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        sdk = client._client
        sdk_message = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ran tests"
        sdk_message.content = [text_block]
        sdk_message.stop_reason = "end_turn"
        sdk.messages.create.return_value = sdk_message

        agent = AutomateAgent(registry, client)
        result = agent.run("pytest tests/ -x")
        assert result.text == "ran tests"
        assert sdk.messages.create.call_count == 1
