"""Unit tests for :class:`GenerateAgent`.

Generate has a Write-only allowlist: the agent can only write
artifacts, never explore the codebase or run commands in-loop. All
context the agent needs must come through the prompt.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import GenerateAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.prompts import GENERATE_SYSTEM_PROMPT
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


class TestGenerateAgentWiring:
    def test_name_is_generate(self, registry: ToolRegistry, client: AnthropicClient) -> None:
        agent = GenerateAgent(registry, client)
        assert agent.name == "generate"

    def test_allowed_tools_is_write_only(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = GenerateAgent(registry, client)
        # Per the PR #3 scope: the agent has Write only. This forces
        # the coordinator to pass all context in the prompt.
        assert agent.allowed_tools == ("Write",)

    def test_does_not_have_read_or_bash_or_grep(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = GenerateAgent(registry, client)
        for forbidden in ("Read", "Bash", "Grep", "Glob", "Edit"):
            assert forbidden not in agent.allowed_tools

    def test_uses_generate_system_prompt(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = GenerateAgent(registry, client)
        assert agent.system_prompt == GENERATE_SYSTEM_PROMPT

    def test_model_uses_generate_tier(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = GenerateAgent(registry, client)
        from scenario_4_dev_productivity.config import config

        # Balanced tier (claude-sonnet) by default.
        assert agent.model == config.generate_model

    def test_description_says_write_only(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        agent = GenerateAgent(registry, client)
        assert "Write" in agent.description
        assert "Writes" in agent.description or "writes" in agent.description


class TestGenerateAgentBehavior:
    def test_run_dispatches_to_loop(
        self, registry: ToolRegistry, client: AnthropicClient
    ) -> None:
        sdk = client._client
        sdk_message = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "wrote file"
        sdk_message.content = [text_block]
        sdk_message.stop_reason = "end_turn"
        sdk.messages.create.return_value = sdk_message

        agent = GenerateAgent(registry, client)
        result = agent.run("Write tests/test_foo.py with one test")
        assert result.text == "wrote file"
        assert sdk.messages.create.call_count == 1
