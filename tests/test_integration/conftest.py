"""Shared fixtures for the integration tests.

The fixtures in this file wire real production components — agents,
loop, tools, registry, context modules — together while swapping the
Anthropic SDK for a scripted fake. That gives us "end-to-end" coverage
without paying the latency or the API-key tax of a live run.

Convention: fixtures that return a *reusable* component use the
``scope="session"`` hint only when the component is genuinely
stateless. The :class:`FakeAnthropicClient` and the helpers below
build fresh state per test to keep cases independent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import CoordinatorAgent
from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.api import APIRequest, APIResponse
from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    StopReason,
    TextBlock,
    ToolUseBlock,
)
from scenario_4_dev_productivity.models.tools import ToolCall
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class FakeAnthropicClient(AnthropicClient):
    """A scripted :class:`AnthropicClient` for integration tests.

    The fake bypasses the SDK entirely: every :meth:`send` call pops
    the next scripted response from an internal queue. When the queue
    runs dry, the fake returns a default ``end_turn`` response so a
    buggy test never hangs the suite.

    Tests can inspect :attr:`requests` to assert on what the loop
    sent (model, tools, messages, etc.) without mocking internals.
    """

    def __init__(self) -> None:
        # Bypass the parent constructor — we don't want a real SDK.
        self._scripted: list[APIResponse] = []
        self._default_response: APIResponse = APIResponse(
            message=AssistantMessage(
                content=[TextBlock(text="(fake: script empty)")],
                stop_reason=StopReason.END_TURN,
            )
        )
        self.requests: list[APIRequest] = []

    # -- scripting API ---------------------------------------------------

    def script(self, *responses: APIResponse) -> None:
        """Queue a sequence of responses; each :meth:`send` pops one."""
        self._scripted.extend(responses)

    def set_default(self, response: APIResponse) -> None:
        """Set the fallback response used when the script runs out."""
        self._default_response = response

    def reset(self) -> None:
        """Clear the script and recorded requests."""
        self._scripted.clear()
        self.requests.clear()

    # -- send ------------------------------------------------------------

    def send(self, request: APIRequest) -> APIResponse:
        self.requests.append(request)
        if self._scripted:
            return self._scripted.pop(0)
        return self._default_response

    # -- proxy any other attribute access back to the underlying client
    # (we never use these in tests, but keeps AttributeError away if
    # some production code accidentally touches ``self._client``). --

    @property
    def _client(self) -> Any:  # pragma: no cover - defensive
        return MagicMock()


def _text_response(text: str, stop: StopReason = StopReason.END_TURN) -> APIResponse:
    return APIResponse(
        message=AssistantMessage(content=[TextBlock(text=text)], stop_reason=stop),
    )


def _tool_response(
    call_id: str, tool: str, args: dict[str, Any]
) -> APIResponse:
    return APIResponse(
        message=AssistantMessage(
            content=[ToolUseBlock(call=ToolCall(id=call_id, name=tool, input=args))],
            stop_reason=StopReason.TOOL_USE,
        ),
    )


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    """A scripted :class:`AnthropicClient` — one per test."""
    return FakeAnthropicClient()


@pytest.fixture
def registry() -> ToolRegistry:
    """A fresh tool registry — keeps tests independent."""
    return ToolRegistry()


@pytest.fixture
def coordinator(
    registry: ToolRegistry, fake_client: FakeAnthropicClient
) -> CoordinatorAgent:
    """A real :class:`CoordinatorAgent` wired to the fake client.

    The agent's :class:`TaskTool` factory is the default one, so the
    subagents it spawns will all share ``fake_client`` and the same
    registry. That is fine for "do subagents share state" tests, and
    harmful for "do they NOT share state" tests — those should
    inject a custom factory.
    """
    return CoordinatorAgent(registry=registry, client=fake_client, max_turns=5)


@pytest.fixture
def make_subagent(
    registry: ToolRegistry, fake_client: FakeAnthropicClient
) -> Callable[[str], BaseAgent]:
    """Factory that returns a fresh :class:`BaseAgent` of the requested type.

    Mirrors the structure of the real ``_default_subagent_factory``
    but inlines it so integration tests don't depend on private
    internals.
    """
    from scenario_4_dev_productivity.agents import (
        AutomateAgent,
        ExploreAgent,
        GenerateAgent,
    )

    def _factory(subagent_type: str) -> BaseAgent:
        if subagent_type == "explore":
            return ExploreAgent(registry=registry, client=fake_client, max_turns=3)
        if subagent_type == "generate":
            return GenerateAgent(registry=registry, client=fake_client, max_turns=3)
        if subagent_type == "automate":
            return AutomateAgent(registry=registry, client=fake_client, max_turns=3)
        raise ValueError(f"unknown subagent_type: {subagent_type!r}")

    return _factory


@pytest.fixture
def tmp_scratchpad_path(tmp_path) -> Iterator[str]:
    """A scratchpad path under a fresh tmp dir; the file is NOT pre-created."""
    yield str(tmp_path / "scratchpad.md")
