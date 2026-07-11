"""Integration tests for error scenarios.

The spec calls out three error-handling invariants:

* transient API failures (429, 5xx, timeout) retry with backoff;
* auth failures (401, 403) do NOT retry;
* tool execution failures feed back to the model as structured
  :class:`ToolResult` instances so it can adapt.

These tests use the real production client, registry, and loop,
but inject failures at the SDK boundary so the retry / no-retry /
error-propagation paths run end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from scenario_4_dev_productivity.agents import ExploreAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.api import (
    APIError,
    APIRequest,
    AuthError,
    ErrorCategory,
)
from scenario_4_dev_productivity.models.messages import (
    StopReason,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema
from scenario_4_dev_productivity.tools.read_tool import ReadTool
from tests.test_integration.conftest import _text_response, _tool_response

# -- helpers --------------------------------------------------------------


def _sdk_message(text: str = "ok", stop_reason: str = "end_turn") -> MagicMock:
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg.content = [block]
    msg.stop_reason = stop_reason
    return msg


def _status_error(
    status: int, message: str = "boom", retry_after: str | None = None
) -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    response = httpx.Response(status_code=status, request=request, headers=headers)
    return APIStatusError(message, response=response, body=None)


def _build_real_client(sdk: MagicMock) -> AnthropicClient:
    """Build a real AnthropicClient with a mock SDK wired in."""
    sleeps: list[float] = []
    client = AnthropicClient(
        api_key="sk-test",
        max_retries=3,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.01,
        timeout_seconds=1.0,
        clock=sleeps.append,
    )
    client._client = sdk
    return client


def _request() -> APIRequest:
    return APIRequest(
        model="claude-haiku-4-5",
        system="you are a test",
        messages=[UserMessage(content="hello")],
        tools=[
            ToolDefinition(
                name="Read",
                description="Read a file",
                parameters=ToolParameterSchema(
                    properties={"path": {"type": "string"}}, required=["path"]
                ),
            )
        ],
        max_tokens=256,
    )


# -- Rate limit retry -----------------------------------------------------


class TestRateLimitRetry:
    def test_429_triggers_exponential_backoff(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = [
            _status_error(429, "rate limited", retry_after="0.1"),
            _sdk_message("done"),
        ]
        client = _build_real_client(sdk)
        response = client.send(_request())
        assert response.text == "done"
        assert sdk.messages.create.call_count == 2

    def test_gives_up_after_max_retries(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = _status_error(500, "boom")
        client = _build_real_client(sdk)
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.SERVER
        # max_retries=3 means 4 total attempts.
        assert sdk.messages.create.call_count == 4

    def test_429_with_retry_after_uses_server_hint(self) -> None:
        """The server's Retry-After hint takes precedence over the
        exponential backoff schedule."""
        sdk = MagicMock()
        sleeps: list[float] = []

        client = AnthropicClient(
            api_key="sk-test",
            max_retries=2,
            initial_backoff_seconds=10.0,  # would dominate without the hint
            max_backoff_seconds=20.0,
            clock=sleeps.append,
        )
        client._client = sdk
        sdk.messages.create.side_effect = [
            _status_error(429, "rate limited", retry_after="0.5"),
            _sdk_message("done"),
        ]
        response = client.send(_request())
        assert response.text == "done"
        # The sleep was the server's hint, not the 10s exponential.
        assert sleeps == [pytest.approx(0.5)]


# -- Auth failure: no retry -----------------------------------------------


class TestAuthFailureNoRetry:
    def test_401_raises_immediately(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = _status_error(401, "bad key")
        client = _build_real_client(sdk)
        with pytest.raises(AuthError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.AUTH
        # No retry — the SDK was called exactly once.
        assert sdk.messages.create.call_count == 1

    def test_403_raises_immediately(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = _status_error(403, "forbidden")
        client = _build_real_client(sdk)
        with pytest.raises(AuthError):
            client.send(_request())
        assert sdk.messages.create.call_count == 1


# -- Network errors -------------------------------------------------------


class TestNetworkRetry:
    def test_connection_error_retries(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            _sdk_message("done"),
        ]
        client = _build_real_client(sdk)
        response = client.send(_request())
        assert response.text == "done"
        assert sdk.messages.create.call_count == 2

    def test_timeout_retries_then_raises(self) -> None:
        sdk = MagicMock()
        sdk.messages.create.side_effect = APITimeoutError(request=MagicMock())
        client = _build_real_client(sdk)
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.TRANSIENT
        assert sdk.messages.create.call_count == 4  # 1 + 3 retries


# -- Tool failure feeds back to model -------------------------------------


class TestToolFailureFeedback:
    def test_tool_failure_appended_to_conversation(
        self, fake_client, registry
    ) -> None:
        """A tool failure is fed back to the model as a structured
        ToolResult with is_error=True. The model can then adapt."""
        from scenario_4_dev_productivity.models.messages import UserMessage

        registry.register(ReadTool())
        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        fake_client.script(
            _tool_response("c1", "Read", {"path": "nope.py"}),
            _text_response("I see — the file is missing. Stopping.", StopReason.END_TURN),
        )
        result = agent.run("Read nope.py")
        assert "missing" in result.text.lower() or "stopping" in result.text.lower()
        # The second request to the model carried the tool result.
        second_request: APIRequest = fake_client.requests[1]
        last = second_request.messages[-1]
        if isinstance(last, UserMessage) and isinstance(last.content, list):
            assert last.content[0].is_error is True
            assert "nope" in last.content[0].content
        else:  # pragma: no cover - defensive
            pytest.fail("expected tool result message in the second request")

    def test_loop_survives_unknown_tool_then_completes(
        self, fake_client, registry
    ) -> None:
        """A hallucinated tool name produces a structured failure; the
        loop keeps running and the model can self-correct."""
        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        fake_client.script(
            _tool_response("c1", "GhostTool", {}),
            _text_response("Self-corrected", StopReason.END_TURN),
        )
        result = agent.run("Do something")
        assert result.text == "Self-corrected"


# -- Max turns guard ------------------------------------------------------


class TestMaxTurnsGuard:
    def test_max_turns_raises(self, fake_client) -> None:
        from scenario_4_dev_productivity.loop.engine import MaxTurnsExceeded
        from scenario_4_dev_productivity.tools.registry import ToolRegistry

        reg = ToolRegistry()
        agent = ExploreAgent(registry=reg, client=fake_client, max_turns=2)
        # Always emit tool_use → the loop bails out at the cap.
        # The fake client returns the default end_turn when the script
        # drains, so we need to keep emitting tool_use forever; using
        # a side_effect lambda that ignores the request.
        from scenario_4_dev_productivity.models.api import APIResponse
        from scenario_4_dev_productivity.models.messages import (
            AssistantMessage,
            StopReason,
            ToolUseBlock,
        )
        from scenario_4_dev_productivity.models.tools import ToolCall

        def _always_tool_use(_req: APIRequest) -> APIResponse:
            return APIResponse(
                message=AssistantMessage(
                    content=[
                        ToolUseBlock(call=ToolCall(id="loop", name="Read", input={}))
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
            )

        fake_client.send = _always_tool_use  # type: ignore[assignment]
        with pytest.raises(MaxTurnsExceeded):
            agent.run("loop forever")
