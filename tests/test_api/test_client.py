"""Unit tests for :class:`AnthropicClient`.

The real Anthropic SDK is wrapped, so we test the wrapper behaviour by
feeding it fake SDK responses and fake SDK errors. The wrapper is
deliberately test-friendly: the underlying ``Anthropic`` instance is
constructed in :meth:`__init__` from the API key, so we replace it
after construction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.api import (
    APIError,
    APIRequest,
    AuthError,
    ErrorCategory,
    RateLimitError,
)
from scenario_4_dev_productivity.models.messages import (
    StopReason,
    ToolResultMessage,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult

# -- helpers ------------------------------------------------------------


def _fake_message(
    content: list[Any],
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a mock SDK ``Message``."""
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = stop_reason
    return msg


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(id: str, name: str, input: dict[str, Any]) -> MagicMock:  # noqa: A002
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _status_error(
    status: int, message: str = "boom", retry_after: str | None = None
) -> APIStatusError:
    """Build a real :class:`APIStatusError` so the wrapper's mapping runs."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    response = httpx.Response(status_code=status, request=request, headers=headers)
    return APIStatusError(message, response=response, body=None)


def _sleep_recorder() -> tuple[Any, list[float]]:
    """Return a ``(clock, sleeps)`` pair that records but does not sleep."""
    sleeps: list[float] = []
    return sleeps.append, sleeps


def _client(clock: Any = None, max_retries: int = 3) -> tuple[AnthropicClient, MagicMock]:
    """Build a client with a mock SDK and a controllable clock."""
    sdk = MagicMock()
    sleeps: list[float] = []
    client = AnthropicClient(
        api_key="sk-test",
        max_retries=max_retries,
        initial_backoff_seconds=0.001,  # keep tests fast
        max_backoff_seconds=0.01,
        timeout_seconds=1.0,
        clock=sleeps.append if clock is None else clock,
    )
    client._client = sdk  # swap in mock
    return client, sdk


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


# -- tests --------------------------------------------------------------


class TestConstruction:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            AnthropicClient(api_key="")

    def test_rejects_negative_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AnthropicClient(api_key="k", max_retries=-1)

    def test_rejects_zero_initial_backoff(self) -> None:
        with pytest.raises(ValueError, match="initial_backoff"):
            AnthropicClient(api_key="k", initial_backoff_seconds=0)

    def test_rejects_backoff_smaller_than_initial(self) -> None:
        with pytest.raises(ValueError, match="max_backoff"):
            AnthropicClient(
                api_key="k",
                initial_backoff_seconds=2.0,
                max_backoff_seconds=1.0,
            )


class TestHappyPath:
    def test_text_response(self) -> None:
        client, sdk = _client()
        sdk.messages.create.return_value = _fake_message(
            [_text_block("hi")], stop_reason="end_turn"
        )
        response = client.send(_request())
        assert response.text == "hi"
        assert response.stop_reason is StopReason.END_TURN
        assert response.tool_calls == []

    def test_tool_use_response(self) -> None:
        client, sdk = _client()
        sdk.messages.create.return_value = _fake_message(
            [_tool_use_block("c1", "Read", {"path": "x.py"})],
            stop_reason="tool_use",
        )
        response = client.send(_request())
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "Read"
        assert response.stop_reason is StopReason.TOOL_USE

    def test_wire_format_uses_snake_case(self) -> None:
        client, sdk = _client()
        sdk.messages.create.return_value = _fake_message([_text_block("ok")])
        client.send(_request())
        kwargs = sdk.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["system"] == "you are a test"
        assert kwargs["max_tokens"] == 256
        assert kwargs["tools"][0]["name"] == "Read"
        assert "input_schema" in kwargs["tools"][0]


class TestToolResultMessageWire:
    def test_tool_result_message_serialised_as_user(self) -> None:
        client, sdk = _client()
        sdk.messages.create.return_value = _fake_message([_text_block("done")])
        request = _request()
        # Manually add a tool-result message; we cast through Any to
        # sidestep the ``Message`` discriminated union which excludes it.
        request.messages.append(
            ToolResultMessage(  # type: ignore[arg-type]
                results=[ToolResult.success("c1", "x")],
            )
        )
        client.send(request)
        sent = sdk.messages.create.call_args.kwargs["messages"]
        # The last sent message is the tool results, serialised as user.
        assert sent[-1]["role"] == "user"
        assert sent[-1]["content"][0]["type"] == "tool_result"
        assert sent[-1]["content"][0]["tool_use_id"] == "c1"


class TestRetry:
    def test_retries_on_429_then_succeeds(self) -> None:
        clock, sleeps = _sleep_recorder()
        client, sdk = _client(clock=clock)
        sdk.messages.create.side_effect = [
            _status_error(429, "rate limited", retry_after="0.1"),
            _fake_message([_text_block("ok")]),
        ]
        response = client.send(_request())
        assert response.text == "ok"
        assert sdk.messages.create.call_count == 2
        # The server-supplied hint should be respected, not exponential backoff.
        assert sleeps and sleeps[0] == pytest.approx(0.1)

    def test_retries_on_5xx_with_exponential_backoff(self) -> None:
        clock, sleeps = _sleep_recorder()
        client, sdk = _client(clock=clock)
        sdk.messages.create.side_effect = [
            _status_error(500, "boom"),
            _status_error(503, "boom"),
            _fake_message([_text_block("ok")]),
        ]
        response = client.send(_request())
        assert response.text == "ok"
        assert sdk.messages.create.call_count == 3
        # Exponential: each sleep should be > the previous one (modulo jitter).
        assert len(sleeps) == 2
        assert all(s > 0 for s in sleeps)

    def test_no_retry_on_401(self) -> None:
        client, sdk = _client()
        sdk.messages.create.side_effect = _status_error(401, "bad key")
        with pytest.raises(AuthError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.AUTH
        assert not exc.value.is_retryable
        assert sdk.messages.create.call_count == 1

    def test_gives_up_after_max_retries(self) -> None:
        client, sdk = _client(max_retries=2)
        sdk.messages.create.side_effect = _status_error(500, "boom")
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.SERVER
        # max_retries=2 means 3 total attempts.
        assert sdk.messages.create.call_count == 3

    def test_retries_on_timeout(self) -> None:
        client, sdk = _client()
        sdk.messages.create.side_effect = APITimeoutError(request=MagicMock())
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.TRANSIENT
        # One initial + max_retries attempts.
        assert sdk.messages.create.call_count == 4

    def test_retries_on_connection_error(self) -> None:
        client, sdk = _client()
        sdk.messages.create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is ErrorCategory.TRANSIENT
        assert sdk.messages.create.call_count == 4


class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected_category", "retryable"),
        [
            (400, ErrorCategory.VALIDATION, False),
            (404, ErrorCategory.VALIDATION, False),
            (408, ErrorCategory.TRANSIENT, True),
            (500, ErrorCategory.SERVER, True),
            (502, ErrorCategory.SERVER, True),
            (503, ErrorCategory.SERVER, True),
        ],
    )
    def test_status_code_mapping(
        self, status: int, expected_category: ErrorCategory, retryable: bool
    ) -> None:
        client, sdk = _client(max_retries=0)
        sdk.messages.create.side_effect = _status_error(status, "x")
        with pytest.raises(APIError) as exc:
            client.send(_request())
        assert exc.value.category is expected_category
        assert exc.value.is_retryable is retryable

    def test_rate_limit_uses_retry_after(self) -> None:
        client, sdk = _client(max_retries=0)
        sdk.messages.create.side_effect = _status_error(429, "limit", retry_after="7")
        with pytest.raises(RateLimitError) as exc:
            client.send(_request())
        assert exc.value.retry_after == 7.0

    def test_retry_after_garbage_falls_back_to_none(self) -> None:
        client, sdk = _client(max_retries=0)
        sdk.messages.create.side_effect = _status_error(429, "limit", retry_after="soonish")
        with pytest.raises(RateLimitError) as exc:
            client.send(_request())
        assert exc.value.retry_after is None


class TestStopReasonMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("end_turn", StopReason.END_TURN),
            ("tool_use", StopReason.TOOL_USE),
            ("max_tokens", StopReason.END_TURN),  # still terminal
            ("stop_sequence", StopReason.END_TURN),
            ("refusal", StopReason.END_TURN),
            (None, None),
        ],
    )
    def test_stop_reason_translation(self, raw: Any, expected: Any) -> None:
        client, sdk = _client()
        sdk.messages.create.return_value = _fake_message([_text_block("x")], stop_reason=raw)
        response = client.send(_request())
        assert response.stop_reason is expected
