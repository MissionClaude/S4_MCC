"""Unit tests for the API request/response wrappers and error hierarchy.

These types are the contract between the agentic loop (Phase 2) and the
Anthropic SDK. The tests pin the ``ErrorCategory.is_retryable`` matrix
because that single boolean drives the loop's recovery strategy.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scenario_4_dev_productivity.models.api import (
    APIError,
    APIRequest,
    APIResponse,
    AuthError,
    ErrorCategory,
    RateLimitError,
)
from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    StopReason,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import (
    ToolCall,
    ToolDefinition,
)


class TestErrorCategory:
    @pytest.mark.parametrize(
        ("category", "retryable"),
        [
            (ErrorCategory.TRANSIENT, True),
            (ErrorCategory.RATE_LIMIT, True),
            (ErrorCategory.SERVER, True),
            (ErrorCategory.AUTH, False),
            (ErrorCategory.VALIDATION, False),
            (ErrorCategory.UNKNOWN, False),
        ],
    )
    def test_retryable_matrix(self, category: ErrorCategory, retryable: bool) -> None:
        assert category.is_retryable is retryable


class TestAPIError:
    def test_default_category_is_unknown(self) -> None:
        err = APIError("boom")
        assert err.category is ErrorCategory.UNKNOWN
        assert err.is_retryable is False
        assert err.retry_after is None

    def test_message_is_preserved(self) -> None:
        err = APIError("the api is angry")
        assert "the api is angry" in str(err)
        assert err.message == "the api is angry"

    def test_custom_category(self) -> None:
        err = APIError("flaky", category=ErrorCategory.TRANSIENT, retry_after=1.5)
        assert err.category is ErrorCategory.TRANSIENT
        assert err.is_retryable is True
        assert err.retry_after == 1.5


class TestRateLimitError:
    def test_is_retryable(self) -> None:
        err = RateLimitError()
        assert err.category is ErrorCategory.RATE_LIMIT
        assert err.is_retryable is True

    def test_captures_retry_after(self) -> None:
        err = RateLimitError("slow down", retry_after=30.0)
        assert err.retry_after == 30.0


class TestAuthError:
    def test_is_not_retryable(self) -> None:
        err = AuthError()
        assert err.category is ErrorCategory.AUTH
        assert err.is_retryable is False


class TestAPIRequest:
    def test_defaults(self) -> None:
        req = APIRequest(model="claude-haiku")
        assert req.messages == []
        assert req.tools == []
        assert req.system == ""
        assert req.max_tokens == 4096

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            APIRequest(model="")

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            APIRequest(model="claude-haiku", max_tokens=0)

    def test_to_wire_shape(self) -> None:
        req = APIRequest(
            model="claude-haiku",
            system="You are helpful.",
            messages=[UserMessage(content="hi")],
            tools=[ToolDefinition(name="Read", description="Read a file")],
        )
        wire = req.to_wire()
        assert wire == {
            "model": "claude-haiku",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                }
            ],
        }


class TestAPIResponse:
    def test_stop_reason_and_tool_calls_delegate(self) -> None:
        msg = AssistantMessage(
            content=[
                TextBlock(text="Calling tool"),
                ToolUseBlock(call=ToolCall(id="t1", name="Read", input={})),
            ],
            stop_reason=StopReason.TOOL_USE,
        )
        response = APIResponse(message=msg)
        assert response.stop_reason is StopReason.TOOL_USE
        assert len(response.tool_calls) == 1
        assert response.text == "Calling tool"

    def test_end_turn_response_has_no_tool_calls(self) -> None:
        msg = AssistantMessage(
            content=[TextBlock(text="done")],
            stop_reason=StopReason.END_TURN,
        )
        response = APIResponse(message=msg)
        assert response.stop_reason is StopReason.END_TURN
        assert response.tool_calls == []
        assert response.text == "done"

    def test_stop_reason_can_be_none(self) -> None:
        response = APIResponse(message=AssistantMessage(content=[]))
        assert response.stop_reason is None
