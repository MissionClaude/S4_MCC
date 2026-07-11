"""Unit tests for the conversation-message models and StopReason enum.

The agentic loop relies on these invariants:

* ``StopReason.is_terminal`` is the only signal the loop uses to decide
  when to stop iterating.
* Assistant messages carry an ordered list of typed content blocks
  (text or tool-use).
* Tool-result messages round-trip to the wire format the Anthropic SDK
  expects.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
    message_to_wire,
)
from scenario_4_dev_productivity.models.tools import ToolCall, ToolResult


class TestStopReason:
    def test_tool_use_is_not_terminal(self) -> None:
        assert StopReason.TOOL_USE.is_terminal is False

    def test_end_turn_is_terminal(self) -> None:
        assert StopReason.END_TURN.is_terminal is True

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            (StopReason.END_TURN, True),
            (StopReason.TOOL_USE, False),
        ],
    )
    def test_terminal_matrix(self, reason: StopReason, expected: bool) -> None:
        assert reason.is_terminal is expected


class TestMessageRole:
    def test_values_match_wire(self) -> None:
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestUserMessage:
    def test_text_content(self) -> None:
        msg = UserMessage(content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_tool_result_content(self) -> None:
        msg = UserMessage(content=[ToolResult.success("t1", "ok")])
        assert isinstance(msg.content, list)
        assert msg.content[0].tool_use_id == "t1"


class TestAssistantMessage:
    def test_tool_calls_collects_only_tool_use_blocks(self) -> None:
        msg = AssistantMessage(
            content=[
                TextBlock(text="I'll read the file."),
                ToolUseBlock(call=ToolCall(id="t1", name="Read", input={"path": "x.py"})),
            ]
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "Read"

    def test_text_concatenates_in_order(self) -> None:
        msg = AssistantMessage(content=[TextBlock(text="hello "), TextBlock(text="world")])
        assert msg.text == "hello world"

    def test_tool_calls_empty_when_only_text(self) -> None:
        msg = AssistantMessage(content=[TextBlock(text="just text")])
        assert msg.tool_calls == []

    def test_default_stop_reason_is_none(self) -> None:
        msg = AssistantMessage(content=[])
        assert msg.stop_reason is None

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AssistantMessage(content=[], extra_field="oops")  # type: ignore[call-arg]


class TestToolResultMessage:
    def test_to_user_message_preserves_results(self) -> None:
        original = ToolResultMessage(
            results=[
                ToolResult.success("t1", "ok"),
                ToolResult.failure("t2", "boom", category="transient", retryable=True),
            ]
        )
        user = original.to_user_message()
        assert user.role == "user"
        assert isinstance(user.content, list)
        assert len(user.content) == 2
        assert user.content[0].tool_use_id == "t1"
        assert user.content[1].is_error is True


class TestMessageToWire:
    def test_user_text(self) -> None:
        assert message_to_wire(UserMessage(content="hi")) == {"role": "user", "content": "hi"}

    def test_user_tool_results(self) -> None:
        msg = UserMessage(
            content=[
                ToolResult.success("t1", "ok"),
                ToolResult.failure("t2", "bad", category="validation", retryable=False),
            ]
        )
        wire = message_to_wire(msg)
        assert wire == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": False},
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "bad",
                    "is_error": True,
                },
            ],
        }

    def test_assistant_with_text_and_tool_use(self) -> None:
        msg = AssistantMessage(
            content=[
                TextBlock(text="Reading..."),
                ToolUseBlock(call=ToolCall(id="t1", name="Read", input={"path": "x"})),
            ],
            stop_reason=StopReason.TOOL_USE,
        )
        # stop_reason is API OUTPUT — never serialised back as input
        assert message_to_wire(msg) == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading..."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "x"}},
            ],
        }

    def test_assistant_without_stop_reason_omits_field(self) -> None:
        msg = AssistantMessage(content=[TextBlock(text="hi")])
        wire = message_to_wire(msg)
        assert "stop_reason" not in wire

    def test_tool_result_message_routes_through_to_user_message(self) -> None:
        msg = ToolResultMessage(results=[ToolResult.success("t1", "ok")])
        assert message_to_wire(msg) == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": False}
            ],
        }
