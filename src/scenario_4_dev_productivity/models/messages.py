"""Conversation message types and stop-reason enum.

The agentic loop works by sending a list of :class:`Message` objects to the
API and reacting to the ``stop_reason`` on the response. These types are
deliberately close to Anthropic's wire format so we can pass them through
with minimal conversion in the API client (Phase 2).

Two design constraints worth calling out:

1. ``StopReason`` is the **only** signal the loop uses to decide whether to
   keep going. Parsing assistant text for "DONE" or similar markers is the
   canonical anti-pattern in the exam guide — the loop will reject it.
2. Messages are tagged by role (user/assistant) and the assistant role
   carries a list of typed content blocks. We use a union (``TextBlock |
   ToolUseBlock``) rather than a free-form string so the loop can pattern
   match on block type when it needs to.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scenario_4_dev_productivity.models.tools import ToolCall, ToolResult


class MessageRole(StrEnum):
    """Conversation role tag — matches the Anthropic ``role`` field."""

    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    """Why the model stopped generating.

    Only two values matter for the agentic loop:

    * ``TOOL_USE`` — the assistant wants to call one or more tools; the
      loop should execute them, append results, and call the API again.
    * ``END_TURN`` — the assistant is done; the loop returns the final
      message to the caller.

    Any other value is treated as an unexpected condition by the loop.
    """

    TOOL_USE = "tool_use"
    END_TURN = "end_turn"

    @property
    def is_terminal(self) -> bool:
        """True when the loop should stop iterating."""
        return self is StopReason.END_TURN


class TextBlock(BaseModel):
    """A plain text content block from the assistant."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str = Field(default="", description="The text the model produced")


class ToolUseBlock(BaseModel):
    """A tool-use content block from the assistant.

    Wraps a :class:`ToolCall` so the conversation history tracks which
    tool calls are still awaiting results.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    call: ToolCall = Field(description="The tool call the model wants made")


# A discriminated union by the ``type`` field. The agentic loop
# pattern-matches on this when reading assistant content.
AssistantContent = Annotated[
    TextBlock | ToolUseBlock,
    Field(discriminator="type"),
]


class UserMessage(BaseModel):
    """A user turn — typically a task prompt or a tool result batch."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str | list[ToolResult] = Field(
        description="Either a plain text string or a batch of tool results",
    )


class AssistantMessage(BaseModel):
    """An assistant turn — produced by the model in response to prior input."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(
        default_factory=list,
        description="Ordered content blocks the model emitted this turn",
    )
    stop_reason: StopReason | None = Field(
        default=None,
        description="Why the model stopped; None until produced by the API",
    )

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Convenience accessor — all ``ToolUseBlock``s in this turn."""
        return [block.call for block in self.content if isinstance(block, ToolUseBlock)]

    @property
    def text(self) -> str:
        """Convenience accessor — concatenated text blocks."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


class ToolResultMessage(BaseModel):
    """A user turn whose only payload is tool results.

    Anthropic expects tool results to come back as a ``user`` message with
    ``tool_result`` blocks. We expose it as its own type so call sites
    read clearly (``ToolResultMessage(results=[...])``) while keeping the
    on-the-wire role as ``user``.

    This is a builder around :class:`UserMessage` and is *not* part of
    the :data:`Message` discriminated union — converting via
    :meth:`to_user_message` is the loop's responsibility.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    results: list[ToolResult] = Field(
        min_length=1,
        description="Tool results to send back to the model",
    )

    def to_user_message(self) -> UserMessage:
        """Convert to the wire-format user message."""
        return UserMessage(role="user", content=list(self.results))


# Convenience union used by the API client to type its message list.
# ``ToolResultMessage`` is excluded because both it and ``UserMessage``
# have ``role == "user"``; callers that hold a ``ToolResultMessage``
# must convert via ``to_user_message()`` before adding to a request.
Message = Annotated[
    UserMessage | AssistantMessage,
    Field(discriminator="role"),
]


def message_to_wire(message: UserMessage | AssistantMessage | ToolResultMessage) -> dict[str, Any]:
    """Render a message as the dict the Anthropic SDK expects on the wire.

    Centralised here so the API client (Phase 2) doesn't need to know about
    the Pydantic type hierarchy. Keeps Phase 1 free of any SDK imports.
    """
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content}
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in message.content
            ],
        }
    if isinstance(message, ToolResultMessage):
        return message_to_wire(message.to_user_message())

    # AssistantMessage
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        else:
            content.append(
                {
                    "type": "tool_use",
                    "id": block.call.id,
                    "name": block.call.name,
                    "input": block.call.input,
                }
            )
    # stop_reason is OUTPUT from the API — never send it back as input
    return {"role": "assistant", "content": content}
