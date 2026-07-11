"""Anthropic API request/response wrappers and structured errors.

These types are the contract between the agentic loop (Phase 2) and the
Anthropic SDK. They give the loop a stable, domain-specific surface so
the rest of the system can be tested without mocking SDK internals.

The error hierarchy mirrors the three failure modes the spec calls out:

* :class:`APIError` — the base type. Carries an ``ErrorCategory`` so the
  loop can decide whether to retry, surface to the model, or fail fast.
* :class:`RateLimitError` — HTTP 429. The API client should apply
  exponential backoff and retry.
* :class:`AuthError` — HTTP 401/403. Not retryable; user must fix the key.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    Message,
    StopReason,
)
from scenario_4_dev_productivity.models.tools import ToolCall, ToolDefinition


class ErrorCategory(StrEnum):
    """Why an API call failed — drives the loop's recovery strategy."""

    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    VALIDATION = "validation"
    SERVER = "server"
    UNKNOWN = "unknown"

    @property
    def is_retryable(self) -> bool:
        """Whether the loop should re-issue the same request.

        Rate limits and transient network errors are retryable. Auth and
        validation errors will fail the same way every time, so retrying
        burns budget without changing the outcome.
        """
        return self in {ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMIT, ErrorCategory.SERVER}


class APIRequest(BaseModel):
    """A single request the loop sends to the Anthropic API.

    Holds everything the API needs to know: which model, the system
    prompt, the conversation so far, and the tools the model may use.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="Anthropic model identifier")
    messages: list[Message] = Field(
        default_factory=list,
        description="Full conversation history up to this point",
    )
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="Tools the model is allowed to call this turn",
    )
    system: str = Field(
        default="",
        description="System prompt for this turn",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Cap on tokens the model may generate this turn",
    )

    def to_wire(self) -> dict[str, Any]:
        """Render as the dict the Anthropic SDK accepts.

        Defers the message-shape conversion to ``message_to_wire`` in the
        messages module so the two stay in sync.
        """
        from scenario_4_dev_productivity.models.messages import message_to_wire

        return {
            "model": self.model,
            "system": self.system,
            "messages": [message_to_wire(m) for m in self.messages],
            "max_tokens": self.max_tokens,
            "tools": [t.to_anthropic_tool() for t in self.tools],
        }


class APIResponse(BaseModel):
    """A single response from the Anthropic API.

    Wraps the model's output in our domain types. The loop reads
    ``stop_reason`` to decide what to do next and pulls ``tool_calls``
    from the message when ``stop_reason == TOOL_USE``.
    """

    model_config = ConfigDict(extra="forbid")

    message: AssistantMessage = Field(
        description="The assistant turn the model produced",
    )

    @property
    def stop_reason(self) -> StopReason | None:
        """Why the model stopped — None when the API didn't set one."""
        return self.message.stop_reason

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls the model wants executed (may be empty)."""
        return self.message.tool_calls

    @property
    def text(self) -> str:
        """Concatenated text the model produced (may be empty)."""
        return self.message.text


class APIError(Exception):
    """Base class for all API errors our code raises.

    Carries a structured :class:`ErrorCategory` so the loop can decide
    how to react without inspecting string messages. ``retry_after``
    is set when the API tells us how long to back off.
    """

    category: ErrorCategory = ErrorCategory.UNKNOWN
    retry_after: float | None = None

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        self.retry_after = retry_after
        self.message = message  # public read-only handle for the loop

    @property
    def is_retryable(self) -> bool:
        """Whether the loop should retry this same request."""
        return self.category.is_retryable


class RateLimitError(APIError):
    """HTTP 429 — too many requests. Always retryable with backoff."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None) -> None:
        super().__init__(message, category=ErrorCategory.RATE_LIMIT, retry_after=retry_after)


class AuthError(APIError):
    """HTTP 401/403 — invalid or missing API key. Not retryable."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, category=ErrorCategory.AUTH)
