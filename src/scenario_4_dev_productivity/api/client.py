"""Anthropic API client wrapper for the agentic loop.

The client owns the SDK lifecycle (one ``anthropic.Anthropic`` instance
per agent), retry/backoff policy, and the conversion between the
domain types (:class:`APIRequest`, :class:`APIResponse`) and the
SDK's wire format. The agentic loop sees only the domain types — it
never imports the SDK.

Failure handling:

* 401/403 → :class:`AuthError` (not retryable; caller surfaces to user).
* 429 → :class:`RateLimitError` with the server's ``Retry-After`` hint
  when available; the client retries with exponential backoff + jitter.
* 5xx → :class:`APIError` with category ``server`` (retryable).
* Network timeouts → :class:`APIError` with category ``transient`` (retryable).
* Anything else → :class:`APIError` with category ``unknown`` (not retryable).
"""

from __future__ import annotations

import random
import time
from typing import Any

from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError

from scenario_4_dev_productivity.models.api import (
    APIError,
    APIRequest,
    APIResponse,
    AuthError,
    ErrorCategory,
    RateLimitError,
)
from scenario_4_dev_productivity.models.messages import (
    AssistantContent,
    AssistantMessage,
    StopReason,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import ToolCall


class AnthropicClient:
    """Thin wrapper around the Anthropic SDK with retry and error mapping.

    The client is cheap to construct; create one per agent and reuse it
    for the lifetime of the loop. The underlying ``Anthropic`` instance
    is also reused so HTTP keep-alive works.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        timeout_seconds: float = 120.0,
        clock: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicClient requires a non-empty api_key")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")
        if max_backoff_seconds < initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")

        self._client = Anthropic(api_key=api_key, timeout=timeout_seconds)
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._timeout_seconds = timeout_seconds
        # Injectable clock for deterministic tests. ``time.sleep`` is
        # normally used; tests pass a stub that records the requested
        # sleep duration without actually sleeping.
        self._sleep: Any = clock if clock is not None else time.sleep

    # -- public API ------------------------------------------------------

    def send(self, request: APIRequest) -> APIResponse:
        """Send a request and return a parsed :class:`APIResponse`.

        Retries on 429 and 5xx with exponential backoff + jitter. Raises
        :class:`AuthError` immediately on 401/403 (no retry).
        """
        wire = self._request_to_wire(request)
        last_error: APIError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._client.messages.create(**wire)
            except APIStatusError as exc:
                mapped = self._map_status_error(exc)
                if not mapped.is_retryable or attempt == self._max_retries:
                    raise mapped from exc
                last_error = mapped
                self._sleep_backoff(attempt, mapped.retry_after)
                continue
            except APITimeoutError as exc:
                mapped = APIError(
                    f"Anthropic request timed out after {self._timeout_seconds:.0f}s",
                    category=ErrorCategory.TRANSIENT,
                )
                if attempt == self._max_retries:
                    raise mapped from exc
                last_error = mapped
                self._sleep_backoff(attempt, None)
                continue
            except APIConnectionError as exc:
                mapped = APIError(
                    f"Network error contacting Anthropic: {exc}",
                    category=ErrorCategory.TRANSIENT,
                )
                if attempt == self._max_retries:
                    raise mapped from exc
                last_error = mapped
                self._sleep_backoff(attempt, None)
                continue
            else:
                return self._parse_response(raw)

        # Defensive — the loop above either returns or raises.
        raise last_error or APIError("Anthropic request failed after retries")

    # -- internal: error mapping ----------------------------------------

    @staticmethod
    def _map_status_error(exc: APIStatusError) -> APIError:
        """Translate an SDK status error into one of our domain errors."""
        status = exc.response.status_code if exc.response is not None else 0
        message = str(exc).strip() or f"HTTP {status}"
        if status in (401, 403):
            return AuthError(f"Authentication failed (HTTP {status}): {message}")
        if status == 429:
            retry_after = _parse_retry_after(
                exc.response.headers.get("retry-after") if exc.response else None
            )
            return RateLimitError(
                f"Rate limit exceeded (HTTP 429): {message}", retry_after=retry_after
            )
        if status == 408:
            return APIError(
                f"Request timeout (HTTP 408): {message}", category=ErrorCategory.TRANSIENT
            )
        if 400 <= status < 500:
            return APIError(
                f"Client error (HTTP {status}): {message}", category=ErrorCategory.VALIDATION
            )
        if 500 <= status < 600:
            return APIError(
                f"Server error (HTTP {status}): {message}", category=ErrorCategory.SERVER
            )
        return APIError(
            f"Unexpected status (HTTP {status}): {message}", category=ErrorCategory.UNKNOWN
        )

    def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        """Sleep for the server hint or exponential backoff + jitter."""
        if retry_after is not None and retry_after > 0:
            self._sleep(retry_after)
            return
        # attempt is 0-indexed; backoff grows as 1, 2, 4, 8 ... capped.
        base = min(self._initial_backoff * (2**attempt), self._max_backoff)
        jitter = random.uniform(0, base * 0.25)
        self._sleep(base + jitter)

    # -- internal: wire conversion --------------------------------------

    @staticmethod
    def _request_to_wire(request: APIRequest) -> dict[str, Any]:
        """Render an :class:`APIRequest` as the dict the SDK expects.

        :class:`ToolResultMessage` is converted to its underlying
        :class:`UserMessage` so the discriminated ``Message`` union
        doesn't have to be re-implemented here.
        """
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            messages.append(_message_to_wire(message))
        return {
            "model": request.model,
            "system": request.system,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "tools": [t.to_anthropic_tool() for t in request.tools],
        }

    @staticmethod
    def _parse_response(raw: Any) -> APIResponse:
        """Turn the SDK's message into our :class:`APIResponse`."""
        content: list[AssistantContent] = []
        for block in raw.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append(TextBlock(text=getattr(block, "text", "")))
            elif btype == "tool_use":
                content.append(
                    ToolUseBlock(
                        call=ToolCall(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            input=dict(getattr(block, "input", {}) or {}),
                        )
                    )
                )
            # Other block types (thinking, server_tool_use, etc.) are
            # ignored on purpose — they're not part of the agentic loop.

        stop_reason = _parse_stop_reason(getattr(raw, "stop_reason", None))
        return APIResponse(
            message=AssistantMessage(content=content, stop_reason=stop_reason),
        )


# -- module-level helpers (re-used by tests) ----------------------------


def _message_to_wire(message: UserMessage | AssistantMessage | ToolResultMessage) -> dict[str, Any]:
    """Render any message variant as the dict the SDK expects."""
    if isinstance(message, ToolResultMessage):
        return _user_message_to_wire(message.to_user_message())
    if isinstance(message, UserMessage):
        return _user_message_to_wire(message)
    return _assistant_message_to_wire(message)


def _user_message_to_wire(message: UserMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_use_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in message.content
        ],
    }


def _assistant_message_to_wire(message: AssistantMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": block.call.id,
                "name": block.call.name,
                "input": block.call.input,
            }
            if isinstance(block, ToolUseBlock)
            else {"type": "text", "text": block.text}
            for block in message.content
        ],
    }
    if message.stop_reason is not None:
        wire["stop_reason"] = message.stop_reason.value
    return wire


def _parse_stop_reason(raw: Any) -> StopReason | None:
    """Translate the SDK's ``stop_reason`` string to our :class:`StopReason` enum."""
    if raw is None:
        return None
    if raw == "tool_use":
        return StopReason.TOOL_USE
    if raw == "end_turn":
        return StopReason.END_TURN
    # Any other value (max_tokens, stop_sequence, refusal, pause_turn)
    # is treated as terminal but not "completed"; the loop still stops.
    return StopReason.END_TURN


def _parse_retry_after(header: str | None) -> float | None:
    """Parse the ``Retry-After`` header (seconds) into a float."""
    if not header:
        return None
    try:
        return float(header)
    except (TypeError, ValueError):
        return None
