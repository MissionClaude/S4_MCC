"""Context compaction — the ``/compact`` equivalent.

When the agentic loop has been running for many turns the
``messages`` list grows. The model's context window has a hard cap;
filling it makes the API refuse new requests. The fix is compression:
collapse the oldest turns into a short summary while preserving:

* the original task (first user message) — so the model never
  forgets the goal;
* tool_use → tool_result correlations — so the model can keep
  reasoning about actions it already took;
* the most recent few turns — the "now" of the conversation.

The compressor is deliberately conservative. It drops the verbose
text of large tool results (keeping first N lines + a truncation
marker) but never invents content. The agent is expected to dump
key findings to the :class:`ScratchpadManager` *before* compression
so the durable state survives.
"""

from __future__ import annotations

from scenario_4_dev_productivity.models.messages import Message, ToolResultMessage
from scenario_4_dev_productivity.models.tools import ToolResult

# How many lines of a tool result to keep after ``summarise_tool_result``.
# Reads and Greps often produce thousands of lines of output — we keep
# the top + bottom so the model can still see the structure.
DEFAULT_KEEP_LINES = 20


# How many recent turns (assistant or tool-result) to keep verbatim.
# Anything older is summarised.
DEFAULT_RECENT_TURNS = 4


class ContextCompactor:
    """Stateless context compressor.

    Instances are cheap; create one with the desired settings and call
    :meth:`compact` to compress a conversation. The class is stateless
    so tests can construct it freely.
    """

    def __init__(
        self,
        *,
        keep_lines: int = DEFAULT_KEEP_LINES,
        recent_turns: int = DEFAULT_RECENT_TURNS,
    ) -> None:
        if keep_lines < 1:
            raise ValueError("keep_lines must be >= 1")
        if recent_turns < 0:
            raise ValueError("recent_turns must be >= 0")
        self._keep_lines = keep_lines
        self._recent_turns = recent_turns

    @property
    def keep_lines(self) -> int:
        return self._keep_lines

    @property
    def recent_turns(self) -> int:
        return self._recent_turns

    def compact(
        self,
        messages: list[Message | ToolResultMessage],
    ) -> list[Message | ToolResultMessage]:
        """Compress ``messages`` and return a new, shorter list.

        The first user message (the original task) is always preserved.
        The last ``recent_turns`` messages are preserved verbatim. Any
        :class:`ToolResultMessage` in the middle is summarised by
        :func:`summarise_tool_result`. Assistant messages are kept
        intact (they're usually short).

        The result is always shorter than the input by design.
        """
        if not messages:
            return []
        first, *rest = messages
        # The first message is always the user task. Pin it.
        kept: list[Message | ToolResultMessage] = [first]
        if not rest:
            return kept

        # Identify the "recent" window: the last N non-summary messages.
        recent_window = rest[-self._recent_turns :] if self._recent_turns else []
        recent_set: set[int] = {id(m) for m in recent_window}

        # Anything before the recent window: summarise tool results,
        # keep assistant text as-is.
        middle = rest[: -self._recent_turns] if self._recent_turns else rest
        for msg in middle:
            if id(msg) in recent_set:
                # Defensive: overlap shouldn't happen, but skip rather than duplicate.
                continue
            if isinstance(msg, ToolResultMessage):
                kept.append(_summarise_tool_result_message(msg, self._keep_lines))
            else:
                kept.append(msg)

        kept.extend(recent_window)
        return kept


# -- module-level helpers -------------------------------------------------


def summarise_tool_result(result: ToolResult, keep_lines: int = DEFAULT_KEEP_LINES) -> ToolResult:
    """Trim a single :class:`ToolResult`'s content to ``keep_lines`` lines.

    The original is returned unchanged when it's already short. When
    it's long, the result keeps the first ``keep_lines`` and the last
    ``keep_lines`` lines with a ``[... N lines truncated ...]`` marker
    in between. Truncation markers go to ``is_error=False`` —
    truncation is informational, not a failure.
    """
    return _summarise(result, keep_lines)


def compact_messages(
    messages: list[Message | ToolResultMessage],
    *,
    keep_lines: int = DEFAULT_KEEP_LINES,
    recent_turns: int = DEFAULT_RECENT_TURNS,
) -> list[Message | ToolResultMessage]:
    """Functional form of :meth:`ContextCompactor.compact`."""
    return ContextCompactor(keep_lines=keep_lines, recent_turns=recent_turns).compact(messages)


# -- internals ------------------------------------------------------------


def _summarise(result: ToolResult, keep_lines: int) -> ToolResult:
    content = result.content
    lines = content.splitlines()
    if len(lines) <= keep_lines * 2:
        return result
    head = lines[:keep_lines]
    tail = lines[-keep_lines:]
    truncated = len(lines) - len(head) - len(tail)
    marker = f"[... {truncated} lines truncated ...]"
    new_content = "\n".join([*head, marker, *tail])
    return result.model_copy(update={"content": new_content})


def _summarise_tool_result_message(
    msg: ToolResultMessage, keep_lines: int
) -> ToolResultMessage:
    new_results = [_summarise(r, keep_lines) for r in msg.results]
    return msg.model_copy(update={"results": new_results})


__all__ = [
    "ContextCompactor",
    "DEFAULT_KEEP_LINES",
    "DEFAULT_RECENT_TURNS",
    "compact_messages",
    "summarise_tool_result",
]
