"""PostToolUse hooks — mutate :class:`ToolResult` after the loop runs a tool.

The spec calls out a "PostToolUse hook to trim verbose tool results
before they enter the conversation context". This module provides:

* :class:`PostToolUseHook` — a small protocol every hook implements.
* :class:`TrimReadOutputHook` — when a Read result is longer than a
  threshold, keep the first + last N lines and mark the middle as
  truncated.
* :class:`LoggingHook` — records every (tool, result) to a list for
  debugging. Useful in tests that want to assert on tool execution
  order without poking the registry.
* :func:`run_hooks` — apply a chain of hooks to one result. Hooks
  run in order; each receives the result of the previous.

Hooks are intentionally simple. They take a tool name and a result
and return a new result. They never raise — the loop must keep
running even when a hook is buggy.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from scenario_4_dev_productivity.models.tools import ToolResult

logger = logging.getLogger(__name__)


@runtime_checkable
class PostToolUseHook(Protocol):
    """A hook that runs after the agentic loop executes a tool.

    Hooks receive the tool name, the call id, and the result. They
    return a (possibly modified) :class:`ToolResult`. Hooks MUST NOT
    raise — see :func:`run_hooks` for the safety wrapper.
    """

    def __call__(
        self, tool_name: str, tool_use_id: str, result: ToolResult
    ) -> ToolResult:
        ...


class TrimReadOutputHook:
    """Trim very long Read results to keep context healthy.

    Triggers when the Read tool's output exceeds ``max_lines``. Keeps
    the first ``keep_lines`` and the last ``keep_lines`` lines, with
    a ``[... N lines truncated ...]`` marker in between.

    The hook only fires for the ``Read`` tool. Other tools' outputs
    are passed through untouched.
    """

    def __init__(self, *, max_lines: int = 200, keep_lines: int = 50) -> None:
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")
        if keep_lines < 1 or keep_lines * 2 > max_lines:
            raise ValueError("keep_lines must be >= 1 and <= max_lines / 2")
        self._max_lines = max_lines
        self._keep_lines = keep_lines

    @property
    def max_lines(self) -> int:
        return self._max_lines

    @property
    def keep_lines(self) -> int:
        return self._keep_lines

    def __call__(
        self, tool_name: str, tool_use_id: str, result: ToolResult
    ) -> ToolResult:
        if tool_name != "Read":
            return result
        if result.is_error:
            return result
        lines = result.content.splitlines()
        if len(lines) <= self._max_lines:
            return result
        head = lines[: self._keep_lines]
        tail = lines[-self._keep_lines :]
        truncated = len(lines) - len(head) - len(tail)
        marker = f"[... {truncated} lines truncated ...]"
        new_content = "\n".join([*head, marker, *tail])
        return result.model_copy(update={"content": new_content})


class LoggingHook:
    """Record every tool execution for debugging.

    Stores a list of ``(tool_name, tool_use_id, is_error)`` tuples in
    insertion order. The list is the hook's only state.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def __call__(
        self, tool_name: str, tool_use_id: str, result: ToolResult
    ) -> ToolResult:
        self.records.append(
            {
                "tool": tool_name,
                "tool_use_id": tool_use_id,
                "is_error": result.is_error,
                "category": result.error_category,
            }
        )
        return result

    def tool_names(self) -> list[str]:
        """Just the tool names in call order — convenient for tests."""
        return [r["tool"] for r in self.records]


def run_hooks(
    hooks: list[PostToolUseHook],
    tool_name: str,
    tool_use_id: str,
    result: ToolResult,
) -> ToolResult:
    """Apply a chain of hooks to a single result.

    Each hook receives the previous hook's output. Hooks that raise
    are caught and logged at WARNING — the result is the last
    successful hook's output (or the original on total failure). This
    is the same safety contract the registry gives tools.
    """
    current = result
    for hook in hooks:
        try:
            current = hook(tool_name, tool_use_id, current)
        except Exception as exc:  # noqa: BLE001 — hook bugs must not break the loop
            logger.warning("PostToolUse hook %r failed: %s", hook, exc)
    return current


__all__ = [
    "LoggingHook",
    "PostToolUseHook",
    "TrimReadOutputHook",
    "run_hooks",
]
