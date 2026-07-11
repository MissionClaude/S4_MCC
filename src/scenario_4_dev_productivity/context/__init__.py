"""Context management for long-running agentic sessions.

Three primitives, all stateless across instances but stateful through
the file system or an injected hook list:

* :class:`ScratchpadManager` — read/write a scratchpad file that
  survives context compression. The agent can dump structured findings
  to it before /compact and re-read them afterwards.
* :class:`ContextCompactor` — summarise a long conversation into a
  compact form, preserving tool_use → result correlations and dropping
  verbose tool outputs.
* :class:`PostToolUseHook` — a small protocol implemented by hooks
  that mutate :class:`ToolResult` after the loop executes a tool.
  Bundled hooks trim large Read outputs and log tool execution.

These three together implement the spec's "context compression via
/compact" and "PostToolUse hooks for result trimming" requirements.
"""

from __future__ import annotations

from scenario_4_dev_productivity.context.compact import (
    ContextCompactor,
    compact_messages,
    summarise_tool_result,
)
from scenario_4_dev_productivity.context.hooks import (
    LoggingHook,
    PostToolUseHook,
    TrimReadOutputHook,
    run_hooks,
)
from scenario_4_dev_productivity.context.scratchpad import ScratchpadEntry, ScratchpadManager

__all__ = [
    "ContextCompactor",
    "LoggingHook",
    "PostToolUseHook",
    "ScratchpadEntry",
    "ScratchpadManager",
    "TrimReadOutputHook",
    "compact_messages",
    "run_hooks",
    "summarise_tool_result",
]
