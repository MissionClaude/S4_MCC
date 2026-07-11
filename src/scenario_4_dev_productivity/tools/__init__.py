"""Built-in tools the agentic loop can dispatch to.

This package wires the abstract :class:`~scenario_4_dev_productivity.models.ToolDefinition`
to actual Python callables that the :class:`ToolRegistry` invokes. Each tool:

* exposes a static :class:`ToolDefinition` (name, description, parameter schema)
  that the model sees when deciding which tool to call;
* has a single ``execute(**kwargs) -> ToolResult`` method that does the work
  and packages the outcome — success or structured failure — for the loop.

The descriptions are deliberately long and rich: the model uses the
description alone to pick the right tool, so we include an input example,
the kind of output the tool produces, and the boundary conditions
(when NOT to use the tool). That matches the exam guide's guidance on
"tool descriptions must be detailed".

Boundary conventions every tool follows:

* Read-only tools (Read, Grep, Glob) return ``ToolResult.success(...)`` and
  surface "not found" as a structured failure with category
  ``"validation"`` (the caller can fix the input).
* Write tools (Write, Edit) treat permission errors and parent-missing
  errors as ``"permission"`` and validation problems as ``"validation"``.
* Bash wraps ``subprocess.run`` and treats non-zero exit as a structured
  failure; the timeout is a hard wall and is reported as ``"transient"``
  with ``is_retryable=True`` when the operation looks idempotent.
"""

from __future__ import annotations

from scenario_4_dev_productivity.tools.base import (
    BUILTIN_TOOLS,
    default_registry,
)
from scenario_4_dev_productivity.tools.bash_tool import BashTool
from scenario_4_dev_productivity.tools.edit_tool import EditTool
from scenario_4_dev_productivity.tools.glob_tool import GlobTool
from scenario_4_dev_productivity.tools.grep_tool import GrepTool
from scenario_4_dev_productivity.tools.read_tool import ReadTool
from scenario_4_dev_productivity.tools.registry import Tool, ToolRegistry
from scenario_4_dev_productivity.tools.write_tool import WriteTool

__all__ = [
    "BUILTIN_TOOLS",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "Tool",
    "ToolRegistry",
    "WriteTool",
    "default_registry",
]
