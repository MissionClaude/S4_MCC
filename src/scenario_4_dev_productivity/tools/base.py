"""Base interface and built-in tool registry for the developer productivity agent.

A :class:`Tool` is the executable half of a :class:`ToolDefinition` — the
declaration goes to the model, the implementation is what the agentic loop
calls when the model asks for a tool. Keeping these in one place makes the
"which tools exist" question answerable from a single list.

The :data:`BUILTIN_TOOLS` tuple is the canonical set. ``default_registry()``
returns a fresh :class:`ToolRegistry` pre-populated with these tools. Both
are re-exported from :mod:`scenario_4_dev_productivity.tools` so the agentic
loop can ask "what's available?" without knowing about any specific tool.
"""

from __future__ import annotations

from scenario_4_dev_productivity.tools.bash_tool import BashTool
from scenario_4_dev_productivity.tools.edit_tool import EditTool
from scenario_4_dev_productivity.tools.glob_tool import GlobTool
from scenario_4_dev_productivity.tools.grep_tool import GrepTool
from scenario_4_dev_productivity.tools.read_tool import ReadTool
from scenario_4_dev_productivity.tools.registry import Tool, ToolRegistry
from scenario_4_dev_productivity.tools.write_tool import WriteTool

# Canonical set — what the agentic loop uses by default.
# The order is read-only first, write-second, then side-effecting, so test
# assertions that depend on tool insertion order are deterministic.
BUILTIN_TOOLS: tuple[Tool, ...] = (
    ReadTool(),
    GrepTool(),
    GlobTool(),
    WriteTool(),
    EditTool(),
    BashTool(),
)


def default_registry() -> ToolRegistry:
    """Return a :class:`ToolRegistry` pre-populated with the built-in tools.

    Each call returns a *new* registry so tests can mutate state without
    bleeding across cases. Production code should call this once at
    startup and pass the result to the agentic loop.
    """
    registry = ToolRegistry()
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    return registry
