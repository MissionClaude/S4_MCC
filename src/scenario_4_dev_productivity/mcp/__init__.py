"""MCP (Model Context Protocol) integration.

This package provides:

* :class:`~scenario_4_dev_productivity.mcp.loader.MCPConfigLoader` —
  parses ``.mcp.json`` files and resolves ``${ENV_VAR}`` placeholders
  against the process environment.
* :class:`~scenario_4_dev_productivity.mcp.discovery.MCPToolDiscovery` —
  turns a loaded :class:`~scenario_4_dev_productivity.mcp.loader.MCPConfig`
  into a list of
  :class:`~scenario_4_dev_productivity.mcp.discovery.MCPToolAdapter`
  instances that can be registered with the agentic loop's
  :class:`ToolRegistry`.

The integration is **config-only** in this build: ``.mcp.json`` is
parsed and tools are advertised to the model with detailed
descriptions, but no MCP server is actually spawned at runtime. Each
adapter's ``execute`` returns a structured failure explaining the
config-only design — the value here is the configuration surface and
the structured error contract, not the network I/O.
"""

from __future__ import annotations

from scenario_4_dev_productivity.mcp.discovery import MCPToolAdapter, MCPToolDiscovery
from scenario_4_dev_productivity.mcp.loader import (
    MCPConfig,
    MCPConfigLoader,
    MCPServerConfig,
    MCPToolSpec,
)

__all__ = [
    "MCPConfig",
    "MCPConfigLoader",
    "MCPServerConfig",
    "MCPToolAdapter",
    "MCPToolDiscovery",
    "MCPToolSpec",
]
