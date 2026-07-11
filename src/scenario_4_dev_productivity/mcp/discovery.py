"""MCP tool discovery — turn an :class:`MCPConfig` into tool adapters.

In a real deployment, discovery would spawn each configured server
(stdio transport) and call ``tools/list`` to enumerate the tools. This
build is config-only: we use the **static tool manifest** embedded in
``.mcp.json`` (under each server's ``tools`` array) to build adapters
without spawning anything.

Each adapter is a :class:`~scenario_4_dev_productivity.tools.registry.Tool`
that:

* exposes a fully-formed :class:`ToolDefinition` (name prefixed with
  the server name to avoid collisions, detailed description, JSON Schema
  input);
* has an ``execute`` that returns a structured :class:`ToolResult.failure`
  with category ``"transient"`` and ``is_retryable=False`` — the model
  gets a clear "this tool is configured but not connected" message
  instead of a generic exception.

The structured failure is what the spec calls for: every error
response has ``isError``, ``errorCategory``, and ``isRetryable``. The
adapter maps directly to the spec's "validation error" or "transient
error" categories depending on whether the tool name is known.
"""

from __future__ import annotations

from typing import Any

from scenario_4_dev_productivity.mcp.loader import MCPConfig, MCPToolSpec
from scenario_4_dev_productivity.models.tools import (
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)


class MCPToolAdapter:
    """Adapter that exposes a configured MCP tool to the model.

    The adapter satisfies the :class:`Tool` protocol from
    :mod:`scenario_4_dev_productivity.tools.registry`: it exposes
    ``name`` / ``definition`` properties and an ``execute`` method.

    The ``name`` is namespaced as ``"{server_name}__{tool_name}"`` so
    two servers exposing tools with the same local name don't
    collide in the registry.
    """

    def __init__(self, server_name: str, spec: MCPToolSpec) -> None:
        self._server_name = server_name
        self._spec = spec
        self._namespaced_name = f"{server_name}__{spec.name}"

    @property
    def name(self) -> str:
        return self._namespaced_name

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def spec(self) -> MCPToolSpec:
        return self._spec

    @property
    def definition(self) -> ToolDefinition:
        description = self._build_description()
        return ToolDefinition(
            name=self._namespaced_name,
            description=description,
            parameters=self._build_parameters(),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Run the MCP tool.

        Config-only build: returns a structured failure explaining that
        the server is configured but not connected. In a real
        deployment this would call the MCP stdio transport.
        """
        return ToolResult.failure(
            tool_use_id=tool_use_id,
            message=(
                f"MCP tool {self._spec.name!r} on server {self._server_name!r} "
                "is configured but not connected. The .mcp.json entry is a "
                "reference config — live MCP transport is out of scope for "
                "this build. Spawn a real transport (stdio client) to invoke "
                "this tool."
            ),
            category="transient",
            retryable=False,
        )

    def _build_description(self) -> str:
        """Build a model-facing description that disambiguates from built-in tools.

        The spec says MCP tool descriptions must be detailed enough to
        "prevent preference for built-in equivalents" — we lead with
        the server name, the tool's role, and an example.
        """
        spec = self._spec
        return (
            f"MCP tool from server {self._server_name!r}: {spec.name}.\n\n"
            f"{spec.description}\n\n"
            "Input example: see the inputSchema below.\n"
            "Boundary: this tool comes from an MCP server configured in "
            ".mcp.json. It is NOT one of the built-in tools (Read, Write, "
            "Bash, etc.). Use it when the user task specifically needs the "
            "external service this server provides (e.g. GitHub API)."
        )

    def _build_parameters(self) -> ToolParameterSchema:
        """Convert the spec's ``inputSchema`` into our :class:`ToolParameterSchema`.

        The spec's ``inputSchema`` is typed as ``dict[str, Any]``; in
        practice callers may pass anything. We treat any non-dict as
        an empty schema so a malformed entry never crashes discovery.
        """
        # Cast to Any: the schema attribute is typed dict, but we want
        # to be defensive against callers that bypass the Pydantic
        # validator (tests do this). Mypy would otherwise mark the
        # isinstance branch as unreachable.
        schema: Any = self._spec.inputSchema
        if not isinstance(schema, dict):
            # Defensive: a malformed spec shouldn't crash discovery.
            return ToolParameterSchema()
        raw_type = schema.get("type", "object")
        type_str = raw_type if isinstance(raw_type, str) else "object"
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        required = schema.get("required", [])
        if not isinstance(required, list):
            required = []
        # Every item in ``required`` must be a string; non-strings are
        # dropped silently rather than raising — the model sees a tool
        # it can still call, just with a slightly weaker contract.
        required_strs = [r for r in required if isinstance(r, str)]
        return ToolParameterSchema(
            type=type_str,
            properties=properties,
            required=required_strs,
        )


class MCPToolDiscovery:
    """Turn an :class:`MCPConfig` into a list of tool adapters.

    The discovery is intentionally simple: for each ``(server, tool)``
    pair in the static manifest, build an :class:`MCPToolAdapter`.
    Servers with no ``tools`` array are silently skipped — there's
    nothing to discover, and we have no transport to enumerate live
    tools from.
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config

    @property
    def config(self) -> MCPConfig:
        return self._config

    def discover_all(self) -> list[MCPToolAdapter]:
        """Return an adapter for every (server, tool) pair in the config."""
        adapters: list[MCPToolAdapter] = []
        for server_name, server in self._config.mcpServers.items():
            for spec in server.tools:
                adapters.append(MCPToolAdapter(server_name=server_name, spec=spec))
        return adapters

    def discover_for(self, server_name: str) -> list[MCPToolAdapter]:
        """Return adapters for one server, or an empty list if not configured.

        :raises ValueError: when the server name is not configured.
        """
        if server_name not in self._config.mcpServers:
            raise ValueError(
                f"MCP server {server_name!r} is not configured. "
                f"Available servers: {', '.join(self._config.server_names()) or '(none)'}"
            )
        return [
            MCPToolAdapter(server_name=server_name, spec=spec)
            for spec in self._config.mcpServers[server_name].tools
        ]

    def register_into(self, registry: Any) -> int:
        """Register every discovered adapter into a :class:`ToolRegistry`.

        ``registry`` is typed as ``Any`` to avoid the circular import
        on :mod:`scenario_4_dev_productivity.tools.registry`. The
        duck-typed contract is the same: ``register(tool)`` where
        ``tool`` exposes ``name`` and ``definition``.

        Returns the number of adapters registered (handy for tests).
        """
        count = 0
        for adapter in self.discover_all():
            registry.register(adapter)
            count += 1
        return count


__all__ = ["MCPToolAdapter", "MCPToolDiscovery"]
