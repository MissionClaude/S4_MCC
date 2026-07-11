"""MCP (Model Context Protocol) configuration loader.

This module parses ``.mcp.json`` and exposes a typed :class:`MCPConfig`
the rest of the system can query. It also resolves ``${ENV_VAR}``
placeholders against the process environment.

## Design note — config-only integration

The system is designed for the exam demo: ``.mcp.json`` is **parsed**
but MCP servers are **not** spawned at runtime. The loader produces a
typed config; :class:`~scenario_4_dev_productivity.mcp.discovery.MCPToolDiscovery`
turns it into tool adapters. Each adapter's ``execute`` returns a
structured failure explaining that real MCP transport is out of scope
for this build — the value is in the configuration surface, not the
network I/O.

This keeps the surface honest: callers see exactly which tools would
be available, with detailed descriptions, and they can swap in a real
transport later without changing the rest of the system.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Pattern that matches ${ENV_VAR} and ${ENV_VAR:-default} placeholders.
# The first capture group is the variable name; the second (optional)
# is the default value to use when the env var is not set or empty.
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class MCPToolSpec(BaseModel):
    """Static description of one tool an MCP server exposes.

    Used as a fallback when the server can't be reached at startup:
    the loader still produces a typed config and a tool manifest, even
    if ``npx @anthropic/mcp-github`` isn't installed locally.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Tool name (unique within a server)")
    description: str = Field(
        min_length=1,
        description=(
            "Natural-language description shown to the model. Should include "
            "an input example and boundary conditions, like every other tool."
        ),
    )
    inputSchema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the tool's input",
    )


class MCPServerConfig(BaseModel):
    """One server entry from ``.mcp.json``.

    The ``command``, ``args``, and ``env`` fields are the standard
    stdio-launch fields. ``tools`` is an optional static manifest; when
    present, the discovery layer uses it to build tool adapters without
    needing to actually launch the server.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="Server name (filled in by the loader)")
    command: str = Field(min_length=1, description="Executable to spawn")
    args: list[str] = Field(default_factory=list, description="Command-line arguments")
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the spawned process (env-expanded)",
    )
    tools: list[MCPToolSpec] = Field(
        default_factory=list,
        description="Optional static tool manifest — used when the server can't be reached",
    )

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_args(cls, value: Any) -> list[str]:
        """Accept both list-of-strings and string forms in the JSON."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class MCPConfig(BaseModel):
    """The top-level shape of an ``.mcp.json`` file."""

    model_config = ConfigDict(extra="forbid")

    mcpServers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Server name -> server config",
    )

    def server_names(self) -> tuple[str, ...]:
        """Names of all configured servers, in insertion order."""
        return tuple(self.mcpServers.keys())

    def tools(self) -> list[tuple[str, MCPToolSpec]]:
        """All (server_name, tool) pairs across every configured server."""
        out: list[tuple[str, MCPToolSpec]] = []
        for server_name, server in self.mcpServers.items():
            for tool in server.tools:
                out.append((server_name, tool))
        return out


class MCPConfigLoader:
    """Load and validate an ``.mcp.json`` file.

    The loader is stateless: every call to :meth:`load` reads the file
    fresh. The env-var resolution is parameterised so tests can inject
    a controlled environment.
    """

    def __init__(self, env: dict[str, str] | None = None) -> None:
        # ``os.environ`` is the default; tests pass a controlled dict.
        self._env: dict[str, str] = dict(env) if env is not None else dict(os.environ)

    def load(self, path: str | Path) -> MCPConfig:
        """Read ``path``, parse, env-expand, and return an :class:`MCPConfig`.

        :raises FileNotFoundError: when ``path`` does not exist.
        :raises ValueError: when the file is not valid JSON or fails validation.
        """
        text = self._read_text(path)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be a JSON object at the top level")

        # Env-expand the entire tree BEFORE Pydantic validation. This
        # way, "${GITHUB_TOKEN}" becomes a real value (or "" if unset),
        # and the schema check is on the resolved shape.
        expanded = self._expand_env(raw)
        try:
            return MCPConfig.model_validate(expanded)
        except ValidationError as exc:
            raise ValueError(f"{path} does not match the MCP config schema: {exc}") from exc

    def load_from_string(self, source: str) -> MCPConfig:
        """Parse an MCP config from a raw JSON string (test-friendly).

        Skips the file-read step; otherwise behaves like :meth:`load`.
        """
        try:
            raw = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"source is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("source must be a JSON object at the top level")
        expanded = self._expand_env(raw)
        try:
            return MCPConfig.model_validate(expanded)
        except ValidationError as exc:
            raise ValueError(f"source does not match the MCP config schema: {exc}") from exc

    # -- env expansion ---------------------------------------------------

    def expand_env(self, value: str) -> str:
        """Resolve ``${ENV_VAR}`` and ``${ENV_VAR:-default}`` in a string.

        Public so callers (and tests) can use the same resolver on
        arbitrary strings.
        """
        return _ENV_VAR_PATTERN.sub(self._substitute, value)

    def _substitute(self, match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        value = self._env.get(var_name)
        if value:
            return value
        if default is not None:
            return default
        # Unset with no default — return the literal placeholder so
        # the error message is debuggable. The downstream Pydantic
        # schema will reject empty strings for required fields.
        return match.group(0)

    def _expand_env(self, value: Any) -> Any:
        """Recursively env-expand every string in a JSON-like tree."""
        if isinstance(value, str):
            return self.expand_env(value)
        if isinstance(value, dict):
            return {k: self._expand_env(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand_env(v) for v in value]
        return value

    # -- I/O --------------------------------------------------------------

    @staticmethod
    def _read_text(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MCP config file not found: {path}")
        return path.read_text(encoding="utf-8")


__all__ = [
    "MCPConfig",
    "MCPConfigLoader",
    "MCPServerConfig",
    "MCPToolSpec",
]
