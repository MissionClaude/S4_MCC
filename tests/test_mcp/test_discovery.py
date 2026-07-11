"""Unit tests for :class:`MCPToolDiscovery` and :class:`MCPToolAdapter`.

The discovery layer turns an :class:`MCPConfig` into a list of tool
adapters. Each adapter:

* exposes a :class:`ToolDefinition` with a namespaced name and
  detailed description;
* has an ``execute`` that returns a structured failure (since this
  build is config-only — the MCP transport is not connected).

Tests cover all of this, plus the registration helper that puts
adapters into a :class:`ToolRegistry`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.mcp.discovery import MCPToolAdapter, MCPToolDiscovery
from scenario_4_dev_productivity.mcp.loader import (
    MCPConfig,
    MCPConfigLoader,
    MCPServerConfig,
    MCPToolSpec,
)
from scenario_4_dev_productivity.models.tools import ToolDefinition
from scenario_4_dev_productivity.tools.registry import ToolRegistry

# -- fixtures --------------------------------------------------------------


@pytest.fixture
def config() -> MCPConfig:
    return MCPConfigLoader(env={"X": "1"}).load_from_string(
        """
        {
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-github"],
                    "env": {"GITHUB_TOKEN": "${X}"},
                    "tools": [
                        {
                            "name": "create_issue",
                            "description": "Open a new GitHub issue",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "owner": {"type": "string"},
                                    "repo": {"type": "string"},
                                    "title": {"type": "string"}
                                },
                                "required": ["owner", "repo", "title"]
                            }
                        },
                        {
                            "name": "list_pull_requests",
                            "description": "List PRs on a repository",
                            "inputSchema": {"type": "object"}
                        }
                    ]
                },
                "slack": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-slack"],
                    "tools": [
                        {
                            "name": "send_message",
                            "description": "Post a message to Slack"
                        }
                    ]
                }
            }
        }
        """
    )


# -- MCPToolAdapter -------------------------------------------------------


class TestMCPToolAdapter:
    def _adapter(self) -> MCPToolAdapter:
        spec = MCPToolSpec(
            name="create_issue",
            description="Open a new GitHub issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                },
                "required": ["owner", "repo"],
            },
        )
        return MCPToolAdapter(server_name="github", spec=spec)

    def test_name_is_namespaced(self) -> None:
        adapter = self._adapter()
        # Namespacing prevents collisions across servers.
        assert adapter.name == "github__create_issue"

    def test_server_name_property(self) -> None:
        adapter = self._adapter()
        assert adapter.server_name == "github"

    def test_spec_property(self) -> None:
        adapter = self._adapter()
        assert adapter.spec.name == "create_issue"

    def test_definition_is_a_tool_definition(self) -> None:
        adapter = self._adapter()
        defn = adapter.definition
        assert isinstance(defn, ToolDefinition)
        assert defn.name == "github__create_issue"

    def test_definition_description_mentions_server(self) -> None:
        """The spec requires descriptions detailed enough to prevent
        preference for built-in equivalents — they MUST surface the
        server name so the model knows it's an external service."""
        adapter = self._adapter()
        assert "github" in adapter.definition.description

    def test_definition_includes_tool_description(self) -> None:
        adapter = self._adapter()
        assert "Open a new GitHub issue" in adapter.definition.description

    def test_parameters_round_trip_from_input_schema(self) -> None:
        adapter = self._adapter()
        params = adapter.definition.parameters
        assert "owner" in params.properties
        assert "repo" in params.properties
        assert set(params.required) == {"owner", "repo"}

    def test_execute_returns_structured_failure(self) -> None:
        """The adapter is config-only — execute returns a structured error
        instead of actually calling the MCP server."""
        adapter = self._adapter()
        result = adapter.execute("call-1", {"owner": "x", "repo": "y"})
        assert result.is_error is True
        # Config-only failure is transient (we *could* connect, just don't).
        assert result.error_category == "transient"
        assert result.is_retryable is False
        # The message is specific, not generic — mentions the server + tool.
        assert "create_issue" in result.content
        assert "github" in result.content

    def test_handles_missing_input_schema(self) -> None:
        """A spec without ``inputSchema`` should not crash discovery."""
        spec = MCPToolSpec(name="ping", description="Pings the server")
        adapter = MCPToolAdapter(server_name="x", spec=spec)
        params = adapter.definition.parameters
        # Defaults: empty properties, empty required.
        assert params.properties == {}
        assert params.required == []

    def test_handles_malformed_input_schema(self) -> None:
        """A spec with a non-dict inputSchema should fall back to defaults."""
        spec = MCPToolSpec(name="ping", description="Pings the server")
        # Mutate after construction since the type says dict.
        spec.inputSchema = "not a dict"  # type: ignore[assignment]
        adapter = MCPToolAdapter(server_name="x", spec=spec)
        # Should not raise; defaults applied.
        assert adapter.definition.parameters.properties == {}

    def test_required_filters_non_strings(self) -> None:
        """A spec with a non-string entry in ``required`` should drop it
        rather than raise — the model still gets a usable tool."""
        spec = MCPToolSpec(
            name="t",
            description="d",
            inputSchema={"type": "object", "required": ["good", 42, "also-good"]},
        )
        adapter = MCPToolAdapter(server_name="x", spec=spec)
        assert adapter.definition.parameters.required == ["good", "also-good"]


# -- MCPToolDiscovery -----------------------------------------------------


class TestMCPToolDiscovery:
    def test_discover_all_returns_one_adapter_per_tool(
        self, config: MCPConfig
    ) -> None:
        disc = MCPToolDiscovery(config)
        adapters = disc.discover_all()
        # 2 from github + 1 from slack = 3
        assert len(adapters) == 3
        names = {a.name for a in adapters}
        assert names == {
            "github__create_issue",
            "github__list_pull_requests",
            "slack__send_message",
        }

    def test_discover_for_returns_only_one_servers_tools(
        self, config: MCPConfig
    ) -> None:
        disc = MCPToolDiscovery(config)
        github_adapters = disc.discover_for("github")
        assert len(github_adapters) == 2
        assert all(a.server_name == "github" for a in github_adapters)

    def test_discover_for_unknown_server_raises(self, config: MCPConfig) -> None:
        disc = MCPToolDiscovery(config)
        with pytest.raises(ValueError, match="not configured"):
            disc.discover_for("nonexistent")

    def test_servers_with_no_tools_yield_no_adapters(self) -> None:
        cfg = MCPConfig(
            mcpServers={"empty": MCPServerConfig(command="x", tools=[])}
        )
        disc = MCPToolDiscovery(cfg)
        assert disc.discover_all() == []

    def test_register_into_tool_registry(self, config: MCPConfig) -> None:
        """``register_into`` puts every adapter into the registry and returns the count."""
        registry = ToolRegistry()
        disc = MCPToolDiscovery(config)
        count = disc.register_into(registry)
        assert count == 3
        # All three are registered under their namespaced names.
        for name in (
            "github__create_issue",
            "github__list_pull_requests",
            "slack__send_message",
        ):
            assert name in registry

    def test_registered_adapter_executes_returns_structured_error(
        self, config: MCPConfig
    ) -> None:
        """End-to-end: an adapter pulled from the registry executes and
        returns the same structured failure the standalone adapter does."""
        registry = ToolRegistry()
        disc = MCPToolDiscovery(config)
        disc.register_into(registry)
        adapter = registry.get("github__create_issue")
        assert adapter is not None
        result = registry.execute("call-x", "github__create_issue", {"owner": "x"})
        assert result.is_error is True
        assert result.error_category == "transient"
        assert result.is_retryable is False

    def test_config_property(self, config: MCPConfig) -> None:
        disc = MCPToolDiscovery(config)
        assert disc.config is config

    def test_register_into_accepts_any_object_with_register(
        self, config: MCPConfig
    ) -> None:
        """``register_into`` is duck-typed — verify it works with a mock."""
        mock_registry = MagicMock()
        disc = MCPToolDiscovery(config)
        disc.register_into(mock_registry)
        assert mock_registry.register.call_count == 3  # type: ignore[attr-defined]

    def test_namespacing_prevents_collision(self) -> None:
        """Two servers can expose a tool with the same local name without colliding."""
        cfg = MCPConfig(
            mcpServers={
                "a": MCPServerConfig(
                    command="x",
                    tools=[MCPToolSpec(name="ping", description="A's ping")],
                ),
                "b": MCPServerConfig(
                    command="y",
                    tools=[MCPToolSpec(name="ping", description="B's ping")],
                ),
            }
        )
        disc = MCPToolDiscovery(cfg)
        registry = ToolRegistry()
        disc.register_into(registry)
        assert "a__ping" in registry
        assert "b__ping" in registry
        # The two descriptions differ — the model can tell them apart.
        a_desc = registry.get("a__ping").definition.description  # type: ignore[union-attr]
        b_desc = registry.get("b__ping").definition.description  # type: ignore[union-attr]
        assert a_desc != b_desc

    def test_execute_with_any_arguments_shape(self) -> None:
        """Adapter's execute accepts whatever the model sends; missing
        fields are not validated here (the MCP transport would handle
        that). Our config-only adapter doesn't care about arguments."""
        spec = MCPToolSpec(name="t", description="d")
        adapter = MCPToolAdapter(server_name="x", spec=spec)
        result = adapter.execute("call-1", {})
        assert result.is_error is True
        # Even with bogus arguments, the response is a structured failure,
        # not an exception.
        assert result.error_category in {"transient", "validation"}
