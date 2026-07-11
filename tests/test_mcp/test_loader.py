"""Unit tests for :class:`MCPConfigLoader`.

The loader has three jobs:

1. Read ``.mcp.json`` from disk (or a raw string in tests).
2. Validate the shape via Pydantic.
3. Resolve ``${ENV_VAR}`` placeholders against the process environment.

We test each job in isolation, with a controlled env so the tests are
deterministic regardless of what the developer's machine has set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenario_4_dev_productivity.mcp.loader import (
    MCPConfig,
    MCPConfigLoader,
    MCPServerConfig,
    MCPToolSpec,
)

# -- fixtures --------------------------------------------------------------


@pytest.fixture
def env() -> dict[str, str]:
    return {
        "GITHUB_TOKEN": "ghp_test_123",
        "API_KEY": "sk-abc",
        "EMPTY_VAR": "",
    }


@pytest.fixture
def loader(env: dict[str, str]) -> MCPConfigLoader:
    return MCPConfigLoader(env=env)


@pytest.fixture
def sample_config() -> str:
    return json.dumps(
        {
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-github"],
                    "env": {
                        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
                        "STATIC": "literal-value",
                    },
                    "tools": [
                        {
                            "name": "create_issue",
                            "description": "Create a GitHub issue",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "owner": {"type": "string"},
                                    "repo": {"type": "string"},
                                },
                                "required": ["owner", "repo"],
                            },
                        }
                    ],
                }
            }
        }
    )


# -- env-expansion unit tests --------------------------------------------


class TestEnvExpansion:
    def test_simple_var_is_substituted(self, loader: MCPConfigLoader) -> None:
        assert loader.expand_env("token=${GITHUB_TOKEN}") == "token=ghp_test_123"

    def test_unset_var_uses_default(self, loader: MCPConfigLoader) -> None:
        assert loader.expand_env("v=${MISSING:-fallback}") == "v=fallback"

    def test_empty_var_uses_default(self, loader: MCPConfigLoader) -> None:
        # Empty env var should still trigger the default.
        assert loader.expand_env("v=${EMPTY_VAR:-fallback}") == "v=fallback"

    def test_unset_var_without_default_returns_placeholder(
        self, loader: MCPConfigLoader
    ) -> None:
        # Leaving the placeholder makes the error debuggable instead
        # of silently turning into an empty string.
        assert loader.expand_env("v=${MISSING}") == "v=${MISSING}"

    def test_no_placeholders_passthrough(self, loader: MCPConfigLoader) -> None:
        assert loader.expand_env("just a string") == "just a string"

    def test_multiple_placeholders_in_one_string(
        self, loader: MCPConfigLoader
    ) -> None:
        assert (
            loader.expand_env("a=${GITHUB_TOKEN}&b=${API_KEY}")
            == "a=ghp_test_123&b=sk-abc"
        )

    def test_recursive_expansion_in_nested_dict(
        self, loader: MCPConfigLoader
    ) -> None:
        result = loader._expand_env(  # noqa: SLF001 — internal, but testable
            {"outer": {"inner": "${GITHUB_TOKEN}"}, "list": ["${API_KEY}"]}
        )
        assert result == {
            "outer": {"inner": "ghp_test_123"},
            "list": ["sk-abc"],
        }


# -- load_from_string (test-friendly entry point) -------------------------


class TestLoadFromString:
    def test_loads_minimal_config(
        self, loader: MCPConfigLoader, sample_config: str
    ) -> None:
        config = loader.load_from_string(sample_config)
        assert isinstance(config, MCPConfig)
        assert config.server_names() == ("github",)

    def test_env_vars_resolved_in_env_block(
        self, loader: MCPConfigLoader, sample_config: str
    ) -> None:
        config = loader.load_from_string(sample_config)
        server = config.mcpServers["github"]
        # The loader resolves placeholders BEFORE Pydantic validates,
        # so the resulting config has real values, not placeholder strings.
        assert server.env["GITHUB_TOKEN"] == "ghp_test_123"
        assert server.env["STATIC"] == "literal-value"

    def test_tools_are_parsed(self, loader: MCPConfigLoader, sample_config: str) -> None:
        config = loader.load_from_string(sample_config)
        server = config.mcpServers["github"]
        assert len(server.tools) == 1
        assert server.tools[0].name == "create_issue"
        assert server.tools[0].description == "Create a GitHub issue"
        # The JSON Schema round-trips.
        assert "owner" in server.tools[0].inputSchema["properties"]

    def test_empty_config_is_valid(self, loader: MCPConfigLoader) -> None:
        config = loader.load_from_string("{}")
        assert config.server_names() == ()

    def test_invalid_json_raises_value_error(self, loader: MCPConfigLoader) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            loader.load_from_string("{ not valid")

    def test_top_level_not_object_raises(self, loader: MCPConfigLoader) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            loader.load_from_string("[1, 2, 3]")

    def test_schema_violation_raises_value_error(
        self, loader: MCPConfigLoader
    ) -> None:
        # ``command`` is required; omitting it must fail validation.
        bad = json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "args": ["x"],
                        "tools": [],
                    }
                }
            }
        )
        with pytest.raises(ValueError, match="schema"):
            loader.load_from_string(bad)


# -- load(path) ----------------------------------------------------------


class TestLoadFromPath:
    def test_loads_real_dot_mcp_json(
        self, loader: MCPConfigLoader, env: dict[str, str]
    ) -> None:
        """The repo's own .mcp.json loads cleanly and contains the github server."""
        path = Path(__file__).resolve().parents[2] / ".mcp.json"
        if not path.exists():
            pytest.skip("no .mcp.json at repo root")
        config = loader.load(path)
        assert "github" in config.mcpServers
        # The github server has at least one tool defined.
        assert len(config.mcpServers["github"].tools) >= 1

    def test_missing_file_raises_file_not_found(
        self, loader: MCPConfigLoader
    ) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/.mcp.json")

    def test_malformed_file_raises_value_error(
        self, loader: MCPConfigLoader, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            loader.load(bad)


# -- data-class behaviour -------------------------------------------------


class TestDataClasses:
    def test_server_config_optional_fields_have_defaults(self) -> None:
        """A bare-minimum server entry should validate with sensible defaults."""
        server = MCPServerConfig(command="node")
        assert server.name == ""
        assert server.args == []
        assert server.env == {}
        assert server.tools == []

    def test_tool_spec_input_schema_defaults_to_empty(
        self, env: dict[str, str]
    ) -> None:
        spec = MCPToolSpec(name="t", description="d")
        assert spec.inputSchema == {}

    def test_mcp_config_server_names_preserves_order(self) -> None:
        cfg = MCPConfig(
            mcpServers={
                "a": MCPServerConfig(command="x"),
                "b": MCPServerConfig(command="y"),
                "c": MCPServerConfig(command="z"),
            }
        )
        assert cfg.server_names() == ("a", "b", "c")

    def test_mcp_config_tools_lists_all_pairs(self) -> None:
        cfg = MCPConfig(
            mcpServers={
                "a": MCPServerConfig(
                    command="x",
                    tools=[
                        MCPToolSpec(name="t1", description="d1"),
                        MCPToolSpec(name="t2", description="d2"),
                    ],
                ),
                "b": MCPServerConfig(
                    command="y",
                    tools=[MCPToolSpec(name="t3", description="d3")],
                ),
            }
        )
        pairs = cfg.tools()
        assert pairs == [
            ("a", cfg.mcpServers["a"].tools[0]),
            ("a", cfg.mcpServers["a"].tools[1]),
            ("b", cfg.mcpServers["b"].tools[0]),
        ]

    def test_args_accept_string_form(self) -> None:
        """Some users write ``"args": "single-arg"`` instead of a list."""
        server = MCPServerConfig.model_validate(
            {"command": "x", "args": "single-arg"}
        )
        assert server.args == ["single-arg"]
