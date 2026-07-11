"""Unit tests for the tool-related Pydantic models.

These tests pin the shape the rest of the system depends on. Any change
to a field name or default here is a breaking change for the agentic
loop, the API client, and the prompt templates that reference tools.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scenario_4_dev_productivity.models.tools import (
    AgentConfig,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)


class TestToolParameterSchema:
    def test_defaults_render_valid_json_schema(self) -> None:
        schema = ToolParameterSchema()
        rendered = schema.to_json_schema()
        assert rendered == {"type": "object"}

    def test_explicit_properties_and_required(self) -> None:
        schema = ToolParameterSchema(
            properties={"path": {"type": "string"}},
            required=["path"],
        )
        assert schema.to_json_schema() == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }


class TestToolDefinition:
    def test_minimal_definition(self) -> None:
        tool = ToolDefinition(name="Read", description="Read a file")
        assert tool.name == "Read"
        assert tool.description == "Read a file"
        assert tool.parameters.to_json_schema() == {"type": "object"}

    def test_to_anthropic_tool_shape(self) -> None:
        tool = ToolDefinition(
            name="Read",
            description="Read a file",
            parameters=ToolParameterSchema(
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
        )
        wire = tool.to_anthropic_tool()
        assert wire == {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolDefinition(name="", description="x")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolDefinition(name="Read", description="x", extra_field="oops")  # type: ignore[call-arg]


class TestToolCall:
    def test_defaults(self) -> None:
        call = ToolCall(id="t1", name="Read")
        assert call.input == {}

    def test_input_passes_through(self) -> None:
        call = ToolCall(id="t1", name="Read", input={"path": "/etc/passwd"})
        assert call.input == {"path": "/etc/passwd"}


class TestToolResult:
    def test_success_constructor_sets_safe_defaults(self) -> None:
        result = ToolResult.success("t1", "file contents")
        assert result.tool_use_id == "t1"
        assert result.content == "file contents"
        assert result.is_error is False
        assert result.error_category is None
        assert result.is_retryable is False

    def test_failure_constructor_sets_recovery_signals(self) -> None:
        result = ToolResult.failure(
            "t1",
            "command timed out after 30s",
            category="transient",
            retryable=True,
        )
        assert result.is_error is True
        assert result.error_category == "transient"
        assert result.is_retryable is True
        assert result.content == "command timed out after 30s"

    def test_validation_error_is_not_retryable(self) -> None:
        result = ToolResult.failure(
            "t1", "missing required field 'path'", category="validation", retryable=False
        )
        assert result.is_retryable is False
        assert result.error_category == "validation"


class TestAgentConfig:
    def test_minimal_config(self) -> None:
        config = AgentConfig(
            name="Explore",
            description="Reads code",
            system_prompt="You explore.",
        )
        assert config.allowed_tools == []
        assert config.model == "claude-haiku"

    def test_full_config(self) -> None:
        config = AgentConfig(
            name="Coordinator",
            description="Orchestrates",
            system_prompt="You orchestrate.",
            allowed_tools=["Task", "Read"],
            model="claude-sonnet",
        )
        assert config.allowed_tools == ["Task", "Read"]
        assert config.model == "claude-sonnet"
