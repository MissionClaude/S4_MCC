"""Unit tests for the :class:`ToolRegistry` dispatch table."""

from __future__ import annotations

from typing import Any

import pytest

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolResult
from scenario_4_dev_productivity.tools import BashTool, GlobTool, ReadTool
from scenario_4_dev_productivity.tools.registry import Tool, ToolRegistry


class _StubTool:
    """A minimal tool for testing registry behaviour."""

    def __init__(self, name: str, content: str = "ok") -> None:
        self.name = name
        self._content = content
        self.definition = ToolDefinition(
            name=name,
            description=f"Stub tool {name}",
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((tool_use_id, arguments))
        return ToolResult.success(tool_use_id=tool_use_id, content=self._content)


class TestRegistration:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        stub = _StubTool("Stub")
        registry.register(stub)
        assert registry.get("Stub") is stub
        assert "Stub" in registry
        assert len(registry) == 1

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool("A"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_StubTool("A"))

    def test_unknown_name_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get("Missing") is None
        assert "Missing" not in registry

    def test_unregister_is_idempotent(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool("A"))
        registry.unregister("A")
        assert "A" not in registry
        registry.unregister("A")  # second call is a no-op

    def test_iteration_is_insertion_order(self) -> None:
        registry = ToolRegistry()
        for name in ("B", "A", "C"):
            registry.register(_StubTool(name))
        assert registry.names == ("B", "A", "C")


class TestDefinitions:
    def test_definitions_returns_all_tools_in_order(self) -> None:
        registry = ToolRegistry()
        a, b = _StubTool("A"), _StubTool("B")
        registry.register(a)
        registry.register(b)
        defs = registry.definitions()
        assert [d.name for d in defs] == ["A", "B"]
        assert all(isinstance(d, ToolDefinition) for d in defs)

    def test_definitions_for_filters_by_allowlist(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool("A"))
        registry.register(_StubTool("B"))
        registry.register(_StubTool("C"))
        defs = registry.definitions_for(["A", "C", "Missing"])
        assert [d.name for d in defs] == ["A", "C"]


class TestExecution:
    def test_execute_returns_tool_result(self) -> None:
        registry = ToolRegistry()
        stub = _StubTool("Stub", content="hi")
        registry.register(stub)
        result = registry.execute("use-1", "Stub", {"x": 1})
        assert result.tool_use_id == "use-1"
        assert result.content == "hi"
        assert not result.is_error
        assert stub.calls == [("use-1", {"x": 1})]

    def test_execute_unknown_tool_returns_structured_failure(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("use-1", "Ghost", {})
        assert result.is_error
        assert result.error_category == "validation"
        assert "Unknown tool" in result.content
        assert not result.is_retryable

    def test_execute_swallows_exceptions_into_failure(self) -> None:
        class _Exploding:
            name = "Boom"

            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(name="Boom", description="boom")

            def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
                raise RuntimeError("kaboom")

        registry = ToolRegistry()
        registry.register(cast_tool(_Exploding()))
        result = registry.execute("use-1", "Boom", {})
        assert result.is_error
        assert "kaboom" in result.content
        assert result.error_category == "transient"
        assert not result.is_retryable


def cast_tool(obj: object) -> Tool:
    """Cast a duck-typed object to the :class:`Tool` protocol for type-checker appeasement."""
    return obj  # type: ignore[return-value]


class TestProtocolConformance:
    def test_builtin_tools_satisfy_protocol(self) -> None:
        """Every built-in tool exposes the ``name`` / ``definition`` / ``execute`` surface."""
        for tool in (ReadTool(), GlobTool(), BashTool()):
            assert isinstance(tool, Tool)
            assert tool.name
            assert tool.definition.name == tool.name
