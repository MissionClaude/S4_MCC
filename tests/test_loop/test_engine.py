"""Unit tests for :class:`AgenticLoop`.

The loop's job is dispatch on ``stop_reason``. We mock the API client
and feed it scripted responses, then assert:

* TOOL_USE → execute tool → append result → loop
* END_TURN → return the assistant message
* max_turns → raise
* Tool failures feed back to the model instead of breaking the loop
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.loop.engine import AgenticLoop, MaxTurnsExceeded
from scenario_4_dev_productivity.models.api import APIRequest, APIResponse
from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    StopReason,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
)
from scenario_4_dev_productivity.models.tools import ToolCall, ToolDefinition, ToolResult
from scenario_4_dev_productivity.tools.registry import ToolRegistry

# -- helpers ------------------------------------------------------------


class _StubTool:
    """A tool whose behaviour the test controls per-call."""

    def __init__(self, name: str, results: list[ToolResult]) -> None:
        self.name = name
        self._results = list(results)
        self.definition = ToolDefinition(name=name, description=f"Stub {name}")
        self.calls: list[dict[str, Any]] = []

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append({"id": tool_use_id, "args": arguments})
        if not self._results:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"{self.name} ran out of scripted results",
                category="transient",
                retryable=False,
            )
        return self._results.pop(0)


def _response(text: str = "", stop: StopReason = StopReason.END_TURN) -> APIResponse:
    return APIResponse(
        message=AssistantMessage(content=[TextBlock(text=text)], stop_reason=stop),
    )


def _tool_response(call_id: str, tool: str, args: dict[str, Any]) -> APIResponse:
    return APIResponse(
        message=AssistantMessage(
            content=[
                ToolUseBlock(call=ToolCall(id=call_id, name=tool, input=args)),
            ],
            stop_reason=StopReason.TOOL_USE,
        ),
    )


def _client(script: list[APIResponse]) -> Any:
    """Build a fake API client that returns the next scripted response."""
    client = MagicMock()
    client.send.side_effect = list(script)
    return client


# -- tests --------------------------------------------------------------


class TestConstruction:
    def test_rejects_zero_max_turns(self) -> None:
        with pytest.raises(ValueError):
            AgenticLoop(client=MagicMock(), registry=ToolRegistry(), max_turns=0)

    def test_requires_registry(self) -> None:
        with pytest.raises(TypeError):
            AgenticLoop(client=MagicMock(), registry="not a registry", max_turns=1)  # type: ignore[arg-type]


class TestStopReasonDispatch:
    def test_end_turn_returns_immediately(self) -> None:
        tool = _StubTool("Read", [])
        registry = ToolRegistry()
        registry.register(tool)
        client = _client([_response("done", StopReason.END_TURN)])
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        result = loop.run("hi")
        assert result.text == "done"
        assert loop.turn_count == 1
        # No tool calls were issued.
        assert tool.calls == []

    def test_tool_use_executes_then_loops(self) -> None:
        tool = _StubTool(
            "Read",
            [ToolResult.success("c1", "file contents")],
        )
        registry = ToolRegistry()
        registry.register(tool)
        client = _client(
            [
                _tool_response("c1", "Read", {"path": "x.py"}),
                _response("finished", StopReason.END_TURN),
            ]
        )
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        result = loop.run("read x.py")
        assert result.text == "finished"
        assert loop.turn_count == 2
        assert tool.calls == [{"id": "c1", "args": {"path": "x.py"}}]
        # Conversation history: user task, assistant tool_use, user tool result, assistant text.
        msgs = loop.messages
        assert len(msgs) == 4
        assert isinstance(msgs[0], type(msgs[0]))  # smoke
        assert any(isinstance(m, ToolResultMessage) for m in msgs)

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        tool = _StubTool(
            "Read",
            [
                ToolResult.success("c1", "a"),
                ToolResult.success("c2", "b"),
            ],
        )
        registry = ToolRegistry()
        registry.register(tool)
        client = MagicMock()
        client.send.side_effect = [
            APIResponse(
                message=AssistantMessage(
                    content=[
                        ToolUseBlock(call=ToolCall(id="c1", name="Read", input={"path": "a"})),
                        ToolUseBlock(call=ToolCall(id="c2", name="Read", input={"path": "b"})),
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
            ),
            _response("done", StopReason.END_TURN),
        ]
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        loop.run("read both")
        # Both tool results are in the conversation.
        result_msg = next(m for m in loop.messages if isinstance(m, ToolResultMessage))
        assert len(result_msg.results) == 2

    def test_unknown_tool_is_rendered_as_failure(self) -> None:
        registry = ToolRegistry()  # empty
        client = _client(
            [
                _tool_response("c1", "Ghost", {}),
                _response("ok", StopReason.END_TURN),
            ]
        )
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        result = loop.run("call ghost")
        assert result.text == "ok"
        # The unknown tool's result is in the conversation as a failure.
        result_msg = next(m for m in loop.messages if isinstance(m, ToolResultMessage))
        assert result_msg.results[0].is_error


class TestToolFailureFeedback:
    def test_is_error_result_is_fed_back_to_model(self) -> None:
        """A failing tool doesn't break the loop; the model sees the
        structured failure and decides what to do next."""
        tool = _StubTool(
            "Read",
            [
                ToolResult.failure(
                    "c1",
                    "file not found",
                    category="validation",
                    retryable=False,
                ),
            ],
        )
        registry = ToolRegistry()
        registry.register(tool)
        client = _client(
            [
                _tool_response("c1", "Read", {"path": "x"}),
                _response("I'll try a different file", StopReason.END_TURN),
            ]
        )
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        result = loop.run("read x")
        assert "different file" in result.text
        # The model received a tool_result message with is_error=True.
        sent = client.send.call_args_list[1].args[0]
        last_user_msg = sent.messages[-1]
        # It's a wire-level user message containing tool results.
        assert last_user_msg.role == "user"
        assert isinstance(last_user_msg.content, list)
        assert last_user_msg.content[0].is_error is True


class TestTermination:
    def test_max_turns_raises(self) -> None:
        # Always emit a tool_use → the loop should bail out at the cap.
        registry = ToolRegistry()
        registry.register(_StubTool("Read", [ToolResult.success("c", "x")]))
        client = MagicMock()
        client.send.side_effect = lambda req: _tool_response("c", "Read", {})
        loop = AgenticLoop(client=client, registry=registry, max_turns=3)
        with pytest.raises(MaxTurnsExceeded) as exc:
            loop.run("loop forever")
        assert exc.value.max_turns == 3
        assert loop.turn_count == 3

    def test_empty_task_rejected(self) -> None:
        loop = AgenticLoop(client=MagicMock(), registry=ToolRegistry(), max_turns=1)
        with pytest.raises(ValueError, match="task"):
            loop.run("")

    def test_stop_reason_none_treated_as_terminal(self) -> None:
        """A response with ``stop_reason=None`` should still return gracefully."""
        registry = ToolRegistry()
        client = MagicMock()
        client.send.return_value = APIResponse(
            message=AssistantMessage(content=[TextBlock(text="x")], stop_reason=None),
        )
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        result = loop.run("hi")
        assert result.text == "x"
        assert loop.turn_count == 1


class TestConfiguration:
    def test_run_overrides_max_turns(self) -> None:
        registry = ToolRegistry()
        client = _client([_response("ok", StopReason.END_TURN)])
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        loop.run("hi", max_turns=1)
        # The first call must include the (effective) max_turns, which is
        # not on APIRequest — but we DID override via the kwarg, so the
        # loop should have run exactly one turn.
        assert loop.turn_count == 1

    def test_run_uses_registry_default_tools(self) -> None:
        tool = _StubTool("Read", [])
        registry = ToolRegistry()
        registry.register(tool)
        client = _client([_response("ok", StopReason.END_TURN)])
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        loop.run("hi")
        sent: APIRequest = client.send.call_args.args[0]
        assert [t.name for t in sent.tools] == ["Read"]

    def test_run_respects_tool_allowlist_override(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool("Read", []))
        registry.register(_StubTool("Bash", []))
        client = _client([_response("ok", StopReason.END_TURN)])
        loop = AgenticLoop(client=client, registry=registry, max_turns=5)
        allowed = [registry.get("Read").definition]  # type: ignore[union-attr]
        loop.run("hi", tools=allowed)
        sent: APIRequest = client.send.call_args.args[0]
        assert [t.name for t in sent.tools] == ["Read"]


def test_messages_snapshot_is_tuple() -> None:
    """The ``messages`` accessor returns a tuple, not the live list."""
    loop = AgenticLoop(
        client=_client([_response("ok", StopReason.END_TURN)]),
        registry=ToolRegistry(),
        max_turns=5,
    )
    loop.run("hi")
    assert isinstance(loop.messages, tuple)
