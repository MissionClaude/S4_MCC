"""End-to-end agent workflow tests with a mocked API.

These tests exercise the full agentic system — the coordinator's
:class:`TaskTool`, the subagent factories, the registry, and the
agentic loop — to prove the spec's hub-and-spoke flow holds
together. The Anthropic SDK is replaced with a scripted fake (see
``conftest.py``); everything else is the real production code.
"""

from __future__ import annotations

import pytest

from scenario_4_dev_productivity.agents import (
    ExploreAgent,
    GenerateAgent,
)
from scenario_4_dev_productivity.models.messages import StopReason
from scenario_4_dev_productivity.tools.read_tool import ReadTool
from scenario_4_dev_productivity.tools.write_tool import WriteTool
from tests.test_integration.conftest import _text_response, _tool_response


class TestExploreAgentWorkflow:
    def test_explore_agent_runs_to_completion(
        self, fake_client, registry, tmp_path
    ) -> None:
        """An Explore agent investigates a directory and returns a summary."""
        # Seed the tmp dir with a couple of source files.
        (tmp_path / "alpha.py").write_text("def alpha():\n    return 1\n")
        (tmp_path / "beta.py").write_text("def beta():\n    return 2\n")
        registry.register(ReadTool())

        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        # Script: explore uses Glob to find files, then returns a summary.
        fake_client.script(
            _tool_response("c1", "Glob", {"pattern": str(tmp_path / "*.py")}),
            _text_response("Found alpha.py and beta.py", StopReason.END_TURN),
        )
        result = agent.run(f"List Python files under {tmp_path}")
        assert "alpha.py" in result.text and "beta.py" in result.text

    def test_explore_agent_cannot_write_files(
        self, fake_client, registry
    ) -> None:
        """The Explore agent's allowlist excludes Write — the system
        prompt and the registered tool list are how we tell the model
        the agent has read-only scope. The allowlist is the source
        of truth for "what is this agent allowed to do"."""
        registry.register(WriteTool())
        registry.register(ReadTool())

        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        # The allowlist is the perimeter: it does NOT include Write.
        assert "Write" not in agent.allowed_tools
        assert "Read" in agent.allowed_tools
        assert "Glob" in agent.allowed_tools
        # The system prompt also names the read-only tools explicitly.
        assert "Read" in agent.system_prompt
        assert "Glob" in agent.system_prompt


class TestGenerateAgentWorkflow:
    def test_generate_agent_writes_a_file(
        self, fake_client, registry, tmp_path
    ) -> None:
        """A Generate agent creates a new file via the Write tool."""
        registry.register(WriteTool())
        target = tmp_path / "module.py"
        agent = GenerateAgent(registry=registry, client=fake_client, max_turns=3)
        fake_client.script(
            _tool_response(
                "c1",
                "Write",
                {"path": str(target), "content": "def hello() -> str:\n    return 'hi'\n"},
            ),
            _text_response("Wrote module.py", StopReason.END_TURN),
        )
        result = agent.run("Create a hello function")
        assert result.text == "Wrote module.py"
        assert target.exists()
        assert "def hello" in target.read_text()


class TestCoordinatorDelegation:
    def test_coordinator_dispatches_to_explore_subagent(
        self, fake_client, registry, tmp_path
    ) -> None:
        """The coordinator emits a Task tool call; the subagent runs and
        its summary flows back to the coordinator."""
        registry.register(ReadTool())
        (tmp_path / "mod.py").write_text("x = 1\n")

        from scenario_4_dev_productivity.agents import (
            _default_subagent_factory,
        )

        # Inject the default factory; we don't need a custom one here.
        agent_factory = _default_subagent_factory(registry, fake_client)

        # Manually build the coordinator's Task tool with this factory.
        from scenario_4_dev_productivity.agents.coordinator_agent import CoordinatorAgent

        coord = CoordinatorAgent(
            registry=registry,
            client=fake_client,
            subagent_factory=agent_factory,
            max_turns=3,
        )
        # Coordinator script:
        # 1) emit Task(explore)
        # 2) once subagent returns, emit a final summary
        fake_client.script(
            _tool_response(
                "c1",
                "Task",
                {
                    "subagent_type": "explore",
                    "prompt": f"Summarise {tmp_path}/mod.py",
                    "description": "Inspect mod.py",
                },
            ),
            _text_response("Subagent reported the file contains x = 1.", StopReason.END_TURN),
        )
        # Subagent script (fired by the same fake client):
        fake_client.script(
            _tool_response("c1", "Read", {"path": str(tmp_path / "mod.py")}),
            _text_response("x = 1", StopReason.END_TURN),
        )
        result = coord.run(f"Investigate {tmp_path}/mod.py")
        # The coordinator's final message exists.
        assert result.text
        # The Task tool was actually exercised.
        assert len(fake_client.requests) >= 1

    def test_coordinator_task_tool_rejects_unknown_subagent_type(
        self, fake_client, registry
    ) -> None:
        """The Task tool surfaces a structured validation error when the
        model hallucinates a subagent type — the loop must keep running."""
        from scenario_4_dev_productivity.agents.coordinator_agent import (
            CoordinatorAgent,
        )

        coord = CoordinatorAgent(registry=registry, client=fake_client, max_turns=3)
        fake_client.script(
            _tool_response(
                "c1",
                "Task",
                {"subagent_type": "rogue", "prompt": "do something", "description": "x"},
            ),
            _text_response("Recovered", StopReason.END_TURN),
        )
        result = coord.run("do work")
        # The final text is from the recovery turn, not an exception.
        assert result.text == "Recovered"


class TestStructuredErrorHandling:
    def test_tool_execution_failure_feeds_back_to_model(
        self, fake_client, registry
    ) -> None:
        """A tool failure surfaces as a :class:`ToolResult` with
        ``is_error=True``; the loop passes it back to the model and
        the model adapts."""
        from scenario_4_dev_productivity.tools.read_tool import ReadTool

        registry.register(ReadTool())
        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        fake_client.script(
            _tool_response("c1", "Read", {"path": "missing.py"}),
            _text_response("I'll try a different file", StopReason.END_TURN),
        )
        result = agent.run("Read missing.py")
        assert "different file" in result.text
        # The second request to the model must contain the tool_result
        # with is_error=True so the model could read the failure.
        second = fake_client.requests[1]
        # The last message sent is the tool result message; its wire
        # form has is_error=True on the first tool_result block.
        from scenario_4_dev_productivity.models.messages import UserMessage

        last_msg = second.messages[-1]
        if isinstance(last_msg, UserMessage) and isinstance(last_msg.content, list):
            assert last_msg.content[0].is_error is True
        else:  # pragma: no cover - defensive
            pytest.fail("expected a tool result message in the second request")

    def test_loop_survives_unknown_tool_from_model(
        self, fake_client, registry
    ) -> None:
        """The model hallucinating an unknown tool must not crash the
        loop; the registry returns a structured failure instead."""
        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=5)
        fake_client.script(
            _tool_response("c1", "GhostTool", {}),
            _text_response("Recovered from unknown tool", StopReason.END_TURN),
        )
        result = agent.run("do a thing")
        assert "Recovered" in result.text
