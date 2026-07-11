"""Integration tests for pipeline mode.

The pipeline runner is the spec's ``claude -p`` equivalent. These
tests prove that:

* a single ``run`` produces a well-formed :class:`PipelineResult`;
* the result renders to JSON when asked;
* two consecutive runs do NOT share state (the spec's session
  isolation rule);
* the multi-pass pattern runs per-file + integration review with
  fresh agents in between;
* the CLI entry point wires everything together and produces the
  expected stdout/stderr.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from scenario_4_dev_productivity.agents import (
    CoordinatorAgent,
    ExploreAgent,
    GenerateAgent,
)
from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.models.messages import StopReason
from scenario_4_dev_productivity.pipeline import (
    JSONOutputFormat,
    MultiPassResult,
    PipelineResult,
    PipelineRunner,
    TextOutputFormat,
    run_multi_pass,
)
from scenario_4_dev_productivity.pipeline.runner import (
    make_agent_factory,
)
from tests.test_integration.conftest import _text_response

# -- PipelineRunner --------------------------------------------------------


class TestPipelineRunnerConstruction:
    def test_requires_anthropic_client(self, registry) -> None:
        with pytest.raises(TypeError, match="AnthropicClient"):
            PipelineRunner(client=MagicMock(), agent_factory=lambda: MagicMock())  # type: ignore[arg-type]

    def test_requires_callable_factory(self, fake_client) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineRunner(client=fake_client, agent_factory="not callable")  # type: ignore[arg-type]


class TestPipelineRunnerRun:
    def test_runs_a_task_and_returns_a_result(
        self, fake_client, registry
    ) -> None:
        """A simple ``run("hi")`` returns a :class:`PipelineResult`
        with the final text and turn count."""
        factory = make_agent_factory(CoordinatorAgent, registry, fake_client, max_turns=3)
        runner = PipelineRunner(client=fake_client, agent_factory=factory)
        fake_client.script(_text_response("hello back", StopReason.END_TURN))
        result = runner.run("say hi")
        assert isinstance(result, PipelineResult)
        assert result.task == "say hi"
        assert result.text == "hello back"
        assert result.stop_reason == "end_turn"
        assert result.turn_count >= 1

    def test_renders_text_output(self, fake_client, registry) -> None:
        factory = make_agent_factory(CoordinatorAgent, registry, fake_client, max_turns=3)
        runner = PipelineRunner(client=fake_client, agent_factory=factory)
        fake_client.script(_text_response("done", StopReason.END_TURN))
        rendered = runner.run_and_render("hi", output=TextOutputFormat)
        assert rendered == "done"

    def test_renders_json_output(self, fake_client, registry) -> None:
        factory = make_agent_factory(CoordinatorAgent, registry, fake_client, max_turns=3)
        runner = PipelineRunner(client=fake_client, agent_factory=factory)
        fake_client.script(_text_response("ok", StopReason.END_TURN))
        rendered = runner.run_and_render(
            "hi",
            output=JSONOutputFormat,
            metadata={"branch": "main"},
        )
        # JSON must round-trip.
        data = json.loads(rendered)
        assert data["task"] == "hi"
        assert data["text"] == "ok"
        assert data["stop_reason"] == "end_turn"
        assert data["metadata"] == {"branch": "main"}

    def test_rejects_empty_task(self, fake_client, registry) -> None:
        factory = make_agent_factory(CoordinatorAgent, registry, fake_client, max_turns=3)
        runner = PipelineRunner(client=fake_client, agent_factory=factory)
        with pytest.raises(ValueError, match="task"):
            runner.run("   ")

    def test_factory_returning_wrong_type_raises(
        self, fake_client, registry
    ) -> None:
        def bad_factory() -> object:
            return object()  # not a BaseAgent

        runner = PipelineRunner(client=fake_client, agent_factory=bad_factory)
        with pytest.raises(TypeError, match="BaseAgent"):
            runner.run("hi")


class TestSessionIsolation:
    def test_two_runs_use_fresh_agents(
        self, fake_client, registry
    ) -> None:
        """The runner must call the factory once per run, never reuse
        the same agent across two ``run`` calls."""
        created: list[int] = []

        class CountingAgent(BaseAgent):
            def __init__(self) -> None:
                from scenario_4_dev_productivity.models.tools import AgentConfig

                # Real BaseAgent.__init__ so the runner's isinstance
                # check passes and the model/registry are wired up.
                super().__init__(
                    config=AgentConfig(
                        name="counting",
                        description="A test agent that records its id on construction.",
                        system_prompt="You are a counting agent.",
                        allowed_tools=[],
                        model="claude-haiku-4-5",
                    ),
                    registry=registry,
                    client=fake_client,
                    max_turns=1,
                )
                self._id = len(created)
                # Record the construction. The runner drives the
                # loop directly, so ``__init__`` is the only hook
                # left for proving session isolation.
                created.append(self._id)

        # The loop will hit the fake client's default response, which
        # is a fixed end_turn message. We don't care about the
        # response text — only about whether two agents were built.
        runner = PipelineRunner(client=fake_client, agent_factory=CountingAgent)
        r1 = runner.run("first")
        r2 = runner.run("second")
        # Two different agent instances were created.
        assert created == [0, 1]
        # Both runs succeeded and returned distinct results.
        assert r1.task == "first"
        assert r2.task == "second"
        # And the two PipelineResult objects are distinct.
        assert r1 is not r2


# -- Multi-pass -------------------------------------------------------------


class TestMultiPass:
    def test_runs_per_file_pass_then_integration(
        self, fake_client, registry
    ) -> None:
        files = ["src/a.py", "src/b.py", "src/c.py"]
        per_file_factory = make_agent_factory(
            ExploreAgent, registry, fake_client, max_turns=2
        )
        integration_agent = GenerateAgent(
            registry=registry, client=fake_client, max_turns=3
        )

        # Script three per-file responses + one integration response.
        for path in files:
            fake_client.script(_text_response(f"Findings for {path}", StopReason.END_TURN))
        fake_client.script(
            _text_response("Integration: nothing shared.", StopReason.END_TURN)
        )

        result = run_multi_pass(
            files=files,
            per_file_agent_factory=per_file_factory,
            integration_agent=integration_agent,
        )
        assert isinstance(result, MultiPassResult)
        assert len(result.per_file) == 3
        assert [p.label for p in result.per_file] == files
        assert result.integration.label == "integration"
        assert "shared" in result.integration.text.lower() or "integration" in result.integration.text.lower()

    def test_rejects_empty_files(self, fake_client, registry) -> None:
        agent = MagicMock(spec=BaseAgent)
        factory = MagicMock(return_value=agent)
        with pytest.raises(ValueError, match="files"):
            run_multi_pass(
                files=[],
                per_file_agent_factory=factory,
                integration_agent=agent,
            )

    def test_per_file_factory_called_per_file(
        self, fake_client, registry
    ) -> None:
        """The factory must be called once per file with a fresh agent."""
        calls: list[str] = []
        agents: list[BaseAgent] = []

        def factory(path: str) -> BaseAgent:
            calls.append(path)
            agent = ExploreAgent(registry=registry, client=fake_client, max_turns=2)
            agents.append(agent)
            return agent

        integration_agent = GenerateAgent(
            registry=registry, client=fake_client, max_turns=2
        )
        for _ in range(3):
            fake_client.script(_text_response("ok", StopReason.END_TURN))
        fake_client.script(_text_response("integration", StopReason.END_TURN))

        run_multi_pass(
            files=["a", "b", "c"],
            per_file_agent_factory=factory,
            integration_agent=integration_agent,
        )
        assert calls == ["a", "b", "c"]
        # Three distinct ExploreAgent instances.
        assert len({id(a) for a in agents}) == 3


# -- CLI entry point --------------------------------------------------------


class TestCliRun:
    def test_run_subcommand_smoke(self, monkeypatch, fake_client, capsys) -> None:
        """A direct invocation of the ``run`` subcommand with a
        pre-scripted client produces the expected stdout."""
        from scenario_4_dev_productivity.pipeline import cli

        # Provide a valid-looking API key so config.validate() passes.
        monkeypatch.setattr(config, "anthropic_api_key", "sk-test-key")
        # Replace the AnthropicClient class with a stub that delegates
        # to the test's fake client. The CLI imports the name into its
        # own namespace, so we have to patch BOTH the source module
        # and the CLI module.
        from scenario_4_dev_productivity import api
        from tests.test_integration.conftest import FakeAnthropicClient

        def _factory(*args: object, **kwargs: object) -> FakeAnthropicClient:
            # Wire the fake's records into the test's instance.
            return fake_client

        monkeypatch.setattr(api.client, "AnthropicClient", _factory)
        monkeypatch.setattr(cli, "AnthropicClient", _factory)
        fake_client.script(_text_response("hello world", StopReason.END_TURN))
        rc = cli.main(["run", "say hi", "--output-format", "text"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_compact_subcommand_smoke(self, tmp_path, capsys) -> None:
        """The ``compact`` subcommand prints a scratchpad's content."""
        from scenario_4_dev_productivity.context import ScratchpadManager
        from scenario_4_dev_productivity.pipeline import cli

        path = tmp_path / "sp.md"
        manager = ScratchpadManager(path)
        manager.append_finding("topic-1", "important finding")
        rc = cli.main(["compact", str(path), "--output-format", "text"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "topic-1" in out
        assert "important finding" in out
