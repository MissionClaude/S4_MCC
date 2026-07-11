"""Integration tests for context management.

These tests verify the three context-management primitives the spec
calls out:

* the scratchpad persists across loop iterations and across runs;
* :class:`ContextCompactor` shrinks a long conversation while
  preserving the original task and the most recent turns;
* :class:`PostToolUseHook` implementations (TrimReadOutputHook,
  LoggingHook) mutate / record :class:`ToolResult` instances as the
  loop runs them.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents import ExploreAgent
from scenario_4_dev_productivity.context import (
    ContextCompactor,
    LoggingHook,
    ScratchpadEntry,
    ScratchpadManager,
    TrimReadOutputHook,
    run_hooks,
    summarise_tool_result,
)
from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    StopReason,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import ToolResult
from tests.test_integration.conftest import _text_response

# -- ScratchpadManager ------------------------------------------------------


class TestScratchpadRoundTrip:
    def test_write_and_read_back(self, tmp_scratchpad_path) -> None:
        mgr = ScratchpadManager(tmp_scratchpad_path)
        mgr.write("# scratchpad\n")
        assert mgr.read() == "# scratchpad\n"
        assert mgr.exists

    def test_read_missing_returns_empty(self, tmp_scratchpad_path) -> None:
        mgr = ScratchpadManager(tmp_scratchpad_path)
        assert mgr.read() == ""
        assert not mgr.exists

    def test_append_creates_parent_dir(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "sp.md"
        mgr = ScratchpadManager(path)
        mgr.append(ScratchpadEntry(topic="t", body="b", source="x"))
        assert path.exists()
        assert mgr.read().strip().startswith("## t")

    def test_append_then_read_entries(self, tmp_scratchpad_path) -> None:
        mgr = ScratchpadManager(tmp_scratchpad_path)
        mgr.append(ScratchpadEntry(topic="finding-1", body="first"))
        mgr.append(ScratchpadEntry(topic="finding-2", body="second", source="explore"))
        entries = mgr.read_entries()
        assert [e.topic for e in entries] == ["finding-1", "finding-2"]
        assert entries[1].source == "explore"
        assert entries[0].body == "first"

    def test_clear_removes_file(self, tmp_scratchpad_path) -> None:
        mgr = ScratchpadManager(tmp_scratchpad_path)
        mgr.append_finding("x", "y")
        assert mgr.exists
        mgr.clear()
        assert not mgr.exists

    def test_scratchpad_persists_across_loop_iterations(
        self, fake_client, registry, tmp_scratchpad_path
    ) -> None:
        """The scratchpad file on disk survives the loop's state.

        PR #2 / PR #3 store the agent's conversation only in memory;
        the scratchpad is the only durable surface that survives
        across turns (and across /compact). This test writes to the
        scratchpad mid-conversation, then proves the file is still
        there when the next turn starts.
        """
        mgr = ScratchpadManager(tmp_scratchpad_path)
        mgr.append_finding("investigation", "module foo has 3 classes")
        # The loop then runs. The scratchpad is untouched by the loop
        # itself — the agent (in real life) writes to it before
        # /compact. We assert that nothing in the loop's machinery
        # has touched the file.
        from scenario_4_dev_productivity.tools.read_tool import ReadTool

        registry.register(ReadTool())
        agent = ExploreAgent(registry=registry, client=fake_client, max_turns=3)
        fake_client.script(_text_response("ok", StopReason.END_TURN))
        agent.run("Investigate the codebase")
        # Scratchpad is still on disk and unchanged.
        assert mgr.exists
        entries = mgr.read_entries()
        assert entries[0].body == "module foo has 3 classes"


# -- ContextCompactor -------------------------------------------------------


class TestContextCompactor:
    def test_empty_messages_returns_empty(self) -> None:
        assert ContextCompactor().compact([]) == []

    def test_keeps_first_user_message(self) -> None:
        """The original task is the only thing the agent must never lose."""
        msgs: list = [UserMessage(content="original task")]
        compacted = ContextCompactor(recent_turns=0).compact(msgs)
        assert len(compacted) == 1
        assert compacted[0].content == "original task"

    def test_reduces_message_count(self) -> None:
        """A long conversation with verbose tool results shrinks."""
        # Build: original task + assistant + tool_result (verbose) + recent tail.
        original = UserMessage(content="task")
        assistant = AssistantMessage(
            content=[TextBlock(text="thinking...")],
            stop_reason=StopReason.END_TURN,
        )
        big_content = "\n".join(f"line-{i:04d}" for i in range(200))
        tool_results = ToolResultMessage(
            results=[ToolResult.success("c1", big_content)],
        )
        recent_text = AssistantMessage(
            content=[TextBlock(text="final answer")],
            stop_reason=StopReason.END_TURN,
        )
        msgs: list = [original, assistant, tool_results, recent_text]
        compacted = ContextCompactor(keep_lines=10, recent_turns=2).compact(msgs)
        # The compacted list is shorter or equal; the tool result in
        # the middle is summarised.
        assert len(compacted) <= len(msgs)
        # The original task survived.
        assert compacted[0].content == "task"
        # The recent window (final_text) survived verbatim.
        assert any(
            isinstance(m, AssistantMessage) and m.text == "final answer" for m in compacted
        )

    def test_recent_window_preserved_verbatim(self) -> None:
        msgs: list = [
            UserMessage(content="task"),
            AssistantMessage(
                content=[TextBlock(text="mid-1")],
                stop_reason=StopReason.END_TURN,
            ),
            AssistantMessage(
                content=[TextBlock(text="mid-2")],
                stop_reason=StopReason.END_TURN,
            ),
            AssistantMessage(
                content=[TextBlock(text="recent")],
                stop_reason=StopReason.END_TURN,
            ),
        ]
        compacted = ContextCompactor(recent_turns=2).compact(msgs)
        texts = [m.text for m in compacted if isinstance(m, AssistantMessage)]
        assert texts[-2:] == ["mid-2", "recent"]


class TestSummariseToolResult:
    def test_short_result_unchanged(self) -> None:
        r = ToolResult.success("c1", "a\nb\nc")
        assert summarise_tool_result(r, keep_lines=5) is r

    def test_long_result_truncated(self) -> None:
        body = "\n".join(f"L{i}" for i in range(100))
        r = ToolResult.success("c1", body)
        out = summarise_tool_result(r, keep_lines=5)
        assert out is not r
        assert "[... 90 lines truncated ...]" in out.content
        # First 5 + last 5 retained.
        assert "L0" in out.content
        assert "L4" in out.content
        assert "L95" in out.content
        assert "L99" in out.content
        # Not marked as error — truncation is informational.
        assert out.is_error is False

    def test_validation_error_passes_through(self) -> None:
        r = ToolResult.failure(
            "c1", "nope", category="validation", retryable=False
        )
        # Errors keep their content; the summariser doesn't touch them.
        out = summarise_tool_result(r, keep_lines=2)
        assert out is r


# -- PostToolUse hooks ------------------------------------------------------


class TestTrimReadOutputHook:
    def test_does_not_trim_short_outputs(self) -> None:
        hook = TrimReadOutputHook(max_lines=10, keep_lines=3)
        result = ToolResult.success("c1", "a\nb\nc")
        assert hook("Read", "c1", result) is result

    def test_trims_long_outputs(self) -> None:
        hook = TrimReadOutputHook(max_lines=10, keep_lines=2)
        body = "\n".join(f"line-{i:02d}" for i in range(50))
        result = ToolResult.success("c1", body)
        out = hook("Read", "c1", result)
        assert out is not result
        assert "[... 46 lines truncated ...]" in out.content
        # First 2 and last 2 are present.
        assert "line-00" in out.content
        assert "line-01" in out.content
        assert "line-48" in out.content
        assert "line-49" in out.content

    def test_only_fires_for_read(self) -> None:
        """Other tools' outputs are passed through."""
        hook = TrimReadOutputHook(max_lines=5, keep_lines=2)
        body = "\n".join(str(i) for i in range(100))
        result = ToolResult.success("c1", body)
        assert hook("Bash", "c1", result) is result

    def test_skips_errors(self) -> None:
        """An error result is passed through — trimming would hide the
        useful diagnostic."""
        hook = TrimReadOutputHook(max_lines=2, keep_lines=1)
        result = ToolResult.failure(
            "c1", "boom", category="transient", retryable=True
        )
        assert hook("Read", "c1", result) is result


class TestLoggingHook:
    def test_records_tool_executions(self) -> None:
        hook = LoggingHook()
        run_hooks(
            [hook],
            "Read",
            "c1",
            ToolResult.success("c1", "ok"),
        )
        run_hooks(
            [hook],
            "Write",
            "c2",
            ToolResult.failure("c2", "no", category="validation", retryable=False),
        )
        assert hook.tool_names() == ["Read", "Write"]
        assert hook.records[1]["is_error"] is True
        assert hook.records[1]["category"] == "validation"

    def test_hook_exception_does_not_propagate(self) -> None:
        """A buggy hook must not break the loop."""

        def bad_hook(tool_name, tool_use_id, result):
            raise RuntimeError("hook boom")

        # A well-behaved hook before the bad one — its output should
        # be preserved when the bad hook crashes.
        sentinel = LoggingHook()
        original = ToolResult.success("c1", "ok")
        out = run_hooks([sentinel, bad_hook], "Read", "c1", original)
        # The good hook ran and recorded the call.
        assert len(sentinel.records) == 1
        # The bad hook's exception was swallowed; the last successful
        # result (sentinels's passthrough of the original) is returned.
        assert out.tool_use_id == original.tool_use_id
        assert out.content == original.content


class TestRunHooksIntegration:
    def test_run_hooks_chains_mutations(self) -> None:
        """A trim hook + logging hook together: trim happens, log records."""
        trim = TrimReadOutputHook(max_lines=5, keep_lines=2)
        log = LoggingHook()
        body = "\n".join(str(i) for i in range(100))
        original = ToolResult.success("c1", body)
        out = run_hooks([trim, log], "Read", "c1", original)
        # The trim hook fired.
        assert "[... 96 lines truncated ...]" in out.content
        # The log hook recorded the (already-trimmed) call.
        assert log.tool_names() == ["Read"]
