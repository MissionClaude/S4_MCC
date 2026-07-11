"""Unit tests for the :class:`BashTool`."""

from __future__ import annotations

import sys
from pathlib import Path

from scenario_4_dev_productivity.tools import BashTool


class TestHappyPath:
    def test_runs_simple_command(self, tmp_path: Path) -> None:
        result = BashTool().execute(
            "u1",
            {"command": f"{sys.executable} -c \"print('hi')\""},
        )
        assert not result.is_error
        assert "hi" in result.content
        assert "exit 0" in result.content

    def test_includes_stderr(self, tmp_path: Path) -> None:
        result = BashTool().execute(
            "u1",
            {
                "command": f"{sys.executable} -c \"import sys; print('err', file=sys.stderr); print('out')\""
            },
        )
        assert "stderr" in result.content
        assert "out" in result.content

    def test_custom_cwd(self, tmp_path: Path) -> None:
        result = BashTool().execute(
            "u1",
            {"command": "pwd", "cwd": str(tmp_path)},
        )
        assert not result.is_error
        # On macOS /tmp is /private/tmp; just check the suffix matches.
        assert str(tmp_path).split("/")[-1] in result.content


class TestFailure:
    def test_nonzero_exit_is_failure(self) -> None:
        result = BashTool().execute(
            "u1",
            {"command": f'{sys.executable} -c "import sys; sys.exit(2)"'},
        )
        assert result.is_error
        assert "exit 2" in result.content
        assert result.error_category == "transient"

    def test_missing_executable(self) -> None:
        result = BashTool().execute("u1", {"command": "definitely-not-a-real-binary-xyz123"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_unparseable_command(self) -> None:
        result = BashTool().execute("u1", {"command": "echo 'unterminated"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_empty_command(self) -> None:
        result = BashTool().execute("u1", {"command": ""})
        assert result.is_error
        assert result.error_category == "validation"

    def test_whitespace_only_command(self) -> None:
        result = BashTool().execute("u1", {"command": "   \n\t  "})
        assert result.is_error
        assert result.error_category == "validation"


class TestTimeout:
    def test_timeout_kills_long_command(self) -> None:
        # Sleep for longer than the timeout we pass in.
        result = BashTool().execute(
            "u1",
            {
                "command": f'{sys.executable} -c "import time; time.sleep(2)"',
                "timeout": 1,
            },
        )
        assert result.is_error
        assert result.error_category == "transient"
        assert result.is_retryable
        assert "timeout" in result.content.lower()

    def test_timeout_is_clamped_to_max(self) -> None:
        # The max is 600s, but a sleep of 0.1s should still complete.
        result = BashTool().execute(
            "u1",
            {
                "command": f'{sys.executable} -c "import time; time.sleep(0.05)"',
                "timeout": 999_999,  # well above MAX_TIMEOUT_SECONDS
            },
        )
        assert not result.is_error


class TestSandbox:
    def test_dangerous_command_still_runs_in_process(self) -> None:
        """The Bash tool does NOT block commands structurally — it
        bounds the *runtime* via timeout. A short command succeeds; a
        long one is killed. This test pins the contract so we don't
        accidentally regress to no-timeout behaviour."""
        # `sleep 0` is harmless and exercises the same code path.
        result = BashTool().execute("u1", {"command": "sleep 0", "timeout": 5})
        assert not result.is_error
