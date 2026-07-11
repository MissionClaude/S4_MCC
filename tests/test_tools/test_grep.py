"""Unit tests for the :class:`GrepTool`.

The tool tries ripgrep first and falls back to a pure-Python scan. Tests
patch :meth:`_has_ripgrep` to force the fallback so they don't depend on
the system having ``rg`` installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scenario_4_dev_productivity.tools import GrepTool


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def python_only_grep() -> GrepTool:
    """A GrepTool that always uses the pure-Python fallback."""
    tool = GrepTool()
    # Force fallback — no reliance on the host having ripgrep.
    tool._has_ripgrep = staticmethod(lambda: False)  # type: ignore[method-assign]
    return tool


class TestHappyPath:
    def test_finds_matches(self, tmp_path: Path, python_only_grep: GrepTool) -> None:
        _write(tmp_path, "a.py", "alpha\nbeta\ngamma\n")
        _write(tmp_path, "b.py", "beta\n")
        result = python_only_grep.execute("u1", {"pattern": "beta", "path": str(tmp_path)})
        assert not result.is_error
        # Two files contain 'beta' — at least one match per file.
        lines = [line for line in result.content.splitlines() if line]
        assert len(lines) >= 2
        for line in lines:
            assert line.endswith(":beta")

    def test_no_matches_returns_empty_marker(
        self, tmp_path: Path, python_only_grep: GrepTool
    ) -> None:
        _write(tmp_path, "a.py", "alpha\n")
        result = python_only_grep.execute("u1", {"pattern": "MISSING", "path": str(tmp_path)})
        assert not result.is_error
        assert "no matches" in result.content.lower()

    def test_file_glob_filter(self, tmp_path: Path, python_only_grep: GrepTool) -> None:
        _write(tmp_path, "a.py", "needle\n")
        _write(tmp_path, "a.txt", "needle\n")
        result = python_only_grep.execute(
            "u1",
            {"pattern": "needle", "path": str(tmp_path), "file_glob": "*.py"},
        )
        assert not result.is_error
        lines = [line for line in result.content.splitlines() if line]
        assert len(lines) == 1
        assert lines[0].endswith(":needle")

    def test_max_matches_truncation(self, tmp_path: Path, python_only_grep: GrepTool) -> None:
        GrepTool.MAX_MATCHES = 3
        try:
            body = "\n".join(f"line {i}" for i in range(20))
            _write(tmp_path, "a.py", body)
            result = python_only_grep.execute("u1", {"pattern": r"line \d+", "path": str(tmp_path)})
        finally:
            GrepTool.MAX_MATCHES = 200
        assert not result.is_error
        assert "truncated" in result.content.lower()


class TestValidation:
    def test_missing_pattern(self, python_only_grep: GrepTool, tmp_path: Path) -> None:
        result = python_only_grep.execute("u1", {"path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"

    def test_empty_pattern(self, python_only_grep: GrepTool, tmp_path: Path) -> None:
        result = python_only_grep.execute("u1", {"pattern": "", "path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"

    def test_invalid_regex(self, python_only_grep: GrepTool, tmp_path: Path) -> None:
        result = python_only_grep.execute("u1", {"pattern": "[unclosed", "path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"
        assert "regex" in result.content.lower()

    def test_missing_path(self, python_only_grep: GrepTool) -> None:
        result = python_only_grep.execute("u1", {"pattern": "x", "path": "/no/such/dir"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_non_string_file_glob(self, python_only_grep: GrepTool, tmp_path: Path) -> None:
        result = python_only_grep.execute(
            "u1", {"pattern": "x", "path": str(tmp_path), "file_glob": 42}
        )
        assert result.is_error
        assert result.error_category == "validation"
