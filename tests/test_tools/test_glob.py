"""Unit tests for the :class:`GlobTool`."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenario_4_dev_productivity.tools import GlobTool


def _write(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


class TestHappyPath:
    def test_matches_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py")
        _write(tmp_path, "b.py")
        _write(tmp_path, "c.txt")
        result = GlobTool().execute("u1", {"pattern": "*.py", "path": str(tmp_path)})
        assert not result.is_error
        assert sorted(result.content.splitlines()) == ["a.py", "b.py"]

    def test_recursive_glob(self, tmp_path: Path) -> None:
        _write(tmp_path, "deep/nested/a.py")
        result = GlobTool().execute("u1", {"pattern": "**/*.py", "path": str(tmp_path)})
        assert not result.is_error
        assert result.content.strip().endswith("a.py")

    def test_no_matches_returns_marker(self, tmp_path: Path) -> None:
        result = GlobTool().execute("u1", {"pattern": "*.nope", "path": str(tmp_path)})
        assert not result.is_error
        assert "no matches" in result.content.lower()

    def test_max_matches_truncation(self, tmp_path: Path) -> None:
        for i in range(5):
            _write(tmp_path, f"f{i}.py")
        GlobTool.MAX_MATCHES = 2
        try:
            result = GlobTool().execute("u1", {"pattern": "*.py", "path": str(tmp_path)})
        finally:
            GlobTool.MAX_MATCHES = 1000
        assert not result.is_error
        assert "truncated" in result.content.lower()


class TestValidation:
    def test_missing_pattern(self, tmp_path: Path) -> None:
        result = GlobTool().execute("u1", {"path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"

    def test_missing_path(self) -> None:
        result = GlobTool().execute("u1", {"pattern": "*.py", "path": "/no/such/dir"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_invalid_pattern_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the underlying glob raises (rare, but possible on some
        platforms / patterns), the tool surfaces a structured failure
        instead of crashing the loop."""

        def boom(self: Path, pattern: str) -> list[Path]:
            raise ValueError("simulated invalid pattern")

        monkeypatch.setattr(Path, "glob", boom)
        result = GlobTool().execute("u1", {"pattern": "*", "path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"
