"""Unit tests for the :class:`EditTool`."""

from __future__ import annotations

from pathlib import Path

from scenario_4_dev_productivity.tools import EditTool


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestHappyPath:
    def test_replaces_unique_chunk(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.txt", "hello world\n")
        result = EditTool().execute(
            "u1",
            {
                "path": str(path),
                "old_string": "hello world",
                "new_string": "hi",
            },
        )
        assert not result.is_error
        assert path.read_text(encoding="utf-8") == "hi\n"

    def test_preserves_surrounding_content(self, tmp_path: Path) -> None:
        original = "line 1\nline 2\nline 3\n"
        path = _write(tmp_path, "a.txt", original)
        EditTool().execute(
            "u1",
            {
                "path": str(path),
                "old_string": "line 2",
                "new_string": "LINE 2",
            },
        )
        assert path.read_text(encoding="utf-8") == "line 1\nLINE 2\nline 3\n"


class TestValidation:
    def test_missing_path(self) -> None:
        result = EditTool().execute("u1", {"old_string": "a", "new_string": "b"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_non_string_old_string(self) -> None:
        result = EditTool().execute("u1", {"path": "/tmp/x", "old_string": 1, "new_string": "b"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_non_string_new_string(self) -> None:
        result = EditTool().execute("u1", {"path": "/tmp/x", "old_string": "a", "new_string": 1})
        assert result.is_error
        assert result.error_category == "validation"

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = EditTool().execute(
            "u1",
            {
                "path": str(tmp_path / "missing.txt"),
                "old_string": "a",
                "new_string": "b",
            },
        )
        assert result.is_error
        assert result.error_category == "validation"

    def test_old_string_not_found(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.txt", "alpha\nbeta\n")
        result = EditTool().execute(
            "u1",
            {"path": str(path), "old_string": "gamma", "new_string": "delta"},
        )
        assert result.is_error
        assert result.error_category == "validation"
        assert "not found" in result.content.lower()

    def test_ambiguous_old_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.txt", "alpha\nalpha\nbeta\n")
        result = EditTool().execute(
            "u1",
            {"path": str(path), "old_string": "alpha", "new_string": "ALPHA"},
        )
        assert result.is_error
        assert result.error_category == "validation"
        assert "2 times" in result.content
