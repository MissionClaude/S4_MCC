"""Unit tests for the :class:`WriteTool`."""

from __future__ import annotations

from pathlib import Path

from scenario_4_dev_productivity.tools import WriteTool


class TestHappyPath:
    def test_writes_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "a.txt"
        result = WriteTool().execute("u1", {"path": str(path), "content": "hi"})
        assert not result.is_error
        assert path.read_text(encoding="utf-8") == "hi"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text("old", encoding="utf-8")
        result = WriteTool().execute("u1", {"path": str(path), "content": "new"})
        assert not result.is_error
        assert path.read_text(encoding="utf-8") == "new"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "a.txt"
        result = WriteTool().execute("u1", {"path": str(path), "content": "ok"})
        assert not result.is_error
        assert path.exists()


class TestValidation:
    def test_missing_path(self, tmp_path: Path) -> None:
        result = WriteTool().execute("u1", {"content": "x"})
        assert result.is_error
        assert result.error_category == "validation"

    def test_missing_content(self, tmp_path: Path) -> None:
        result = WriteTool().execute("u1", {"path": str(tmp_path / "a.txt")})
        assert result.is_error
        assert result.error_category == "validation"

    def test_non_string_content(self, tmp_path: Path) -> None:
        result = WriteTool().execute("u1", {"path": str(tmp_path / "a.txt"), "content": 42})
        assert result.is_error
        assert result.error_category == "validation"

    def test_oversize_content_rejected(self) -> None:
        WriteTool.MAX_BYTES = 4
        try:
            result = WriteTool().execute("u1", {"path": "/tmp/x.txt", "content": "x" * 100})
        finally:
            WriteTool.MAX_BYTES = 4_194_304
        assert result.is_error
        assert result.error_category == "validation"
