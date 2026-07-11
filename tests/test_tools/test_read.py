"""Unit tests for the :class:`ReadTool`."""

from __future__ import annotations

from pathlib import Path

from scenario_4_dev_productivity.tools import ReadTool


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestHappyPath:
    def test_reads_text_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "hi.txt", "hello world\n")
        tool = ReadTool()
        result = tool.execute("u1", {"path": str(path)})
        assert not result.is_error
        assert result.content == "hello world\n"

    def test_reads_unicode(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "u.txt", "café — ñ")
        tool = ReadTool()
        result = tool.execute("u1", {"path": str(path)})
        assert not result.is_error
        assert "café" in result.content


class TestValidation:
    def test_missing_path_argument(self) -> None:
        result = ReadTool().execute("u1", {})
        assert result.is_error
        assert result.error_category == "validation"

    def test_non_string_path(self) -> None:
        result = ReadTool().execute("u1", {"path": 42})
        assert result.is_error
        assert result.error_category == "validation"

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = ReadTool().execute("u1", {"path": str(tmp_path / "missing.txt")})
        assert result.is_error
        assert result.error_category == "validation"
        assert "not found" in result.content.lower()

    def test_directory_rejected(self, tmp_path: Path) -> None:
        result = ReadTool().execute("u1", {"path": str(tmp_path)})
        assert result.is_error
        assert result.error_category == "validation"

    def test_oversize_file_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "big.txt", "a")
        # Patch the limit down so the test doesn't have to allocate >1 MiB.
        ReadTool.MAX_BYTES = 0
        try:
            result = ReadTool().execute("u1", {"path": str(path)})
        finally:
            ReadTool.MAX_BYTES = 1_048_576
        assert result.is_error
        assert result.error_category == "validation"
        assert "limit" in result.content.lower()

    def test_binary_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"\xff\xfe\x00\x01")
        result = ReadTool().execute("u1", {"path": str(path)})
        assert result.is_error
        assert result.error_category == "validation"
        assert "utf-8" in result.content.lower()
