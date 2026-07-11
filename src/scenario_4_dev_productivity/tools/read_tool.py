"""Read tool — read the full contents of a file.

Boundary: do NOT use this for searching content (use Grep) or for finding
files by pattern (use Glob). Use it when you already know the path and
want the full text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


class ReadTool:
    """Read a file's full contents as UTF-8 text.

    Limitations:

    * Files larger than 1 MiB are rejected with a validation error so the
      loop's context isn't blown up by a single read call. The caller
      can re-issue with a smaller file or read a different range.
    * Binary files (anything that fails UTF-8 decoding) are rejected.
      Use a Bash command if you really need to inspect binaries.
    """

    name: str = "Read"

    #: 1 MiB — large enough for most source files, small enough to keep
    #: the agent's context healthy.
    MAX_BYTES = 1_048_576

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read the full contents of a file as UTF-8 text.\n\n"
                "Use this when you already know the file's path and want "
                "the entire text. For content search, use Grep. For path "
                "matching, use Glob.\n\n"
                "Input example:\n"
                '  {"path": "src/example.py"}\n\n'
                "Boundary conditions:\n"
                "- Files larger than 1 MiB are rejected.\n"
                "- Binary files (non-UTF-8) are rejected with a clear message."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read, relative to the working directory or absolute.",
                    }
                },
                required=["path"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        if not isinstance(path_str, str) or not path_str:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Read requires a non-empty 'path' string argument.",
                category="validation",
                retryable=False,
            )

        path = Path(path_str)
        if not path.exists():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"File not found: {path}",
                category="validation",
                retryable=False,
            )
        if not path.is_file():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Path is not a regular file: {path}",
                category="validation",
                retryable=False,
            )

        size = path.stat().st_size
        if size > self.MAX_BYTES:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"Refusing to read {path}: file is {size} bytes, "
                    f"exceeds the {self.MAX_BYTES}-byte limit."
                ),
                category="validation",
                retryable=False,
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"File is not valid UTF-8 ({exc.reason}). Use Bash to inspect binaries.",
                category="validation",
                retryable=False,
            )
        except PermissionError:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Permission denied reading {path}.",
                category="permission",
                retryable=False,
            )
        except OSError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"OS error reading {path}: {exc.strerror or exc}",
                category="transient",
                retryable=True,
            )

        # Surface a hint when the working dir would have given a different path.
        if not os.path.isabs(path_str) and not path.exists() and Path(path_str).exists():
            pass  # defensive — the earlier exists() check should catch this
        return ToolResult.success(tool_use_id=tool_use_id, content=content)
