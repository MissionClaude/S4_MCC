"""Write tool — write content to a file, creating parent dirs as needed.

Boundary: do NOT use this for small edits to an existing file (use Edit)
— overwriting with the full new content is wasteful and racy. Use Write
for new files or full rewrites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


class WriteTool:
    """Write a file's full content, creating parent directories on demand.

    Overwrites unconditionally — callers that want to preserve existing
    content should use :class:`EditTool` instead. Parent directories
    are created with ``parents=True`` so the tool "just works" for
    scaffolding new modules.
    """

    name: str = "Write"

    #: 4 MiB — generous enough for documentation and config files, still
    #: small enough to keep the context sane.
    MAX_BYTES = 4_194_304

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Write content to a file, overwriting any existing "
                "contents. Parent directories are created if they don't "
                "exist.\n\n"
                "Use this for new files or full rewrites. For precise "
                "edits to an existing file, use Edit — it is safer and "
                "preserves surrounding content.\n\n"
                "Input example:\n"
                '  {"path": "src/example.py", '
                '"content": "def hello() -> str:\\n    return \\"hi\\"\\n"}\n\n'
                "Boundary conditions:\n"
                "- Path and content are both required.\n"
                "- Content larger than 4 MiB is rejected."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                required=["path", "content"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path_str, str) or not path_str:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Write requires a non-empty 'path' string argument.",
                category="validation",
                retryable=False,
            )
        if not isinstance(content, str):
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Write requires 'content' to be a string.",
                category="validation",
                retryable=False,
            )
        if len(content.encode("utf-8")) > self.MAX_BYTES:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Content is {len(content)} bytes; the limit is {self.MAX_BYTES}.",
                category="validation",
                retryable=False,
            )

        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Permission denied creating parent directory for {path}: {exc}",
                category="permission",
                retryable=False,
            )
        except OSError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"OS error creating parent directory for {path}: {exc.strerror or exc}",
                category="transient",
                retryable=True,
            )

        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Permission denied writing {path}.",
                category="permission",
                retryable=False,
            )
        except OSError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"OS error writing {path}: {exc.strerror or exc}",
                category="transient",
                retryable=True,
            )

        return ToolResult.success(
            tool_use_id=tool_use_id,
            content=f"Wrote {len(content)} bytes to {path}.",
        )
