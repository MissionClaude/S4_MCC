"""Edit tool — precise text replacement within an existing file.

Boundary: do NOT use this for full rewrites (use Write) or for new
files (use Write). Use Edit when the user wants a targeted change to
existing content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


def _count_occurrences(haystack: str, needle: str) -> int:
    """Count non-overlapping occurrences of ``needle`` in ``haystack``."""
    if not needle:
        return 0
    return haystack.count(needle)


class EditTool:
    """Replace a unique chunk of text in an existing file.

    The tool finds ``old_string`` in the file and replaces it with
    ``new_string``. The replacement is rejected when the target is not
    found or appears more than once — a safety net against the model
    editing the wrong site.
    """

    name: str = "Edit"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Replace a unique chunk of text in an existing file.\n\n"
                "Use this for targeted edits. The old_string must match "
                "exactly and must occur exactly once in the file. For "
                "new files or full rewrites, use Write.\n\n"
                "Input example:\n"
                "  {\n"
                '    "path": "src/example.py",\n'
                '    "old_string": "return \\"hi\\"",\n'
                '    "new_string": "return \\"hello\\""\n'
                "  }\n\n"
                "Boundary conditions:\n"
                "- All three arguments are required.\n"
                "- old_string must occur exactly once; 0 or >1 matches "
                "are rejected so the model can refine the input."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find (must occur once in the file).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                required=["path", "old_string", "new_string"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path")
        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        if not isinstance(path_str, str) or not path_str:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Edit requires a non-empty 'path' string argument.",
                category="validation",
                retryable=False,
            )
        if not isinstance(old_string, str):
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Edit requires 'old_string' to be a string.",
                category="validation",
                retryable=False,
            )
        if not isinstance(new_string, str):
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Edit requires 'new_string' to be a string.",
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

        try:
            original = path.read_text(encoding="utf-8")
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

        occurrences = _count_occurrences(original, old_string)
        if occurrences == 0:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"old_string not found in {path}. "
                    "Read the file and copy the exact text, including "
                    "indentation and line endings."
                ),
                category="validation",
                retryable=False,
            )
        if occurrences > 1:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"old_string occurs {occurrences} times in {path}; "
                    "it must be unique. Add more surrounding context to "
                    "disambiguate."
                ),
                category="validation",
                retryable=False,
            )

        updated = original.replace(old_string, new_string, 1)
        try:
            path.write_text(updated, encoding="utf-8")
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
            content=f"Edited {path} (replaced 1 occurrence).",
        )
