"""Glob tool — find files by path pattern.

Boundary: do NOT use this for content search (use Grep). Use this for
path patterns like ``**/*.py`` or ``src/**/test_*.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


class GlobTool:
    """Match file paths against a glob pattern.

    Wraps :meth:`pathlib.Path.glob` so the supported syntax is whatever
    Python's stdlib supports (``*``, ``**``, ``?``, ``[abc]``). Results
    are sorted and relative to the search root for stable output.
    """

    name: str = "Glob"

    #: Cap on matches — protects the loop from a runaway ``**`` scan.
    MAX_MATCHES = 1000

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Match file paths against a glob pattern. Returns paths "
                "relative to the search root, one per line.\n\n"
                "Use this to find files by name or path. For content "
                "search, use Grep. For full file contents, use Read.\n\n"
                "Input examples:\n"
                '  {"pattern": "src/**/*.py"}\n'
                '  {"pattern": "tests/**/test_*.py", "path": "."}\n'
                '  {"pattern": "**/config.{toml,json}"}\n\n'
                "Boundary conditions:\n"
                "- Pattern is required and must be a non-empty string.\n"
                "- At most 1000 matches are returned."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. 'src/**/*.py', '**/test_*.py').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search from. Defaults to the current directory.",
                    },
                },
                required=["pattern"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Glob requires a non-empty 'pattern' string argument.",
                category="validation",
                retryable=False,
            )

        root = Path(arguments.get("path", "."))
        if not root.exists():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Search root not found: {root}",
                category="validation",
                retryable=False,
            )

        try:
            matches = sorted(p for p in root.glob(pattern) if p.is_file())
        except (NotImplementedError, ValueError) as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Glob pattern {pattern!r} is not supported: {exc}",
                category="validation",
                retryable=False,
            )
        except PermissionError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Permission denied walking {root}: {exc}",
                category="permission",
                retryable=False,
            )

        if not matches:
            return ToolResult.success(tool_use_id=tool_use_id, content="(no matches)")

        truncated = len(matches) > self.MAX_MATCHES
        matches = matches[: self.MAX_MATCHES]
        # Render relative to the search root so output is concise and stable.
        rendered = "\n".join(_relative(p, root) for p in matches)
        suffix = f"\n\n[truncated at {self.MAX_MATCHES} matches]" if truncated else ""
        return ToolResult.success(tool_use_id=tool_use_id, content=rendered + suffix)


def _relative(path: Path, root: Path) -> str:
    """Render ``path`` relative to ``root`` when possible, else as string."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
