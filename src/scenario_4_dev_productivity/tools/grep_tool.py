"""Grep tool — search file contents by regex.

Boundary: do NOT use this for finding files by name pattern (use Glob).
Use this for content search across the codebase.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


class GrepTool:
    """Search files for a regex pattern.

    Implementation prefers the system ``rg`` (ripgrep) binary when
    available because it is fast, handles binary file skipping
    correctly, and respects ``.gitignore``. Falls back to a pure-Python
    line-by-line scan otherwise.

    Output format: ``path:line_number:matched_line`` for each match.
    Empty output (no matches) is a valid success result.
    """

    name: str = "Grep"

    #: Cap on matches returned — protects the loop's context from
    #: "search for 'a' in src/" explosions.
    MAX_MATCHES = 200

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Search file contents for a regex pattern. Returns "
                "matches in the form 'path:line_number:matched_line'.\n\n"
                "Use this to find where a symbol, string, or pattern is "
                "defined or used. For path matching, use Glob. For full "
                "file contents, use Read.\n\n"
                "Input example:\n"
                '  {"pattern": "class AgentConfig", "path": "src/", '
                '"file_glob": "*.py"}\n\n'
                "Boundary conditions:\n"
                "- Empty pattern is rejected.\n"
                "- At most 200 matches are returned; refine the pattern or "
                "scope if you hit the cap."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Python regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to the current directory.",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional filename pattern (e.g. '*.py') to restrict the search.",
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
                message="Grep requires a non-empty 'pattern' string argument.",
                category="validation",
                retryable=False,
            )

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Invalid regex: {exc}",
                category="validation",
                retryable=False,
            )

        search_path = Path(arguments.get("path", "."))
        if not search_path.exists():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Search path not found: {search_path}",
                category="validation",
                retryable=False,
            )

        file_glob = arguments.get("file_glob")
        if file_glob is not None and not isinstance(file_glob, str):
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="'file_glob' must be a string when provided.",
                category="validation",
                retryable=False,
            )

        # Prefer ripgrep when available — much faster on large repos.
        if self._has_ripgrep() and not file_glob:
            return self._run_ripgrep(tool_use_id, pattern, search_path)

        # Pure-Python fallback. When file_glob is given, route through
        # the Python path so the glob filter is honoured consistently.
        return self._run_python(tool_use_id, compiled, search_path, file_glob)

    @staticmethod
    def _has_ripgrep() -> bool:
        try:
            subprocess.run(
                ["rg", "--version"],
                check=False,
                capture_output=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        return True

    def _run_ripgrep(
        self,
        tool_use_id: str,
        pattern: str,
        search_path: Path,
    ) -> ToolResult:
        try:
            result = subprocess.run(
                [
                    "rg",
                    "--line-number",
                    "--no-heading",
                    "--max-count",
                    str(self.MAX_MATCHES),
                    pattern,
                    str(search_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Grep timed out after 30s. Narrow the search path or pattern.",
                category="transient",
                retryable=True,
            )
        except FileNotFoundError:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="ripgrep binary disappeared mid-execution.",
                category="transient",
                retryable=True,
            )

        if result.returncode == 1:
            # rg exit code 1 = no matches — a valid success.
            return ToolResult.success(tool_use_id=tool_use_id, content="(no matches)")
        if result.returncode not in (0, 1):
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"ripgrep exited {result.returncode}: {result.stderr.strip() or '(no stderr)'}",
                category="transient",
                retryable=False,
            )

        return ToolResult.success(tool_use_id=tool_use_id, content=result.stdout)

    def _run_python(
        self,
        tool_use_id: str,
        compiled: re.Pattern[str],
        search_path: Path,
        file_glob: str | None,
    ) -> ToolResult:
        lines: list[str] = []
        try:
            iterator = (
                search_path.rglob(file_glob)
                if file_glob
                else (p for p in search_path.rglob("*") if p.is_file())
            )
            for path in iterator:
                if len(lines) >= self.MAX_MATCHES:
                    break
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, start=1):
                            if compiled.search(line):
                                lines.append(f"{path}:{lineno}:{line.rstrip()}")
                                if len(lines) >= self.MAX_MATCHES:
                                    break
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError) as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Error walking {search_path}: {exc}",
                category="permission",
                retryable=False,
            )

        if not lines:
            return ToolResult.success(tool_use_id=tool_use_id, content="(no matches)")
        suffix = ""
        if len(lines) >= self.MAX_MATCHES:
            suffix = f"\n\n[truncated at {self.MAX_MATCHES} matches]"
        return ToolResult.success(tool_use_id=tool_use_id, content="\n".join(lines) + suffix)
