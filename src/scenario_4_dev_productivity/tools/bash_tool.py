"""Bash tool — execute a shell command with a hard timeout.

The most dangerous tool in the registry. It calls ``subprocess.run`` with:

* ``shell=False`` and ``shlex.split`` to avoid shell injection from
  untrusted model output;
* a hard wall-clock timeout (configurable, defaults to 30s) so a
  runaway command can't hang the loop;
* stdout/stderr captured separately so the model gets a clear picture
  of what happened when a command fails.

Sandboxing (cgroups, namespaces, seccomp) is out of scope for this
exam demo — the threat model is "model emits something stupid", not
"untrusted code execution".
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolParameterSchema, ToolResult


class BashTool:
    """Run a shell command and return its output.

    The command is passed as a single string and split with
    :func:`shlex.split`, so the model can use natural shell syntax
    (``ls -la | head``) without us giving it a raw shell. That keeps
    the surface area smaller than ``shell=True`` while still feeling
    like a shell.
    """

    name: str = "Bash"

    DEFAULT_TIMEOUT_SECONDS = 30.0

    #: Cap on the timeout a caller can request in a single call.
    MAX_TIMEOUT_SECONDS = 600.0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Run a shell command and return its combined output. The "
                "command is split with shlex so each token is a separate "
                "argv element; pipes, redirects, and globbing do NOT work "
                "(use sh -c '...' via Bash only if you really need them).\n\n"
                "Use this for tests, builds, formatting, one-off scripts, "
                "and any side-effecting operation. For reading files, use "
                "Read. For searching contents, use Grep. For finding files, "
                "use Glob.\n\n"
                "Input examples:\n"
                '  {"command": "pytest tests/ -x", "timeout": 60}\n'
                '  {"command": "git status --short"}\n'
                '  {"command": "uv run ruff check src/"}\n\n'
                "Boundary conditions:\n"
                "- Non-zero exit code is reported as a failure, not a success.\n"
                "- Commands exceeding the timeout are killed and reported as a "
                "transient, retryable failure."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "command": {
                        "type": "string",
                        "description": "Command line to execute. Split with shlex.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Wall-clock timeout in seconds. Default 30. "
                            f"Capped at {self.MAX_TIMEOUT_SECONDS:g}."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to the agent's CWD.",
                    },
                },
                required=["command"],
            ),
        )

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Bash requires a non-empty 'command' string argument.",
                category="validation",
                retryable=False,
            )

        timeout = self._resolve_timeout(arguments.get("timeout"))
        cwd = self._resolve_cwd(arguments.get("cwd"))

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Could not parse command: {exc}",
                category="validation",
                retryable=False,
            )
        if not argv:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message="Command parsed to zero arguments.",
                category="validation",
                retryable=False,
            )

        start = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S602 — intentional; argv is split
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                check=False,
            )
        except FileNotFoundError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Executable not found: {exc.filename or argv[0]}",
                category="validation",
                retryable=False,
            )
        except PermissionError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Permission denied executing {argv[0]}: {exc}",
                category="permission",
                retryable=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"Command {argv!r} exceeded the {timeout:.0f}s timeout and "
                    "was killed. Increase the timeout or narrow the command."
                ),
                category="transient",
                retryable=True,
            )
        except OSError as exc:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"OS error executing command: {exc.strerror or exc}",
                category="transient",
                retryable=True,
            )

        elapsed = time.monotonic() - start
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        body = self._format_output(argv, completed.returncode, elapsed, stdout, stderr)
        if completed.returncode == 0:
            return ToolResult.success(tool_use_id=tool_use_id, content=body)
        return ToolResult.failure(
            tool_use_id=tool_use_id,
            message=body,
            category="transient",
            retryable=False,
        )

    @staticmethod
    def _format_output(
        argv: list[str],
        returncode: int,
        elapsed: float,
        stdout: str,
        stderr: str,
    ) -> str:
        """Render stdout/stderr/exit code in a stable, easy-to-diagnose shape."""
        header = f"$ {' '.join(argv)}  [exit {returncode}, {elapsed:.2f}s]"
        chunks = [header]
        if stdout:
            chunks.append("--- stdout ---\n" + stdout.rstrip("\n"))
        if stderr:
            chunks.append("--- stderr ---\n" + stderr.rstrip("\n"))
        return "\n".join(chunks)

    def _resolve_timeout(self, raw: Any) -> float:
        """Coerce a timeout argument into a positive float, clamped to MAX."""
        if raw is None:
            return self.DEFAULT_TIMEOUT_SECONDS
        if not isinstance(raw, (int, float)) or raw <= 0:
            return self.DEFAULT_TIMEOUT_SECONDS
        return min(float(raw), self.MAX_TIMEOUT_SECONDS)

    @staticmethod
    def _resolve_cwd(raw: Any) -> str | None:
        """Validate the ``cwd`` argument; ``None`` means inherit the agent CWD."""
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        if not path.is_dir():
            return None
        return str(path)
