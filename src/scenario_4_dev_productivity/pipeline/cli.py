"""CLI entry point — the ``scenario-4`` console script.

Designed for CI / non-interactive use. Two subcommands today:

* ``scenario-4 run "task"`` — run a task in pipeline mode and print
  the result. ``--output-format json`` switches to JSON.
* ``scenario-4 compact <scratchpad>`` — show the parsed scratchpad
  entries (smoke test for the context module).

The CLI is intentionally thin: it builds a :class:`PipelineRunner`
from the project config and forwards. Real projects would build
their own runner with their own agent factory; this is the minimum
viable entry point that satisfies the spec's ``claude -p`` analogue.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from scenario_4_dev_productivity.agents import CoordinatorAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.context import ScratchpadManager
from scenario_4_dev_productivity.pipeline.runner import (
    JSONOutputFormat,
    OutputFormat,
    PipelineRunner,
    TextOutputFormat,
    make_agent_factory,
)
from scenario_4_dev_productivity.tools import default_registry


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level :class:`argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        prog="scenario-4",
        description="Developer Productivity Agent — Scenario 4 (non-interactive CLI).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a task in pipeline mode (equivalent to 'claude -p').")
    run.add_argument("task", help="The task to run.")
    run.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Output format. 'json' for machine-parseable output.",
    )
    run.add_argument(
        "--model",
        default=None,
        help="Override the coordinator model (default: from env / config).",
    )
    run.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override the max_turns cap on the agentic loop.",
    )

    compact = sub.add_parser(
        "compact",
        help="Show the contents of a scratchpad file as parsed entries.",
    )
    compact.add_argument(
        "path",
        nargs="?",
        default=config.scratchpad_path,
        help="Path to the scratchpad file (default: $SCRATCHPAD_PATH or .scratchpad.md).",
    )
    compact.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Output format. 'json' for a list of entries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a Unix-style exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "compact":
        return _cmd_compact(args)
    parser.error(f"unknown command: {args.command}")
    raise AssertionError("unreachable: parser.error should exit")


def _cmd_run(args: argparse.Namespace) -> int:
    """Implement the ``run`` subcommand."""
    missing = config.validate()
    if missing:
        sys.stderr.write("error: " + "; ".join(missing) + "\n")
        return 1
    client = AnthropicClient(
        api_key=config.anthropic_api_key,
        max_retries=3,
        timeout_seconds=float(config.anthropic_timeout_seconds),
    )
    registry = default_registry()
    factory = make_agent_factory(
        CoordinatorAgent,
        registry=registry,
        client=client,
        model=args.model,
        max_turns=args.max_turns or 15,
    )
    runner = PipelineRunner(client=client, agent_factory=factory)
    fmt: OutputFormat = JSONOutputFormat if args.output_format == "json" else TextOutputFormat
    rendered = runner.run_and_render(
        args.task,
        output=fmt,
        metadata={"model": args.model or config.coordinator_model},
    )
    sys.stdout.write(rendered)
    if not rendered.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    """Implement the ``compact`` subcommand (scratchpad inspection)."""
    manager = ScratchpadManager(args.path)
    if args.output_format == "json":
        entries = [e.model_dump() for e in manager.read_entries()]
        sys.stdout.write(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    else:
        text = manager.read()
        sys.stdout.write(text or "(scratchpad is empty)\n")
    return 0


__all__ = ["build_parser", "main"]
