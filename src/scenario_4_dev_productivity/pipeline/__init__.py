"""Pipeline mode — non-interactive agentic execution.

Two primitives, both in service of the spec's "non-interactive
execution via ``claude -p``" requirement:

* :class:`PipelineRunner` — runs a single task end-to-end. Each call
  gets a fresh :class:`BaseAgent` (session isolation) and returns a
  :class:`PipelineResult` the caller can serialise to JSON.
* :func:`run_multi_pass` — the spec's multi-pass pattern: per-file
  analysis pass followed by an integration review pass. Each pass
  uses a fresh agent so the integration reviewer never sees the
  per-file pass's verbose tool outputs.

The :mod:`scenario_4_dev_productivity.pipeline.cli` module provides
the ``scenario-4`` console entry point used by ``uv run scenario-4 run``.
"""

from __future__ import annotations

from scenario_4_dev_productivity.pipeline.multi_pass import (
    MultiPassResult,
    PassResult,
    run_multi_pass,
)
from scenario_4_dev_productivity.pipeline.runner import (
    JSONOutputFormat,
    OutputFormat,
    PipelineResult,
    PipelineRunner,
    TextOutputFormat,
)

__all__ = [
    "JSONOutputFormat",
    "MultiPassResult",
    "OutputFormat",
    "PassResult",
    "PipelineResult",
    "PipelineRunner",
    "TextOutputFormat",
    "run_multi_pass",
]
