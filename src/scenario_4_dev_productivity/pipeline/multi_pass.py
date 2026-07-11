"""Multi-pass analysis — per-file pass + integration review pass.

The spec describes the multi-pass pattern as: per-file analysis
followed by an integration review. This module implements that as two
:func:`run_pass` calls wired together:

1. **Per-file pass.** For each file in ``files``, run a fresh
   read-only agent and ask for a structured finding. Each pass is
   isolated — the agent for file A has no memory of file B.
2. **Integration pass.** Once the per-file findings are collected,
   run a separate agent with all findings in its prompt. The
   integration agent's job is to spot cross-file concerns: shared
   types, conflicting assumptions, missing tests, etc.

Session isolation is enforced at the call site: each
:func:`run_pass` invocation creates a fresh agent. The integration
agent never sees the per-file pass's tool calls; it only sees the
extracted findings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.models.messages import AssistantMessage


@dataclass(frozen=True)
class PassResult:
    """The outcome of one pass (one file, or the integration pass)."""

    label: str
    text: str
    turn_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "text": self.text,
            "turn_count": self.turn_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiPassResult:
    """The combined output of a multi-pass run."""

    per_file: tuple[PassResult, ...]
    integration: PassResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_file": [p.to_dict() for p in self.per_file],
            "integration": self.integration.to_dict(),
        }

    @property
    def summary(self) -> str:
        """A short text summary suitable for CI logs."""
        lines = [
            f"Multi-pass: {len(self.per_file)} per-file finding(s) + integration review",
            "",
            "Per-file findings:",
            *(
                f"  - {p.label}: {p.text.splitlines()[0] if p.text else '(empty)'}"
                for p in self.per_file
            ),
            "",
            "Integration review:",
            self.integration.text or "(empty)",
        ]
        return "\n".join(lines)


def run_pass(
    agent: BaseAgent,
    task: str,
    *,
    label: str,
) -> PassResult:
    """Run a single pass on ``task`` and wrap the output in a :class:`PassResult`."""
    if not isinstance(agent, BaseAgent):
        raise TypeError("agent must be a BaseAgent")
    # Build the loop directly so we can read the turn count after the
    # run; same approach as :meth:`PipelineRunner.run`.
    loop = agent.build_loop()
    assistant: AssistantMessage = loop.run(task)
    return PassResult(
        label=label,
        text=assistant.text,
        turn_count=int(getattr(loop, "turn_count", 0)),
    )


def run_multi_pass(
    *,
    files: Sequence[str],
    per_file_agent_factory: Callable[..., BaseAgent],
    integration_agent: BaseAgent,
    per_file_prompt_template: str = "Analyse the following file and report findings:\n\n{file}",
    integration_prompt_builder: Callable[[tuple[PassResult, ...]], str] | None = None,
) -> MultiPassResult:
    """Run the per-file pass + integration pass in sequence.

    ``per_file_agent_factory`` is called once per file with the
    file path; it MUST return a fresh :class:`BaseAgent` so the
    per-file passes don't share conversation state. The integration
    agent is a single :class:`BaseAgent` used exactly once at the
    end, with all per-file findings in its prompt.
    """
    if not files:
        raise ValueError("files must be a non-empty sequence")
    if not callable(per_file_agent_factory):
        raise TypeError("per_file_agent_factory must be callable")
    if not isinstance(integration_agent, BaseAgent):
        raise TypeError("integration_agent must be a BaseAgent")

    per_file_results: list[PassResult] = []
    for path in files:
        agent = per_file_agent_factory(path)
        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"per_file_agent_factory must return a BaseAgent, got {type(agent).__name__}"
            )
        prompt = per_file_prompt_template.format(file=path)
        per_file_results.append(run_pass(agent, prompt, label=path))

    builder = integration_prompt_builder or _default_integration_prompt
    integration_prompt = builder(tuple(per_file_results))
    integration = run_pass(integration_agent, integration_prompt, label="integration")

    return MultiPassResult(per_file=tuple(per_file_results), integration=integration)


# -- internals ------------------------------------------------------------


def _default_integration_prompt(passes: tuple[PassResult, ...]) -> str:
    sections = "\n\n".join(
        f"### {p.label}\n{p.text or '(no findings)'}" for p in passes
    )
    return (
        "You are reviewing cross-file concerns across the following "
        "per-file findings. Identify integration issues: shared types, "
        "conflicting assumptions, missing tests, and risks.\n\n"
        f"{sections}"
    )


__all__ = ["MultiPassResult", "PassResult", "run_multi_pass", "run_pass"]
