"""PipelineRunner — non-interactive, single-task agentic execution.

Pipeline mode is the ``claude -p "task"`` equivalent for this SDK.
The runner:

* creates a **fresh** :class:`BaseAgent` for each call so two
  consecutive ``run(...)`` invocations cannot share conversation
  state (the spec calls this out explicitly);
* drives the agentic loop on the task and captures the final
  assistant text;
* wraps the result in a :class:`PipelineResult` that renders as
  either plain text or JSON (``--output-format json``).

The runner is the one place tests should hit when verifying
"non-interactive execution". Unit tests for the loop live in
``tests/test_loop``; integration tests live in
``tests/test_integration/test_pipeline.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.messages import AssistantMessage
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class OutputFormat(StrEnum):
    """Output rendering for a pipeline result.

    * ``text`` — the final assistant text, nothing else.
    * ``json`` — a structured dict with the task, the final text,
      turn count, and any pipeline-level metadata.
    """

    TEXT = "text"
    JSON = "json"


# Convenience singletons — kept lowercase to match the dataclass-style
# factory convention used elsewhere in the package.
TextOutputFormat = OutputFormat.TEXT
JSONOutputFormat = OutputFormat.JSON


@runtime_checkable
class AgentFactory(Protocol):
    """Build a fresh :class:`BaseAgent` for one pipeline run."""

    def __call__(self) -> BaseAgent:
        ...


@dataclass(frozen=True)
class PipelineResult:
    """The outcome of a single pipeline run.

    ``text`` is the final assistant message's text — the same string
    the user would see at the terminal. ``turn_count`` is the number
    of API calls the loop made; ``metadata`` is an open dict for
    callers that want to attach their own fields (CI job id, branch,
    etc.) before serialising to JSON.
    """

    task: str
    text: str
    stop_reason: str | None
    turn_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, fmt: OutputFormat) -> str:
        """Render this result in the requested output format."""
        if fmt is OutputFormat.TEXT:
            return self.text
        return json.dumps(self.to_json_dict(), indent=2, ensure_ascii=False)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for callers that want to post-process."""
        return {
            "task": self.task,
            "text": self.text,
            "stop_reason": self.stop_reason,
            "turn_count": self.turn_count,
            "metadata": dict(self.metadata),
        }


class PipelineRunner:
    """Run a single task in pipeline mode.

    Construction is cheap: hold a :class:`AnthropicClient` and a
    factory that builds the right :class:`BaseAgent` for the job. The
    factory is called once per :meth:`run` so each run gets a fresh
    agent — session isolation is enforced by construction, not by
    convention.
    """

    def __init__(
        self,
        client: AnthropicClient,
        agent_factory: AgentFactory,
        *,
        default_output: OutputFormat = OutputFormat.TEXT,
    ) -> None:
        if not isinstance(client, AnthropicClient):
            raise TypeError("client must be an AnthropicClient")
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self._client = client
        self._factory = agent_factory
        self._default_output = default_output

    # -- public API -------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        output: OutputFormat | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run a task end-to-end and return a :class:`PipelineResult`.

        ``output`` overrides the runner's default. ``metadata`` is
        attached to the result for downstream JSON consumers.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        # Fresh agent per run — the spec's session isolation rule.
        agent = self._factory()
        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"agent_factory must return a BaseAgent, got {type(agent).__name__}"
            )
        # Build the loop directly so we can read ``turn_count`` after
        # the run. ``BaseAgent.run`` would discard the loop, hiding
        # the metadata pipeline mode needs to report.
        loop = agent.build_loop()
        assistant: AssistantMessage = loop.run(task)
        return PipelineResult(
            task=task,
            text=assistant.text,
            stop_reason=assistant.stop_reason.value if assistant.stop_reason else None,
            turn_count=int(getattr(loop, "turn_count", 0)),
            metadata=dict(metadata or {}),
        )

    def run_and_render(
        self,
        task: str,
        *,
        output: OutputFormat | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Run the task and return the rendered output as a string."""
        fmt = output or self._default_output
        return self.run(task, output=fmt, metadata=metadata).render(fmt)


# -- helpers --------------------------------------------------------------


def make_agent_factory(
    agent_class: type[BaseAgent],
    registry: ToolRegistry,
    client: AnthropicClient,
    **kwargs: Any,
) -> AgentFactory:
    """Build an :class:`AgentFactory` that produces fresh ``agent_class`` instances.

    Convenience for callers (and the CLI) that want a runner without
    subclassing :class:`PipelineRunner`. Each call instantiates
    ``agent_class(registry=..., client=..., **kwargs)``.

    The returned callable accepts an optional ``path`` argument so it
    plugs into :func:`run_multi_pass` as a per-file factory. The path
    is currently ignored — per-file prompts come from the multi-pass
    helper's template — but accepting the argument keeps the API
    uniform.
    """

    def _factory(path: str | None = None) -> BaseAgent:  # noqa: ARG001
        return agent_class(registry=registry, client=client, **kwargs)

    return _factory


__all__ = [
    "AgentFactory",
    "JSONOutputFormat",
    "OutputFormat",
    "PipelineResult",
    "PipelineRunner",
    "TextOutputFormat",
    "make_agent_factory",
]
