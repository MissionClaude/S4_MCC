r"""Agent base class — the contract every concrete agent implements.

A :class:`BaseAgent` is the smallest possible "agent" object: a
declarative bundle of (system prompt, allowed tools, model). The
agentic loop is the runtime; the base class is the data.

Two things worth calling out:

* Tools are referenced by *name*. The base class doesn't own a
  :class:`ToolRegistry`; the loop passes the relevant
  :class:`ToolDefinition`\s to the model. This keeps the agent object
  trivially serialisable — useful for tests, and for the day we want
  to ship agent configs as JSON over MCP.
* The ``with_model`` / ``with_tool`` / ``with_prompt`` builders return
  a new agent so callers can tweak one aspect without mutating
  shared state. The fluent style matches the spec's "scoped
  configuration" guidance.
"""

from __future__ import annotations

from typing import Any

from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.loop.engine import AgenticLoop
from scenario_4_dev_productivity.models.messages import AssistantMessage
from scenario_4_dev_productivity.models.tools import AgentConfig
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class BaseAgent:
    """Declarative agent — a system prompt, a model, and a tool allowlist.

    Concrete agents (Coordinator, Explore, Generate, Automate) will
    extend this class in a later phase. For now the base class is
    enough to wire the loop to per-agent model selection and to drive
    subagent spawning with the right scoping.
    """

    def __init__(
        self,
        config: AgentConfig,
        registry: ToolRegistry,
        client: AnthropicClient,
        *,
        max_turns: int = 15,
    ) -> None:
        self._config = config
        self._registry = registry
        self._client = client
        self._max_turns = max_turns

    # -- read-only accessors --------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self._config.allowed_tools)

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def config(self) -> AgentConfig:
        """The underlying Pydantic :class:`AgentConfig`."""
        return self._config

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def client(self) -> AnthropicClient:
        return self._client

    # -- builders (return a new instance) -------------------------------

    def with_model(self, model: str) -> BaseAgent:
        """Return a copy of this agent using a different model."""
        return BaseAgent(
            config=self._config.model_copy(update={"model": model}),
            registry=self._registry,
            client=self._client,
            max_turns=self._max_turns,
        )

    def with_prompt(self, system_prompt: str) -> BaseAgent:
        """Return a copy of this agent with a different system prompt."""
        return BaseAgent(
            config=self._config.model_copy(update={"system_prompt": system_prompt}),
            registry=self._registry,
            client=self._client,
            max_turns=self._max_turns,
        )

    def with_tools(self, allowed_tools: list[str]) -> BaseAgent:
        """Return a copy of this agent with a different tool allowlist."""
        return BaseAgent(
            config=self._config.model_copy(update={"allowed_tools": list(allowed_tools)}),
            registry=self._registry,
            client=self._client,
            max_turns=self._max_turns,
        )

    def with_max_turns(self, max_turns: int) -> BaseAgent:
        """Return a copy of this agent with a different turn cap."""
        return BaseAgent(
            config=self._config,
            registry=self._registry,
            client=self._client,
            max_turns=max_turns,
        )

    # -- runtime ---------------------------------------------------------

    def build_loop(self) -> AgenticLoop:
        """Return a fresh :class:`AgenticLoop` scoped to this agent.

        Each call returns a new loop so agents can run concurrently or
        be retried without cross-contamination. The loop is configured
        with this agent's model, system prompt, and the subset of tools
        the agent is allowed to use.
        """
        return AgenticLoop(
            client=self._client,
            registry=self._registry,
            max_turns=self._max_turns,
            model=self._config.model,
            system_prompt=self._config.system_prompt,
        )

    def run(self, task: str, **overrides: Any) -> AssistantMessage:
        r"""Run the agent on a task and return the final assistant message.

        ``overrides`` may contain ``tools`` (a narrowed list of
        :class:`ToolDefinition`\s), ``system_prompt``, and
        ``max_turns`` — useful for one-off invocations without
        constructing a new agent.
        """
        loop = self.build_loop()
        return loop.run(
            task,
            tools=overrides.get("tools"),
            system_prompt=overrides.get("system_prompt"),
            max_turns=overrides.get("max_turns"),
        )

    # -- dunder ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, "
            f"tools={list(self.allowed_tools)!r})"
        )


def build_agent(
    config: AgentConfig,
    registry: ToolRegistry,
    client: AnthropicClient,
    **kwargs: Any,
) -> BaseAgent:
    """Factory: build a :class:`BaseAgent` from the shared primitives.

    Centralised so concrete agents (Coordinator, Explore, Generate,
    Automate) can be created with the same wiring in tests and
    production. Returns the base class — subclassing is a Phase 3
    concern.
    """
    return BaseAgent(config=config, registry=registry, client=client, **kwargs)


__all__ = ["BaseAgent", "build_agent"]
