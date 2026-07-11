"""ExploreAgent — read-only codebase investigator.

Inherits the declarative wrapper from :class:`BaseAgent` and pins the
three read-only tools (Read, Grep, Glob). The agent never has Write or
Bash; the tool allowlist is the perimeter.

The constructor is intentionally cheap: it builds an
:class:`AgentConfig` from the prompt + allowlist + model and delegates
to ``super().__init__``. The hub-and-spoke topology means callers only
need the agent's :meth:`run`; the registry and client are shared with
the rest of the system.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.models.tools import AgentConfig
from scenario_4_dev_productivity.prompts.explore import EXPLORE_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class ExploreAgent(BaseAgent):
    """Read-only codebase investigator — Read, Grep, Glob only."""

    #: Display description surfaced in the Coordinator's Task tool list.
    DESCRIPTION: str = (
        "Read-only codebase investigator. Use for understanding structure, "
        "finding files, tracing dependencies, and summarising modules. "
        "Has Read, Grep, and Glob — no Write, no Bash."
    )

    #: Tools every Explore agent gets, regardless of caller. The
    #: allowlist is intentionally narrow: the agent cannot mutate the
    #: filesystem or run commands, only observe it.
    DEFAULT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob")

    def __init__(
        self,
        registry: ToolRegistry,
        client: AnthropicClient,
        *,
        model: str | None = None,
        max_turns: int = 15,
    ) -> None:
        agent_config = AgentConfig(
            name="explore",
            description=self.DESCRIPTION,
            system_prompt=EXPLORE_SYSTEM_PROMPT,
            allowed_tools=list(self.DEFAULT_TOOLS),
            model=model or config.explore_model,
        )
        super().__init__(
            config=agent_config,
            registry=registry,
            client=client,
            max_turns=max_turns,
        )


__all__ = ["ExploreAgent"]
