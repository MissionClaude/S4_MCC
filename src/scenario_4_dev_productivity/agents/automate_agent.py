"""AutomateAgent — shell command executor with safety rails.

Runs tests, builds, formatters, and one-off scripts. The narrow
allowlist (Bash only) prevents the agent from editing files or reading
code in-loop — anything that needs editing goes through GenerateAgent
via the coordinator, anything that needs investigation goes through
ExploreAgent.

The system prompt carries the safety rails (no ``sudo``, no
``rm -rf`` against paths the agent didn't create, structured failure
reporting). This file only owns the wiring.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.models.tools import AgentConfig
from scenario_4_dev_productivity.prompts.automate import AUTOMATE_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class AutomateAgent(BaseAgent):
    """Shell command executor — Bash only.

    The narrow scope is the safety perimeter: the agent can run commands
    but cannot edit files or read code in-loop. If the agent needs to
    edit a file, it returns a structured failure and the coordinator
    dispatches GenerateAgent.
    """

    DESCRIPTION: str = (
        "Runs shell commands with safety rails. Use for tests, builds, "
        "formatters, and one-off scripts. Has Bash only — no editing, no "
        "codebase exploration."
    )

    DEFAULT_TOOLS: tuple[str, ...] = ("Bash",)

    def __init__(
        self,
        registry: ToolRegistry,
        client: AnthropicClient,
        *,
        model: str | None = None,
        max_turns: int = 15,
    ) -> None:
        agent_config = AgentConfig(
            name="automate",
            description=self.DESCRIPTION,
            system_prompt=AUTOMATE_SYSTEM_PROMPT,
            allowed_tools=list(self.DEFAULT_TOOLS),
            model=model or config.automate_model,
        )
        super().__init__(
            config=agent_config,
            registry=registry,
            client=client,
            max_turns=max_turns,
        )


__all__ = ["AutomateAgent"]
