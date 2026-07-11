"""GenerateAgent — code writer with Write-only scope.

Writes boilerplate, snippets, and small files. The tool allowlist is
narrow by design: the model is told to read the project style from the
*prompt* and the *coordinator's context*, not from in-loop exploration.

The spec note that contradicts the narrow allowlist is the system
prompt's "Read before you write" guidance — when the coordinator's
prompt already includes the file contents, Read is unnecessary. This is
deliberate: it forces the coordinator to pass all context explicitly,
matching the "subagents receive no parent context" rule.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents.base import BaseAgent
from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.config import config
from scenario_4_dev_productivity.models.tools import AgentConfig
from scenario_4_dev_productivity.prompts.generate import GENERATE_SYSTEM_PROMPT
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class GenerateAgent(BaseAgent):
    """Code writer — Write only.

    The narrow allowlist enforces context isolation: the agent can only
    write the artifact the coordinator asked for, never explore the
    codebase or run commands. If the coordinator needs the agent to read
    a file first, it must include the contents in the Task prompt.
    """

    DESCRIPTION: str = (
        "Writes boilerplate, snippets, and small files. Use for scaffolding, "
        "fixing patterns, and refactoring. Has Write only — no exploration, no "
        "shell access."
    )

    DEFAULT_TOOLS: tuple[str, ...] = ("Write",)

    def __init__(
        self,
        registry: ToolRegistry,
        client: AnthropicClient,
        *,
        model: str | None = None,
        max_turns: int = 15,
    ) -> None:
        agent_config = AgentConfig(
            name="generate",
            description=self.DESCRIPTION,
            system_prompt=GENERATE_SYSTEM_PROMPT,
            allowed_tools=list(self.DEFAULT_TOOLS),
            model=model or config.generate_model,
        )
        super().__init__(
            config=agent_config,
            registry=registry,
            client=client,
            max_turns=max_turns,
        )


__all__ = ["GenerateAgent"]
