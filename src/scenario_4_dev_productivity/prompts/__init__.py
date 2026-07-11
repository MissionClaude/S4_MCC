"""System prompt templates for each agent role.

Each module exposes a ``SYSTEM_PROMPT`` constant — a string the
:class:`~scenario_4_dev_productivity.models.AgentConfig` can drop in
verbatim. Splitting per role keeps prompts short, reviewable, and
independently editable.

The prompts encode the four invariants the spec cares about:

* Coordinator never does exploration or generation itself; it decomposes
  and delegates via the Task tool.
* Subagents receive all context in their initial prompt — they do not
  inherit the coordinator's conversation history.
* The agentic loop terminates on ``stop_reason == end_turn`` only.
* Tools are scoped per role so a GenerateAgent can't call Bash, etc.
"""

from scenario_4_dev_productivity.prompts.automate import AUTOMATE_SYSTEM_PROMPT
from scenario_4_dev_productivity.prompts.coordinator import COORDINATOR_SYSTEM_PROMPT
from scenario_4_dev_productivity.prompts.explore import EXPLORE_SYSTEM_PROMPT
from scenario_4_dev_productivity.prompts.generate import GENERATE_SYSTEM_PROMPT

__all__ = [
    "AUTOMATE_SYSTEM_PROMPT",
    "COORDINATOR_SYSTEM_PROMPT",
    "EXPLORE_SYSTEM_PROMPT",
    "GENERATE_SYSTEM_PROMPT",
]
