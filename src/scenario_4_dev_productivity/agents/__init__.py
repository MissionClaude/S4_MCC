"""Agent package — the hub-and-spoke primitives.

PR #2 shipped the :class:`BaseAgent` (declarative bundle) and the
:func:`build_agent` factory. PR #3 adds the four concrete agents
(Coordinator, Explore, Generate, Automate) and the :class:`TaskTool`
that the Coordinator uses to spawn subagents.

The concrete agents are thin subclasses of :class:`BaseAgent`: they
pin a tool allowlist, a model tier, and a system prompt. All the real
work happens in the agentic loop the base class composes.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents.automate_agent import AutomateAgent
from scenario_4_dev_productivity.agents.base import BaseAgent, build_agent
from scenario_4_dev_productivity.agents.coordinator_agent import (
    SUBAGENT_TYPES,
    CoordinatorAgent,
    TaskTool,
    _default_subagent_factory,
)
from scenario_4_dev_productivity.agents.explore_agent import ExploreAgent
from scenario_4_dev_productivity.agents.generate_agent import GenerateAgent

__all__ = [
    "AutomateAgent",
    "BaseAgent",
    "CoordinatorAgent",
    "ExploreAgent",
    "GenerateAgent",
    "SUBAGENT_TYPES",
    "TaskTool",
    "build_agent",
    # Exposed for tests that want to inject a custom factory.
    "_default_subagent_factory",
]
