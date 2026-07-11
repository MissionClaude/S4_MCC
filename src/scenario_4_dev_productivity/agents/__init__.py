"""Agent package — the hub-and-spoke primitives.

Phase 2 ships the :class:`BaseAgent` (declarative bundle) and the
:func:`build_agent` factory. Phase 3 will add the four concrete agents
(Coordinator, Explore, Generate, Automate) and the Task tool that
spawns them.
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents.base import BaseAgent, build_agent

__all__ = ["BaseAgent", "build_agent"]
