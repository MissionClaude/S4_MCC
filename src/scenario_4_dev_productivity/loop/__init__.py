"""Agentic loop package.

Exposes :class:`AgenticLoop` — the stop_reason-driven turn cycle that
drives every agent in the system. The loop knows nothing about
specific agents; it composes an :class:`AnthropicClient` and a
:class:`ToolRegistry` to turn user tasks into final assistant
messages.
"""

from __future__ import annotations

from scenario_4_dev_productivity.loop.engine import AgenticLoop, MaxTurnsExceeded

__all__ = ["AgenticLoop", "MaxTurnsExceeded"]
