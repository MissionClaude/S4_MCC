"""Anthropic API client package.

Exposes :class:`AnthropicClient` — the only entry point the agentic
loop uses to talk to the model. The wrapper translates between the
domain types in :mod:`scenario_4_dev_productivity.models` and the
underlying SDK, applies retry/backoff, and maps SDK errors onto the
structured :class:`APIError` hierarchy.
"""

from __future__ import annotations

from scenario_4_dev_productivity.api.client import AnthropicClient

__all__ = ["AnthropicClient"]
