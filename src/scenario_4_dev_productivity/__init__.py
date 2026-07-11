"""Developer Productivity Agent — Scenario 4: Claude Certified Architect.

Public API surface. Phase 1 (Foundation) exposes the data models and
agent prompt templates that later phases (core infra, agents, tests,
docs) will compose into a working agentic loop.

Usage::

    from scenario_4_dev_productivity import (
        AgentConfig,
        COORDINATOR_SYSTEM_PROMPT,
        ToolDefinition,
        ToolResult,
    )
"""

from __future__ import annotations

from scenario_4_dev_productivity.models import (
    AgentConfig,
    APIError,
    APIRequest,
    APIResponse,
    AssistantMessage,
    AuthError,
    ErrorCategory,
    MessageRole,
    RateLimitError,
    StopReason,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
)
from scenario_4_dev_productivity.prompts import (
    AUTOMATE_SYSTEM_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    EXPLORE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # agents / config
    "AgentConfig",
    # messages
    "AssistantMessage",
    "MessageRole",
    "StopReason",
    "TextBlock",
    "ToolResultMessage",
    "ToolUseBlock",
    "UserMessage",
    # tools
    "ToolCall",
    "ToolDefinition",
    "ToolParameterSchema",
    "ToolResult",
    # api
    "APIError",
    "APIRequest",
    "APIResponse",
    "AuthError",
    "ErrorCategory",
    "RateLimitError",
    # prompts
    "AUTOMATE_SYSTEM_PROMPT",
    "COORDINATOR_SYSTEM_PROMPT",
    "EXPLORE_SYSTEM_PROMPT",
    "GENERATE_SYSTEM_PROMPT",
]
