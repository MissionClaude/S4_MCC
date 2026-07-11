"""Developer Productivity Agent — Scenario 4: Claude Certified Architect.

Public API surface. Phases 1–2 expose the data models, the agent
prompt templates, and the core infrastructure (tools, API client,
agentic loop, agent base) that later phases (concrete agents, MCP,
pipeline, context management, tests, docs) will compose into a
working agentic loop.

Usage::

    from scenario_4_dev_productivity import (
        AgentConfig,
        AgenticLoop,
        AnthropicClient,
        COORDINATOR_SYSTEM_PROMPT,
        ToolDefinition,
        ToolRegistry,
        ToolResult,
    )
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents import BaseAgent, build_agent
from scenario_4_dev_productivity.api import AnthropicClient
from scenario_4_dev_productivity.loop import AgenticLoop, MaxTurnsExceeded
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
from scenario_4_dev_productivity.tools import (
    BUILTIN_TOOLS,
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    ToolRegistry,
    WriteTool,
    default_registry,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # agents
    "BaseAgent",
    "build_agent",
    # loop
    "AgenticLoop",
    "MaxTurnsExceeded",
    # api
    "AnthropicClient",
    "APIError",
    "APIRequest",
    "APIResponse",
    "AuthError",
    "ErrorCategory",
    "RateLimitError",
    # models — agents / config
    "AgentConfig",
    # models — messages
    "AssistantMessage",
    "MessageRole",
    "StopReason",
    "TextBlock",
    "ToolResultMessage",
    "ToolUseBlock",
    "UserMessage",
    # models — tools
    "ToolCall",
    "ToolDefinition",
    "ToolParameterSchema",
    "ToolResult",
    # prompts
    "AUTOMATE_SYSTEM_PROMPT",
    "COORDINATOR_SYSTEM_PROMPT",
    "EXPLORE_SYSTEM_PROMPT",
    "GENERATE_SYSTEM_PROMPT",
    # tools
    "BUILTIN_TOOLS",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "ToolRegistry",
    "WriteTool",
    "default_registry",
]
