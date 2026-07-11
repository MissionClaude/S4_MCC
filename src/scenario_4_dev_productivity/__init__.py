"""Developer Productivity Agent — Scenario 4: Claude Certified Architect.

Public API surface. Phases 1–3 expose the data models, the agent
prompt templates, the core infrastructure (tools, API client, agentic
loop, agent base), the four concrete agents (Coordinator, Explore,
Generate, Automate), and the MCP integration (loader, discovery).

Usage::

    from scenario_4_dev_productivity import (
        AgentConfig,
        AgenticLoop,
        AnthropicClient,
        AutomateAgent,
        COORDINATOR_SYSTEM_PROMPT,
        CoordinatorAgent,
        ExploreAgent,
        GenerateAgent,
        MCPConfig,
        MCPConfigLoader,
        MCPToolDiscovery,
        TaskTool,
        ToolDefinition,
        ToolRegistry,
        ToolResult,
    )
"""

from __future__ import annotations

from scenario_4_dev_productivity.agents import (
    SUBAGENT_TYPES,
    AutomateAgent,
    BaseAgent,
    CoordinatorAgent,
    ExploreAgent,
    GenerateAgent,
    TaskTool,
    build_agent,
)
from scenario_4_dev_productivity.api import AnthropicClient
from scenario_4_dev_productivity.loop import AgenticLoop, MaxTurnsExceeded
from scenario_4_dev_productivity.mcp import (
    MCPConfig,
    MCPConfigLoader,
    MCPServerConfig,
    MCPToolAdapter,
    MCPToolDiscovery,
    MCPToolSpec,
)
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

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # agents
    "AutomateAgent",
    "BaseAgent",
    "CoordinatorAgent",
    "ExploreAgent",
    "GenerateAgent",
    "SUBAGENT_TYPES",
    "TaskTool",
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
    # mcp
    "MCPConfig",
    "MCPConfigLoader",
    "MCPServerConfig",
    "MCPToolAdapter",
    "MCPToolDiscovery",
    "MCPToolSpec",
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
