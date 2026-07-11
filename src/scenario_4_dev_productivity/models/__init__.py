"""Core data models for the developer productivity agent.

Re-exports the public model surface so callers can do::

    from scenario_4_dev_productivity.models import ToolDefinition, ToolResult

This package is intentionally small: it contains plain Pydantic types that
the rest of the system (api, tools, loop, agents) composes into a working
agentic loop. No I/O, no side effects, no Anthropic SDK imports here.
"""

from scenario_4_dev_productivity.models.api import (
    APIError,
    APIRequest,
    APIResponse,
    AuthError,
    ErrorCategory,
    RateLimitError,
)
from scenario_4_dev_productivity.models.messages import (
    AssistantContent,
    AssistantMessage,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import (
    AgentConfig,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)

__all__ = [
    # tools
    "AgentConfig",
    "ToolCall",
    "ToolDefinition",
    "ToolParameterSchema",
    "ToolResult",
    # messages
    "AssistantContent",
    "AssistantMessage",
    "Message",
    "MessageRole",
    "StopReason",
    "TextBlock",
    "ToolResultMessage",
    "ToolUseBlock",
    "UserMessage",
    # api
    "APIError",
    "APIRequest",
    "APIResponse",
    "AuthError",
    "ErrorCategory",
    "RateLimitError",
]
