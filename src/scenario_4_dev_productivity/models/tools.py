"""Tool-related Pydantic models.

Three building blocks:

* :class:`ToolDefinition` — the static schema exposed to the model so it can
  decide when to call a tool. The ``parameters`` field is a JSON Schema
  describing the tool's input shape.
* :class:`ToolCall` — a single tool-use request emitted by the assistant.
  Maps to Anthropic's ``tool_use`` content block.
* :class:`ToolResult` — the structured outcome of executing a tool, with the
  three fields the agentic loop needs to decide what to do next: success vs
  failure, the error category, and whether retrying makes sense.

An :class:`AgentConfig` ties a tool allowlist to a system prompt and
description — the hub-and-spoke pattern from the design.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolParameterSchema(BaseModel):
    """JSON Schema for a tool's input parameters.

    Wraps the raw schema dict so we can validate that callers always
    pass something shaped like a real JSON Schema (type=object, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="object", description="JSON Schema root type")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Property name -> JSON Schema for that property",
    )
    required: list[str] = Field(
        default_factory=list,
        description="Property names that must be present in the input",
    )

    def to_json_schema(self) -> dict[str, Any]:
        """Render the schema as a plain dict the Anthropic SDK accepts.

        Empty ``properties`` and ``required`` are dropped so the wire
        format is minimal: a tool with no declared inputs becomes
        ``{"type": "object"}`` instead of a noisy dict.
        """
        rendered: dict[str, Any] = {"type": self.type}
        if self.properties:
            rendered["properties"] = self.properties
        if self.required:
            rendered["required"] = self.required
        return rendered


class ToolDefinition(BaseModel):
    """A tool the model is allowed to call.

    Mirrors the shape Anthropic's ``tools`` array expects, so this object
    can be passed straight to ``client.messages.create``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Unique tool identifier")
    description: str = Field(
        min_length=1,
        description="Natural-language description shown to the model; "
        "should include an input example and boundary conditions",
    )
    parameters: ToolParameterSchema = Field(
        default_factory=ToolParameterSchema,
        description="JSON Schema for the tool's input",
    )

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Render as the dict shape the Anthropic SDK accepts."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.to_json_schema(),
        }


class ToolCall(BaseModel):
    """A single tool-use block emitted by the assistant.

    The ``id`` is what the API expects back in the corresponding
    :class:`ToolResult` so the model can correlate results with calls.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Tool-use ID from the API")
    name: str = Field(min_length=1, description="Name of the tool to call")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments matching the tool's JSON Schema",
    )


class ToolResult(BaseModel):
    """Structured outcome of executing a tool.

    The agentic loop branches on ``is_error``: a successful result is
    appended to the conversation as-is, while an error result is rendered
    so the model can decide whether to retry, change approach, or give up.
    ``is_retryable`` lets the loop itself re-invoke the tool when the
    failure is transient (e.g. a flaky network call).
    """

    model_config = ConfigDict(extra="forbid")

    tool_use_id: str = Field(
        min_length=1,
        description="ID of the ToolCall this result answers",
    )
    content: str = Field(
        default="",
        description="Tool output — plain text or a JSON-serialised string",
    )
    is_error: bool = Field(
        default=False,
        description="True when the tool failed and the model should adapt",
    )
    error_category: str | None = Field(
        default=None,
        description=(
            "One of 'transient', 'validation', 'permission' — describes "
            "WHY the tool failed so the loop can pick the right recovery"
        ),
    )
    is_retryable: bool = Field(
        default=False,
        description="True when the loop may safely re-invoke the same tool call",
    )

    @classmethod
    def success(cls, tool_use_id: str, content: str) -> ToolResult:
        """Build a successful result with the defaults that imply 'all good'."""
        return cls(tool_use_id=tool_use_id, content=content)

    @classmethod
    def failure(
        cls,
        tool_use_id: str,
        message: str,
        *,
        category: str,
        retryable: bool,
    ) -> ToolResult:
        """Build a structured failure with the right retry signals."""
        return cls(
            tool_use_id=tool_use_id,
            content=message,
            is_error=True,
            error_category=category,
            is_retryable=retryable,
        )


class AgentConfig(BaseModel):
    """Definition of one agent in the hub-and-spoke topology.

    Each agent has a fixed role (encoded in its system prompt) and a
    scoped tool allowlist. The Coordinator's allowlist always includes
    ``Task`` so it can spawn subagents; subagents get a narrower set.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Agent identifier")
    description: str = Field(
        min_length=1,
        description="What this agent does — surfaces in the Task tool list",
    )
    system_prompt: str = Field(
        min_length=1,
        description="System prompt establishing role, scope, and rules",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Names of tools this agent may call",
    )
    model: str = Field(
        default="claude-haiku",
        description="Anthropic model ID to use for this agent",
    )
