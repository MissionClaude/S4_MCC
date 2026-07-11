r"""Agentic loop engine — the stop_reason-driven turn cycle.

The loop is the heart of the agent:

1. Send the conversation + tools to the model.
2. Read ``stop_reason`` on the response.
3. ``TOOL_USE`` → execute each tool call, append a
   :class:`ToolResultMessage` to the conversation, loop.
4. ``END_TURN`` → return the final :class:`AssistantMessage`.

The loop NEVER inspects assistant text for completion markers. That is
the canonical exam-guide anti-pattern: text can lie, the API's
``stop_reason`` cannot. If a turn comes back with ``end_turn``, the
loop stops — period.

Failure handling:

* Errors from the API client bubble up to the caller. Auth errors are
  not retryable, so the loop won't even try.
* Tool failures (``is_error=True``) are fed back to the model as
  ordinary tool results. The model gets to decide what to do next —
  retry, change approach, or give up.
* ``max_conversation_turns`` is a safety belt against infinite loops
  when the model keeps calling tools. When the cap is hit, the loop
  raises :class:`MaxTurnsExceeded` and the caller decides whether to
  restart or surface the situation.
"""

from __future__ import annotations

from scenario_4_dev_productivity.api.client import AnthropicClient
from scenario_4_dev_productivity.models.api import APIRequest
from scenario_4_dev_productivity.models.messages import (
    AssistantMessage,
    Message,
    StopReason,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
)
from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolResult
from scenario_4_dev_productivity.tools.registry import ToolRegistry


class MaxTurnsExceeded(RuntimeError):
    """Raised when the agentic loop hits the configured turn cap.

    The partial conversation is preserved on ``self.turns`` so callers
    can inspect what the model was doing before bailing out.
    """

    def __init__(self, max_turns: int, last_response: AssistantMessage | None) -> None:
        super().__init__(f"Agentic loop exceeded {max_turns} turns")
        self.max_turns = max_turns
        self.last_response = last_response


class AgenticLoop:
    r"""Drive a stop_reason-based agentic loop.

    The loop is a small state machine over a list of :class:`Message`\s.
    Each iteration: render the current conversation as an
    :class:`APIRequest`, call the client, dispatch on
    ``response.stop_reason``.
    """

    def __init__(
        self,
        client: AnthropicClient,
        registry: ToolRegistry,
        *,
        max_turns: int = 15,
        model: str = "claude-haiku-4-5",
        system_prompt: str = "",
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")

        self._client = client
        self._registry = registry
        self._max_turns = max_turns
        self._model = model
        self._system_prompt = system_prompt

    # -- properties exposed for tests / introspection --------------------

    @property
    def turn_count(self) -> int:
        """Number of API calls made in the most recent :meth:`run`."""
        return self._turn_count

    @property
    def messages(self) -> tuple[Message | ToolResultMessage, ...]:
        """Snapshot of the most recent conversation history."""
        return tuple(self._messages)

    # -- public API ------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        tools: list[ToolDefinition] | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> AssistantMessage:
        """Execute the agentic loop on a task and return the final response.

        ``tools`` defaults to every tool the registry knows about. Pass
        a narrower list when running a subagent (e.g. an ExploreAgent
        with only Read/Grep/Glob).
        """
        if not task:
            raise ValueError("task must be a non-empty string")

        cap = max_turns if max_turns is not None else self._max_turns
        if cap <= 0:
            raise ValueError("max_turns must be > 0")

        active_tools = tools if tools is not None else self._registry.definitions()
        active_system = system_prompt if system_prompt is not None else self._system_prompt
        # The wire-level type is ``UserMessage | AssistantMessage``; we
        # also hold ``ToolResultMessage`` between turns for the
        # ``TOOL_USE`` → next request transition. We convert to wire
        # form (UserMessage) at request-build time.
        self._messages: list[Message | ToolResultMessage] = [UserMessage(content=task)]
        self._turn_count = 0

        while self._turn_count < cap:
            wire_messages: list[Message] = [
                m.to_user_message() if isinstance(m, ToolResultMessage) else m
                for m in self._messages
            ]
            request = APIRequest(
                model=self._model,
                system=active_system,
                messages=wire_messages,
                tools=active_tools,
            )
            response = self._client.send(request)
            self._turn_count += 1
            assistant = response.message
            self._messages.append(assistant)

            if assistant.stop_reason is StopReason.END_TURN:
                return assistant
            if assistant.stop_reason is StopReason.TOOL_USE:
                tool_results = self._execute_tool_calls(assistant)
                self._messages.append(ToolResultMessage(results=tool_results))
                continue
            # Anything else (None, unexpected value) — treat as terminal
            # and return what we have. The loop never panics on
            # surprise stop reasons; it just stops.
            return assistant

        last = self._messages[-1] if self._messages else None
        raise MaxTurnsExceeded(
            max_turns=cap,
            last_response=last if isinstance(last, AssistantMessage) else None,
        )

    # -- internal --------------------------------------------------------

    def _execute_tool_calls(self, assistant: AssistantMessage) -> list[ToolResult]:
        """Run every tool call the assistant emitted and collect results.

        One assistant turn can request several tools in parallel. We
        execute them in order — the registry is in-process, so the
        parallelism is mostly about not blocking on network. Either way
        we always return a result for every call so the model never
        sees a "missing tool result" error.
        """
        results: list[ToolResult] = []
        for block in assistant.content:
            if isinstance(block, ToolUseBlock):
                call = block.call
                result = self._registry.execute(call.id, call.name, call.input)
                results.append(result)
        return results or [self._no_tool_calls_placeholder()]

    def _no_tool_calls_placeholder(self) -> ToolResult:
        """Defensive fallback when ``stop_reason`` says ``tool_use`` but no blocks were found."""
        return ToolResult.failure(
            tool_use_id="synthetic-no-calls",
            message="Model returned stop_reason=tool_use with no tool_use blocks.",
            category="validation",
            retryable=False,
        )


__all__ = ["AgenticLoop", "MaxTurnsExceeded"]
