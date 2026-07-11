"""Tool protocol and registry for the agentic loop.

The :class:`ToolRegistry` is the dispatch table the agentic loop uses to
turn a model-emitted :class:`ToolCall` into a :class:`ToolResult`. It
also projects the registered tools' :class:`ToolDefinition`s to the wire
format the Anthropic API expects, so the loop has one place to ask
"what tools does the model see?".

Design notes:

* Registration is by ``name`` — the same string the model uses. Names
  are unique; a duplicate ``register`` raises :class:`ValueError` so a
  typo never silently shadows a real tool.
* Tools are plain objects exposing a ``definition`` and an ``execute``.
  No magic decorators, no ``functools.wraps`` of arbitrary callables.
  Keeping the protocol tiny makes tests easy to write.
* :meth:`ToolRegistry.execute` returns a :class:`ToolResult` and never
  raises. Unknown tools, validation errors, and exceptions from the
  tool implementation are all rendered as a structured
  ``ToolResult.failure(...)`` so the loop can feed them back to the
  model as a normal tool result instead of crashing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from scenario_4_dev_productivity.models.tools import ToolDefinition, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Executable half of a :class:`ToolDefinition`.

    A ``Tool`` exposes its model-facing declaration (``definition``) and
    a single ``execute`` method that takes the model's arguments and
    returns a :class:`ToolResult`. The method MUST NOT raise — it
    should catch its own exceptions and return a structured failure so
    the loop can keep iterating.
    """

    @property
    def name(self) -> str:
        """The tool's unique name (matches the :class:`ToolDefinition`)."""
        ...

    @property
    def definition(self) -> ToolDefinition:
        """The static declaration sent to the model."""
        ...

    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool and package the outcome for the loop.

        ``tool_use_id`` is what the API expects back in the
        :class:`ToolResult` so the model can correlate results with
        calls. ``arguments`` is the raw dict the model produced; the
        tool is responsible for validating it.
        """
        ...


class ToolRegistry:
    """Dispatch table of :class:`Tool` instances keyed by name.

    The registry is the single source of truth for "what tools does this
    agent have?". The agentic loop calls :meth:`execute` when the model
    wants a tool called and :meth:`definitions` to render the wire
    payload for the API.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- registration ----------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        :raises ValueError: when a tool with the same name is already
            registered. Typos are easy to make; failing loudly is better
            than silently shadowing a real tool.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op when the name is unknown."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name, or ``None`` if not registered."""
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Any:
        return iter(tuple(self._tools.values()))

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names in insertion order."""
        return tuple(self._tools.keys())

    # -- projection ------------------------------------------------------

    def definitions(self) -> list[ToolDefinition]:
        """Render all tools as :class:`ToolDefinition`s for the API payload.

        Insertion order is preserved so the order tools appear to the
        model is stable across calls — useful for both tests and the
        real API where a stable order helps the model pick the right
        tool.
        """
        return [tool.definition for tool in self._tools.values()]

    def definitions_for(self, allowed: list[str]) -> list[ToolDefinition]:
        """Project definitions for a specific agent's tool allowlist.

        Unknown names are silently skipped — the registry is not the
        right place to surface "agent misconfigured" errors. The agent
        layer is responsible for that. Here we just give the model the
        set of tools the agent is allowed to use.
        """
        return [self._tools[name].definition for name in allowed if name in self._tools]

    # -- dispatch --------------------------------------------------------

    def execute(self, tool_use_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool by name and return a structured :class:`ToolResult`.

        This method **never raises**. Any exception from the tool's
        ``execute`` is caught and rendered as a structured failure with
        category ``"transient"`` and ``is_retryable=False`` — the loop
        should not retry an arbitrary exception without knowing what it
        was, and the model deserves to see the message.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=(
                    f"Unknown tool {name!r}. Available tools: {', '.join(self.names) or '(none)'}."
                ),
                category="validation",
                retryable=False,
            )
        try:
            return tool.execute(tool_use_id, arguments)
        except Exception as exc:  # noqa: BLE001 — intentional safety net
            return ToolResult.failure(
                tool_use_id=tool_use_id,
                message=f"Tool {name!r} raised an unhandled exception: {exc}",
                category="transient",
                retryable=False,
            )
