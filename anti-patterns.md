# Anti-Patterns — Developer Productivity Agent

Ten mistakes the exam guide calls out, the failure mode each one causes, and the
pattern this codebase uses instead. Each entry shows a **WRONG** example,
explains **WHY it fails**, and shows the **CORRECT** pattern actually implemented
in this repo.

> Every code snippet in the "CORRECT" column is taken (or adapted) from the
> actual implementation. File paths are noted so you can read the real version.

---

## 1. Pipeline hang — running `claude` without `-p`

A CI job calls the SDK in interactive mode. The process sits forever waiting
for stdin because no `-p`/`--print` flag tells it the prompt is on the command
line.

**WRONG** — assumes the runner is interactive:

```bash
# In a CI step
claude "Summarise the test failures"
# ↑ Waits for interactive input that will never come. Job times out.
```

**WHY it fails.** Claude Code's default mode is interactive REPL. In a
non-interactive environment (`!tty`), the SDK cannot fall back to a prompt and
blocks on `read(stdin)`. The exit code is `0` only when the user types
`/exit` — never.

**CORRECT** — pipeline mode is non-interactive by construction:

```bash
# `claude -p` is the documented non-interactive mode.
# This project's analogue:
uv run scenario-4 run "Summarise the test failures" --output-format json
# Always reads the prompt from argv, never blocks on stdin, exits 0/1.
```

The corresponding `PipelineRunner.run` (in `src/scenario_4_dev_productivity/pipeline/runner.py`):

```python
def run(self, task: str, *, output: OutputFormat | None = None, ...) -> PipelineResult:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    # Fresh agent per run — session isolation enforced by construction.
    agent = self._factory()
    loop = agent.build_loop()
    assistant: AssistantMessage = loop.run(task)
    return PipelineResult(task=task, text=assistant.text, ...)
```

There is no path by which `PipelineRunner` can read from stdin — `task` is
positional, validated, and never `None`.

---

## 2. Text-parsing stop signals

Looking for the string `"DONE"` (or `"###TASK_COMPLETE###"`, or any other
token) in the assistant's text to decide when the loop is finished.

**WRONG**:

```python
def run(self, task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(model="claude-haiku", messages=messages, ...)
        text = response.content[0].text
        messages.append({"role": "assistant", "content": text})
        if "DONE" in text.upper():
            return text
        # Ask the model to continue...
        messages.append({"role": "user", "content": "continue"})
```

**WHY it fails.** Text can lie. A model can say `"Looks DONE to me"` mid-task
or `"I'm not quite DONE"` as part of its reasoning. The model has no contract
to emit any particular token — only the API's `stop_reason` is a real signal
that the assistant is finished.

**CORRECT** — the loop branches on `stop_reason`, not on text:

```python
# src/scenario_4_dev_productivity/loop/engine.py
while self._turn_count < cap:
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
    return assistant  # anything else is terminal
```

The loop **never** inspects `assistant.text`. If the API says
`stop_reason == "end_turn"`, the loop stops — period.

---

## 3. Thin tool descriptions

A tool is registered with `description="search code"` and the model has to
guess what input shape to use, what counts as a match, and where the boundary
sits.

**WRONG**:

```python
@property
def definition(self) -> ToolDefinition:
    return ToolDefinition(
        name="Grep",
        description="search code",  # ←
        parameters=ToolParameterSchema(
            properties={"pattern": {"type": "string"}},
            required=["pattern"],
        ),
    )
```

**WHY it fails.** The model sees ~7–8 tool names per call. With vague
descriptions it picks the wrong one, calls it with the wrong arguments, or
avoids it entirely in favour of a built-in it already knows. The
"description quality for adoption" guidance in the exam guide is exactly
this: agents prefer built-ins when MCP tools are vague.

**CORRECT** — every tool description includes an input example and boundary
conditions. From `src/scenario_4_dev_productivity/tools/bash_tool.py`:

```python
@property
def definition(self) -> ToolDefinition:
    return ToolDefinition(
        name="Bash",
        description=(
            "Run a shell command and return its combined output. The "
            "command is split with shlex so each token is a separate "
            "argv element; pipes, redirects, and globbing do NOT work "
            "(use sh -c '...' via Bash only if you really need them).\n\n"
            "Use this for tests, builds, formatting, one-off scripts, "
            "and any side-effecting operation. For reading files, use "
            "Read. For searching contents, use Grep. For finding files, "
            "use Glob.\n\n"
            "Input examples:\n"
            '  {"command": "pytest tests/ -x", "timeout": 60}\n'
            '  {"command": "git status --short"}\n'
            '  {"command": "uv run ruff check src/"}\n\n'
            "Boundary conditions:\n"
            "- Non-zero exit code is reported as a failure, not a success.\n"
            "- Commands exceeding the timeout are killed and reported as a "
            "transient, retryable failure."
        ),
        parameters=ToolParameterSchema(...),
    )
```

Same pattern in `TaskTool.definition` (`coordinator_agent.py`) and every
other tool in `src/scenario_4_dev_productivity/tools/`.

---

## 4. Missing `isError` flag on tool failures

A tool returns a regular text result containing the word `"error"` and hopes
the model notices. The loop has no structured signal that the call failed.

**WRONG**:

```python
def execute(self, tool_use_id: str, arguments: dict) -> ToolResult:
    path = arguments.get("path")
    if not path:
        return ToolResult(tool_use_id=tool_use_id, content="Error: path is required")
    # ... or worse:
    return ToolResult(tool_use_id=tool_use_id, content="")
```

**WHY it fails.** The agentic loop appends the result to the conversation
exactly the same way as a success. The model has no programmatic signal that
the call failed, no error category, and no retry hint. It might retry, it
might not, and the loop has no way to know which.

**CORRECT** — every failure returns a structured `ToolResult` with category
and retry signal. From `bash_tool.py`:

```python
except subprocess.TimeoutExpired:
    return ToolResult.failure(
        tool_use_id=tool_use_id,
        message=(
            f"Command {argv!r} exceeded the {timeout:.0f}s timeout and "
            "was killed. Increase the timeout or narrow the command."
        ),
        category="transient",
        retryable=True,
    )
```

The factory in `models/tools.py` sets all three fields at once:

```python
@classmethod
def failure(cls, tool_use_id: str, message: str, *, category: str, retryable: bool) -> ToolResult:
    return cls(
        tool_use_id=tool_use_id, content=message,
        is_error=True, error_category=category, is_retryable=retryable,
    )
```

Categories used in this codebase: `"transient"` (timeouts, network blips,
retryable), `"validation"` (bad input, unknown tool, non-retryable),
`"permission"` (denied, non-retryable). The loop and the model can branch on
these.

---

## 5. Context window exhaustion — no truncation of verbose tool output

A `Read` tool returns 5,000 lines of code. The next turn pushes the
conversation past the model's context window and the API rejects the request.

**WRONG**:

```python
def execute(self, tool_use_id: str, arguments: dict) -> ToolResult:
    content = path.read_text()
    return ToolResult(tool_use_id=tool_use_id, content=content)  # Always raw
```

**WHY it fails.** Verbose tool results eat tokens that the conversation
otherwise needs. A single 5,000-line `Read` is enough to push a 200k-token
window over the edge if the conversation has been running for a while.

**CORRECT** — `PostToolUse` hooks trim large results *before* they enter the
conversation, and `ContextCompactor` summarises history when the loop has
been running for many turns.

Trim hook (`src/scenario_4_dev_productivity/context/hooks.py`):

```python
class TrimReadOutputHook:
    def __init__(self, *, max_lines: int = 200, keep_lines: int = 50) -> None: ...
    def __call__(self, tool_name: str, tool_use_id: str, result: ToolResult) -> ToolResult:
        if tool_name != "Read" or result.is_error:
            return result
        lines = result.content.splitlines()
        if len(lines) <= self._max_lines:
            return result
        head = lines[: self._keep_lines]
        tail = lines[-self._keep_lines :]
        marker = f"[... {len(lines) - len(head) - len(tail)} lines truncated ...]"
        return result.model_copy(update={"content": "\n".join([*head, marker, *tail])})
```

`/compact` equivalent (`context/compact.py`):

```python
compactor = ContextCompactor(keep_lines=20, recent_turns=4)
trimmed = compactor.compact(messages)   # First user msg + last 4 turns kept; middle summarised
```

---

## 6. Subagent context leak — sharing the parent's conversation

A subagent is constructed by passing the coordinator's `messages` list to it.
The subagent sees every prior tool result and makes decisions on stale state.

**WRONG**:

```python
class TaskTool:
    def execute(self, tool_use_id, arguments):
        subagent = ExploreAgent(registry=..., client=...)
        # Inject the parent's history so the subagent has "context":
        subagent.messages = self._coordinator.messages
        return subagent.run(arguments["prompt"])
```

**WHY it fails.** Subagents do not inherit coordinator history. They see
irrelevant prior tool results, can ramble on about them, and can leak
secrets (tokens, paths) the coordinator was given in a private turn. It also
makes subagent behaviour non-deterministic — the same Task call gives
different results depending on what the coordinator did before.

**CORRECT** — explicit prompt passing, no history sharing. From
`coordinator_agent.py`:

```python
# The subagent constructor takes its own messages list starting from
# a single UserMessage with the prompt. There is no path by which the
# coordinator's history leaks into the subagent's conversation.
class TaskTool:
    def execute(self, tool_use_id: str, arguments: dict[str, Any]) -> ToolResult:
        subagent = self._factory(subagent_type)   # Fresh instance
        response = subagent.run(prompt)           # Prompt is the only input
        return ToolResult.success(tool_use_id=tool_use_id, content=response.text)
```

The `Task` tool's `description` makes this rule explicit to the model:

> "The subagent does NOT inherit your conversation history — the prompt
> you pass must contain all context the subagent needs (paths,
> constraints, prior findings)."

---

## 7. Tool overload — every tool available to every agent

The model sees all 6 tools for every agent. The Explore agent can run `Bash`
and the Generate agent can `Grep` the codebase.

**WRONG**:

```python
class ExploreAgent(BaseAgent):
    DEFAULT_TOOLS = ("Read", "Write", "Bash", "Grep", "Glob", "Edit")  # ←
```

**WHY it fails.** It violates the principle of least privilege. The model can
write files when it should only be reading, and run `rm -rf` when it should
only be investigating. It also makes the tool selection ambiguous — when
there are 6 tools with overlapping use cases, the model picks inconsistently.
The exam guide's "enhance MCP tool descriptions" guidance is the same idea:
narrow, well-described tools beat broad, vague ones.

**CORRECT** — each agent gets a narrow allowlist. The allowlist is metadata
the loop uses to project the tools the model sees.

| Agent | Allowlist |
|---|---|
| `ExploreAgent` | `("Read", "Grep", "Glob")` |
| `GenerateAgent` | `("Write",)` |
| `AutomateAgent` | `("Bash",)` |
| `CoordinatorAgent` | `("Task", "Read", "Write", "Edit", "Bash", "Grep", "Glob")` |

From `src/scenario_4_dev_productivity/agents/explore_agent.py`:

```python
class ExploreAgent(BaseAgent):
    """Read-only codebase investigator — Read, Grep, Glob only."""
    DEFAULT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob")
    def __init__(self, registry, client, *, model=None, max_turns=15):
        agent_config = AgentConfig(
            name="explore", description=self.DESCRIPTION,
            system_prompt=EXPLORE_SYSTEM_PROMPT,
            allowed_tools=list(self.DEFAULT_TOOLS),
            model=model or config.explore_model,
        )
        super().__init__(config=agent_config, registry=registry, client=client, max_turns=max_turns)
```

The Explore agent is structurally incapable of mutating the filesystem or
running shell commands — the allowlist is the perimeter.

---

## 8. Generic error messages — `"Error occurred"`

A tool returns `content="Error: something went wrong"`. The model has no
clue whether to retry, change approach, or give up.

**WRONG**:

```python
return ToolResult(tool_use_id=tool_use_id, content="Error occurred")
# or
raise RuntimeError("something went wrong")  # Loop doesn't catch this either
```

**WHY it fails.** The error message is a string. The model has to infer
category, retry intent, and severity from natural language — and it often
guesses wrong. A flake that needs a retry looks the same as a bug that
should be reported to the user.

**CORRECT** — every error has a category, a retry flag, and an actionable
message. From `bash_tool.py`:

```python
except PermissionError as exc:
    return ToolResult.failure(
        tool_use_id=tool_use_id,
        message=f"Permission denied executing {argv[0]}: {exc}",
        category="permission",     # not transient, not validation
        retryable=False,           # no point retrying
    )
except subprocess.TimeoutExpired:
    return ToolResult.failure(
        tool_use_id=tool_use_id,
        message=(
            f"Command {argv!r} exceeded the {timeout:.0f}s timeout and "
            "was killed. Increase the timeout or narrow the command."
        ),
        category="transient",      # might succeed on retry
        retryable=True,
    )
```

The message tells the model **what to do next** ("increase the timeout" /
"narrow the command"). The category + retryable flags let the loop itself
make the retry decision.

---

## 9. No rate-limit handling — crash on 429

The client calls the API, gets `HTTP 429`, and propagates the exception to
the user. The agent dies at the first rate-limit blip.

**WRONG**:

```python
def send(self, request):
    return self._client.messages.create(**self._to_wire(request))
    # SDK raises APIStatusError on 429, exception propagates, loop dies.
```

**WHY it fails.** 429 is a server-side rate limit — by definition, it
resolves itself after a wait. Crashing means the user has to manually
re-run, losing any state that wasn't in a scratchpad. The exam guide is
explicit: "rate limiting" is a known production constraint.

**CORRECT** — exponential backoff with jitter, server `Retry-After` honoured
when present. From `src/scenario_4_dev_productivity/api/client.py`:

```python
def send(self, request: APIRequest) -> APIResponse:
    wire = self._request_to_wire(request)
    last_error: APIError | None = None
    for attempt in range(self._max_retries + 1):
        try:
            raw = self._client.messages.create(**wire)
        except APIStatusError as exc:
            mapped = self._map_status_error(exc)  # 401/403/429/5xx -> typed errors
            if not mapped.is_retryable or attempt == self._max_retries:
                raise mapped from exc
            last_error = mapped
            self._sleep_backoff(attempt, mapped.retry_after)  # honour Retry-After
            continue
        except APITimeoutError as exc:
            mapped = APIError(f"Anthropic request timed out after {self._timeout_seconds:.0f}s",
                              category=ErrorCategory.TRANSIENT)
            if attempt == self._max_retries:
                raise mapped from exc
            self._sleep_backoff(attempt, None)
            continue
        else:
            return self._parse_response(raw)
    raise last_error or APIError("Anthropic request failed after retries")

def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
    if retry_after is not None and retry_after > 0:
        self._sleep(retry_after)
        return
    base = min(self._initial_backoff * (2**attempt), self._max_backoff)
    jitter = random.uniform(0, base * 0.25)         # ±25% jitter
    self._sleep(base + jitter)
```

- 401/403 → `AuthError` (not retryable — never resolve on their own).
- 429 → `RateLimitError` with `retry_after` parsed from the header.
- 5xx → `APIError(category=SERVER, retryable=True)`.
- Network timeouts / connection errors → `APIError(category=TRANSIENT, retryable=True)`.

The `clock` argument is injectable so tests verify backoff durations
deterministically without sleeping.

---

## 10. Single-agent for everything — the monolithic agent

One agent has all tools, all system prompt context, and runs the whole task.
A 200-line system prompt tries to cover exploration, code generation, AND
shell execution.

**WRONG**:

```python
SYSTEM_PROMPT = """You are an expert developer assistant. You can read
files, search code, write new files, edit existing ones, run shell commands,
deploy applications, ... """  # 200 lines of "you can do anything"

class DoEverythingAgent(BaseAgent):
    DEFAULT_TOOLS = ("Read", "Write", "Edit", "Bash", "Grep", "Glob", "Task", "WebFetch", ...)
```

**WHY it fails.** The system prompt is the agent's contract. When it tries to
cover every responsibility, the model attends to all of them and optimises
for none. The model also can't tell when it should switch modes — there's no
boundary between "I'm exploring" and "I'm writing". Tool selection gets
ambiguous. Per the exam guide, this is the canonical multi-agent
anti-pattern: when one agent does everything, specialisation suffers.

**CORRECT** — hub-and-spoke with four specialised subagents. The
coordinator dispatches; the subagents do their one job well.

```python
# src/scenario_4_dev_productivity/agents/coordinator_agent.py
class CoordinatorAgent(BaseAgent):
    """The hub of the hub-and-spoke topology."""
    DESCRIPTION: str = (
        "Orchestrator of the multi-agent developer productivity system. "
        "Decomposes tasks and dispatches them to specialized subagents "
        "(explore, generate, automate) via the Task tool. Has all "
        "built-in tools as a fallback for trivial work."
    )
    DEFAULT_TOOLS: tuple[str, ...] = ("Task", "Read", "Write", "Edit", "Bash", "Grep", "Glob")
```

Each subagent has a single, narrow responsibility:

| Subagent | Job | Allowlist | Model |
|---|---|---|---|
| `ExploreAgent` | Read-only investigation | `Read`, `Grep`, `Glob` | `claude-haiku` |
| `GenerateAgent` | Write artifacts | `Write` | `claude-sonnet` |
| `AutomateAgent` | Run shell commands | `Bash` | `claude-haiku` |
| `CoordinatorAgent` | Dispatch + synthesise | All of the above + `Task` | `claude-sonnet` |

The benefits:

- **Smaller system prompts** — each subagent prompt can be ~50 lines and
  focused.
- **Cleaner tool selection** — the Explore agent has 3 tools, not 7; the
  model picks the right one reliably.
- **Independent model tiers** — cheap Haiku for investigation, Sonnet for
  generation, no over-spending.
- **Easier testing** — each subagent can be tested in isolation against
  a fixed allowlist.

---

## Summary table

| # | Anti-pattern | This project's response |
|---|---|---|
| 1 | Pipeline hang (no `-p` flag) | `PipelineRunner` is non-interactive by construction; CLI requires `task` on argv. |
| 2 | Text-parsing stop signals | `AgenticLoop` branches on `stop_reason` only; never inspects text. |
| 3 | Thin tool descriptions | Every tool description includes input example + boundary conditions. |
| 4 | Missing `isError` flag | `ToolResult.failure(...)` sets `is_error`, `error_category`, `is_retryable`. |
| 5 | Context window exhaustion | `TrimReadOutputHook` + `ContextCompactor` keep the window healthy. |
| 6 | Subagent context leak | Fresh factory, explicit `prompt` arg, no `messages` sharing. |
| 7 | Tool overload | Per-agent allowlist enforced at construction; Explore has no `Write`, Automate has no `Read`. |
| 8 | Generic error messages | Structured errors with category, retryable, actionable message. |
| 9 | No rate-limit handling | `AnthropicClient` retries 429/5xx with exponential backoff + jitter. |
| 10 | Single-agent for everything | Hub-and-spoke: 1 coordinator + 3 specialised subagents. |

---

**See also:**

- [`architecture.md`](architecture.md) for the topology and data flow.
- [`README.md`](README.md) for the public API and setup.
- [`quiz.md`](quiz.md) for exam-style questions on these anti-patterns.
