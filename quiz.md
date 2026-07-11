# Quiz — Developer Productivity Agent

Ten exam-style questions covering the five domains of the **Claude Certified
Architect — Foundations** exam, with answers and explanations grounded in
this codebase. Use it as a self-test after reading
[`README.md`](README.md), [`architecture.md`](architecture.md), and
[`anti-patterns.md`](anti-patterns.md).

> **Domains tested:**
> - D1 — Agentic Architecture & Orchestration (27%)
> - D2 — Tool Design & MCP Integration (18%)
> - D3 — Claude Code Configuration & Workflows (20%)
> - D4 — Prompt Engineering & Structured Output (20%)
> - D5 — Context Management & Reliability (15%)

---

### Q1: Loop termination

In this project's `AgenticLoop`, what is the **only** condition that causes
the loop to return a final `AssistantMessage`?

A) When the assistant's text contains the word `"DONE"`.
B) When the turn count reaches `max_turns`.
C) When `response.message.stop_reason == StopReason.END_TURN`.
D) When the model emits a `tool_use` block with no arguments.

**Correct**: C
**Explanation**: The loop in `src/scenario_4_dev_productivity/loop/engine.py` is
explicit:

```python
if assistant.stop_reason is StopReason.END_TURN:
    return assistant
if assistant.stop_reason is StopReason.TOOL_USE:
    tool_results = self._execute_tool_calls(assistant)
    self._messages.append(ToolResultMessage(results=tool_results))
    continue
return assistant  # anything else is terminal
```

The loop **never** inspects `assistant.text` for completion markers — that is
the canonical exam-guide anti-pattern (#2 in
[`anti-patterns.md`](anti-patterns.md)). `stop_reason` is the API's
authoritative signal; text can lie. Option B is a safety belt
(`MaxTurnsExceeded`) that fires only when the loop is stuck, not a normal
termination. Option D is a defensive placeholder, not a return path.
**Domain**: D1 — Agentic Architecture & Orchestration.

---

### Q2: Structured tool errors

A tool needs to tell the model that its last call failed AND that retrying
is likely to succeed. Which `ToolResult` shape achieves that?

A) `ToolResult(tool_use_id=id, content="Error: something went wrong")`
B) `ToolResult(tool_use_id=id, content="", is_error=True)`
C) `ToolResult.failure(tool_use_id=id, message="...", category="transient", retryable=True)`
D) Raise a `RuntimeError` from inside the tool's `execute` method.

**Correct**: C
**Explanation**: `ToolResult.failure` (defined in
`src/scenario_4_dev_productivity/models/tools.py`) sets all three fields
the loop and the model need:

```python
@classmethod
def failure(cls, tool_use_id, message, *, category, retryable) -> ToolResult:
    return cls(tool_use_id=tool_use_id, content=message,
               is_error=True, error_category=category, is_retryable=retryable)
```

Option A returns a normal-looking result — the model has no programmatic
signal it failed. Option B has `is_error=True` but no category or retry
flag. Option D leaks an exception, but `ToolRegistry.execute` catches it
and renders an *unstructured* failure (`retryable=False, category="transient"`)
— the model loses the actionable message. The exam guide calls this out as
anti-pattern #4 + #8.
**Domain**: D2 — Tool Design & MCP Integration.

---

### Q3: MCP env-var expansion

A `.mcp.json` entry uses `"GITHUB_TOKEN": "${GITHUB_TOKEN}"` but the
environment variable is not set. What happens?

A) The loader raises `KeyError` immediately.
B) The placeholder is left as the literal string `"${GITHUB_TOKEN}"` so the
   downstream Pydantic schema can flag the missing field.
C) The placeholder is replaced with an empty string and the server still loads.
D) The loader substitutes the value `None`.

**Correct**: B
**Explanation**: From `src/scenario_4_dev_productivity/mcp/loader.py`:

```python
def _substitute(self, match: re.Match[str]) -> str:
    var_name = match.group(1)
    default = match.group(2)
    value = self._env.get(var_name)
    if value:
        return value
    if default is not None:
        return default
    # Unset with no default — return the literal placeholder so
    # the error message is debuggable.
    return match.group(0)
```

The expansion happens **before** Pydantic validation, so the user sees a
schema error that points to the actual missing variable name. This is
deliberate: silent defaults hide config bugs; a literal placeholder makes
the failure mode debuggable. (The `${VAR:-default}` form does support
explicit defaults.)
**Domain**: D2 — Tool Design & MCP Integration.

---

### Q4: Pipeline mode

A CI job calls `uv run scenario-4 run "Audit the test coverage in tests/"`
and the process hangs. The job logs show no API call was ever made. What is
the most likely cause?

A) The `ANTHROPIC_API_KEY` env var is not set, so the client blocks.
B) The `scenario-4` CLI is waiting for interactive input on stdin because
   the task string was misinterpreted.
C) The Bash tool has a 30-second timeout that is too short for the audit.
D) The Coordinator's `Task` tool is failing because no subagent is registered.

**Correct**: B
**Explanation**: The `scenario-4` CLI is designed to be non-interactive. The
`task` argument is **required and positional** in `_cmd_run` — it must be
on `argv`. If the user invoked the command without the task string,
`argparse` exits with an error and a non-zero code; it does **not** block
on stdin. The hang scenario in the exam guide is the `claude` binary
itself, used *without* `-p` in a CI environment — it falls back to
interactive REPL and waits for stdin. Option A would cause an immediate
exit-1 with `"error: ANTHROPIC_API_KEY is not set"`, not a hang. The
correct fix is to invoke the CLI with a non-empty `task` arg, or to use
`claude -p` (or the equivalent) in CI scripts (anti-pattern #1).
**Domain**: D3 — Claude Code Configuration & Workflows.

---

### Q5: Subagent context isolation

The Coordinator wants an `ExploreAgent` to "find every place we use
AnthropicClient". Which prompt satisfies the context-isolation rule?

A) `"Use the Grep tool to find AnthropicClient in src/."`
B) `"Find every place we use AnthropicClient and summarise the retry policy. The
   files I already read are: api/client.py, api/__init__.py. Continue from
   there."`
C) `"Find every place we use AnthropicClient and summarise the retry policy.
   The codebase is a Python project at the repo root. The relevant files
   are under src/scenario_4_dev_productivity/api/. Return paths and
   surrounding context for each match."`
D) `""` (an empty prompt; the subagent should infer what to do)

**Correct**: C
**Explanation**: Anti-pattern #6 — subagents do not inherit the
coordinator's conversation history. The `Task` tool's description makes
this explicit:

> "The subagent does NOT inherit your conversation history — the prompt
> you pass must contain all context the subagent needs (paths,
> constraints, prior findings)."

Option A is too narrow (prescribes the tool). Option B is wrong because it
implies the subagent shares the coordinator's read history. Option D
violates `TaskTool.execute`'s `not prompt.strip()` validation, which
returns a `validation` failure. Option C carries the full context
explicitly.
**Domain**: D1 — Agentic Architecture & Orchestration.

---

### Q6: Tool allowlist

Why does `ExploreAgent.DEFAULT_TOOLS = ("Read", "Grep", "Glob")` and **not**
include `Write` or `Bash`?

A) For performance — fewer tools means a shorter request payload.
B) To support tool-result caching at the API level.
C) To enforce the principle of least privilege: the Explore agent
   investigates, never mutates.
D) Because Pydantic validation rejects `Write` on this agent class.

**Correct**: C
**Explanation**: This is anti-pattern #7. The allowlist is metadata that
the agentic loop uses to project the tool definitions the model sees —
the Explore agent is **structurally incapable** of writing files or
running shell commands. The system prompt reinforces it, but the
allowlist is the perimeter. From `explore_agent.py`:

```python
class ExploreAgent(BaseAgent):
    """Read-only codebase investigator — Read, Grep, Glob only."""
    DEFAULT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob")
```

The same logic applies to `GenerateAgent` (Write only) and
`AutomateAgent` (Bash only). The Coordinator gets the full set plus
`Task` because it dispatches.
**Domain**: D1 — Agentic Architecture & Orchestration.

---

### Q7: PostToolUse hooks

What does `TrimReadOutputHook` do when a `Read` result has 5,000 lines and
`max_lines=200, keep_lines=50`?

A) Raises an exception — the result is too large to process.
B) Returns the result unchanged; the loop is responsible for truncation.
C) Keeps the first 50 and last 50 lines with a `[... N lines truncated ...]`
   marker between them.
D) Drops the entire result and returns an empty string.

**Correct**: C
**Explanation**: From `src/scenario_4_dev_productivity/context/hooks.py`:

```python
if len(lines) <= self._max_lines:
    return result
head = lines[: self._keep_lines]
tail = lines[-self._keep_lines :]
truncated = len(lines) - len(head) - len(tail)
marker = f"[... {truncated} lines truncated ...]"
new_content = "\n".join([*head, marker, *tail])
return result.model_copy(update={"content": new_content})
```

The hook is the canonical exam-guide pattern for keeping the context
window healthy (anti-pattern #5). It fires only on the `Read` tool — other
tools' outputs pass through untouched. The truncation marker is
informational (`is_error=False`) so the model doesn't treat the trim as a
failure.
**Domain**: D5 — Context Management & Reliability.

---

### Q8: Retry policy

The Anthropic API returns `HTTP 429` with `Retry-After: 5`. What does
`AnthropicClient.send` do?

A) Raises `APIError(category=VALIDATION)` immediately — the request is bad.
B) Retries indefinitely until the server stops returning 429.
C) Sleeps for 5 seconds (the server's hint) and retries, up to
   `max_retries + 1` total attempts.
D) Sleeps for `initial_backoff_seconds * (2 ** attempt)` regardless of the
   header.

**Correct**: C
**Explanation**: From `src/scenario_4_dev_productivity/api/client.py`:

```python
if status == 429:
    retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
    return RateLimitError(f"Rate limit exceeded (HTTP 429): {message}",
                          retry_after=retry_after)
```

and the retry loop:

```python
if not mapped.is_retryable or attempt == self._max_retries:
    raise mapped from exc
self._sleep_backoff(attempt, mapped.retry_after)  # honour Retry-After
```

and the backoff function:

```python
def _sleep_backoff(self, attempt, retry_after):
    if retry_after is not None and retry_after > 0:
        self._sleep(retry_after)
        return
    base = min(self._initial_backoff * (2**attempt), self._max_backoff)
    jitter = random.uniform(0, base * 0.25)
    self._sleep(base + jitter)
```

The server's `Retry-After` hint is **honoured first**. The exponential
backoff with jitter is the fallback when the header is absent. 401/403
are not retryable (auth errors won't resolve themselves). Anti-pattern #9.
**Domain**: D5 — Context Management & Reliability.

---

### Q9: Per-agent model tier

The default `.env.example` ships with `EXPLORE_MODEL=claude-haiku` and
`GENERATE_MODEL=claude-sonnet`. Why the difference?

A) Haiku is newer than Sonnet; the project uses the latest models.
B) The Explore agent runs read-only, lower-risk tool calls; cheaper Haiku
   is sufficient. The Generate agent writes code where quality matters
   more; Sonnet is the cost/quality sweet spot.
C) Sonnet is unavailable for the Explore agent's account tier.
D) The orchestrator randomly assigns models per agent.

**Correct**: B
**Explanation**: This is per-agent model tiering. The Explore agent's
allowlist is `("Read", "Grep", "Glob")` — it investigates code, summarises
findings, and returns short text. Haiku is fast, cheap, and accurate
enough for that. The Generate agent writes boilerplate, snippets, and
small files where mistakes cost more; Sonnet's higher quality is worth
the spend. The Coordinator is also Sonnet because it does the reasoning
(dispatch, synthesis). The Automate agent is Haiku because it runs
deterministic shell commands. See `src/scenario_4_dev_productivity/config.py`
and the `agents/*.py` files for the `model=model or config.<name>_model`
wiring.
**Domain**: D3 — Claude Code Configuration & Workflows.

---

### Q10: Context compression

A long conversation has been running for 20 turns. The original task
("audit the retry policy") and the most recent 4 turns are essential; the
middle 16 turns are verbose tool results. What does
`ContextCompactor.compact(messages)` do?

A) Drops everything older than 4 turns.
B) Keeps the first user message, the last 4 messages verbatim, and
   summarises the middle `ToolResultMessage`s with `summarise_tool_result`.
C) Sends the full conversation to the API and asks Claude to summarise it.
D) Compresses every `ToolResult` to a single line.

**Correct**: B
**Explanation**: From `src/scenario_4_dev_productivity/context/compact.py`:

```python
def compact(self, messages):
    first, *rest = messages
    kept: list[Message | ToolResultMessage] = [first]   # Pin the task
    if not rest:
        return kept
    recent_window = rest[-self._recent_turns :] if self._recent_turns else []
    recent_set: set[int] = {id(m) for m in recent_window}
    middle = rest[: -self._recent_turns] if self._recent_turns else rest
    for msg in middle:
        if id(msg) in recent_set:
            continue
        if isinstance(msg, ToolResultMessage):
            kept.append(_summarise_tool_result_message(msg, self._keep_lines))
        else:
            kept.append(msg)                           # Assistant messages kept intact
    kept.extend(recent_window)
    return kept
```

Three guarantees:

1. The first user message (the original task) is **always** preserved —
   the model never forgets the goal.
2. The last `recent_turns` messages are kept verbatim — the model's
   "now" is intact.
3. `ToolResultMessage`s in the middle are trimmed by
   `summarise_tool_result` (first N + last N lines + marker).
   `AssistantMessage`s are kept intact (they're usually short).

The result is always shorter than the input by design. Anti-pattern #5.
**Domain**: D5 — Context Management & Reliability.

---

## Domain coverage

| Question | Domain | Weight | Anti-pattern cross-ref |
|---|---|---|---|
| Q1 | D1 — Agentic Architecture & Orchestration | 27% | #2 |
| Q2 | D2 — Tool Design & MCP Integration | 18% | #4, #8 |
| Q3 | D2 — Tool Design & MCP Integration | 18% | — |
| Q4 | D3 — Claude Code Configuration & Workflows | 20% | #1 |
| Q5 | D1 — Agentic Architecture & Orchestration | 27% | #6 |
| Q6 | D1 — Agentic Architecture & Orchestration | 27% | #7 |
| Q7 | D5 — Context Management & Reliability | 15% | #5 |
| Q8 | D5 — Context Management & Reliability | 15% | #9 |
| Q9 | D3 — Claude Code Configuration & Workflows | 20% | — |
| Q10 | D5 — Context Management & Reliability | 15% | #5 |

All five domains appear at least once. D1 (3 questions), D2 (2), D3 (2), D5
(3), D4 (0) — Prompt Engineering & Structured Output is covered in the
codebase (per-agent system prompts, `AssistantContent` blocks) but doesn't
lend itself to a single multiple-choice question; the
[`prompts/`](src/scenario_4_dev_productivity/prompts/) module is the
canonical reference for D4 in this project.

---

## How to use this quiz

1. **Read first, answer later.** Skim [`README.md`](README.md) and
   [`architecture.md`](architecture.md) before attempting the questions.
2. **For each wrong answer, jump to the cited anti-pattern.** Every
   explanation points to a numbered pattern in
   [`anti-patterns.md`](anti-patterns.md) and the source file where the
   correct pattern lives.
3. **Re-run the test suite after reading the answers.** `uv run pytest
   tests/ -q` should print `304 passed` — the quiz is a code review, not
   a replacement for the actual tests.
4. **Re-take the quiz in 24 hours.** If you can answer all 10 from memory,
   the underlying patterns are locked in.

---

## Further reading

- [`README.md`](README.md) — quick path, usage, project layout.
- [`architecture.md`](architecture.md) — full system map, data flow, risks, scaling.
- [`anti-patterns.md`](anti-patterns.md) — 10 mistakes and how this code avoids them.
- Exam references in `docs/guide_en.md` and the instructor exam guide.
