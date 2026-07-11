# Architecture — Developer Productivity Agent

A reference design for a hub-and-spoke multi-agent developer productivity
system built on the Claude Agent SDK. The Coordinator is the only agent that
talks to the user; everything else is a specialised subagent dispatched
through the `Task` tool.

---

## Topology

```
                          ┌─────────────────────────────────────┐
                          │           User / CLI / CI           │
                          │  `scenario-4 run "..."`             │
                          │  Python API: PipelineRunner.run()   │
                          └────────────────┬────────────────────┘
                                           │ task (string)
                                           ▼
                  ┌──────────────────────────────────────────────┐
                  │             CoordinatorAgent                  │
                  │   (claude-sonnet — full tier)                 │
                  │                                              │
                  │   Tools: Task, Read, Write, Edit,            │
                  │          Bash, Grep, Glob                    │
                  │                                              │
                  │   System prompt: COORDINATOR_SYSTEM_PROMPT   │
                  │   Role: decompose + dispatch + synthesise    │
                  └────────────────┬─────────────────────────────┘
                                   │
                                   │ Task(subagent_type, prompt, description)
                                   │  ↓ a fresh subagent per call
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐        ┌─────────────────┐
│  ExploreAgent   │       │  GenerateAgent  │        │  AutomateAgent  │
│ (claude-haiku)  │       │ (claude-sonnet) │        │  (claude-haiku) │
│                 │       │                 │        │                 │
│  Read, Grep,    │       │  Write only     │        │  Bash only      │
│  Glob           │       │                 │        │                 │
│                 │       │                 │        │                 │
│  Read-only      │       │  Boilerplate,   │        │  Tests, builds, │
│  investigation  │       │  snippets, fix  │        │  formatters,    │
│  of the code-   │       │  patterns       │        │  one-off        │
│  base.          │       │                 │        │  scripts.       │
└────────┬────────┘       └────────┬────────┘        └────────┬────────┘
         │                         │                          │
         │  ToolResult (text)      │  ToolResult (text)       │  ToolResult
         └─────────────────────────┼──────────────────────────┘  (text)
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │              AgenticLoop                     │
                  │   stop_reason dispatch                       │
                  │     TOOL_USE → execute + loop                 │
                  │     END_TURN → return final                  │
                  └────────────────┬─────────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  ToolRegistry    │  │  AnthropicClient │  │  PostToolUse     │
    │                  │  │                  │  │  hooks           │
    │  Read Grep Glob  │  │  Retry/backoff   │  │                  │
    │  Write Edit Bash │  │  401/403/429/5xx │  │  TrimReadOutput  │
    │  Task (in Coord) │  │  Exponential +   │  │  LoggingHook     │
    │                  │  │  jitter          │  │                  │
    └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
             │                     │
             │                     │  HTTP
             │                     ▼
             │           ┌──────────────────┐
             │           │  Anthropic API   │
             │           │  (claude-haiku,  │
             │           │   claude-sonnet) │
             │           └──────────────────┘
             │
             ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  Subprocess      │    │  Filesystem      │
    │  (Bash tool)     │    │  (Read/Write/    │
    │  shlex + timeout │    │   Edit/Grep/Glob)│
    └──────────────────┘    └──────────────────┘

    ┌────────────────────────────────────────────────────┐
    │  Cross-cutting infrastructure (used by all agents) │
    │                                                    │
    │  • ScratchpadManager  — Markdown round-trip file   │
    │  • ContextCompactor   — /compact equivalent        │
    │  • MCPConfigLoader    — .mcp.json + env expansion  │
    │  • Config             — env-var settings           │
    └────────────────────────────────────────────────────┘
```

---

## Data flow

End-to-end trace of a single user request: *"Find every place we use
AnthropicClient and summarise the retry policy."*

```
1. USER → CLI / API
   `scenario-4 run "Find every place we use AnthropicClient and summarise the retry policy."`

2. CLI builds the wiring
   ├── AnthropicClient(api_key=...)              # retry/backoff
   ├── default_registry()                         # 6 built-in tools
   └── PipelineRunner(client, factory)            # fresh agent per run

3. PipelineRunner.run(task)
   agent = factory()                              # fresh CoordinatorAgent
   loop   = agent.build_loop()                    # fresh AgenticLoop
   assistant = loop.run(task)                     # step 4 ↓

4. AgenticLoop.run(task) — turn 1
   ├── Build APIRequest(model=claude-sonnet, system=COORDINATOR_SYSTEM_PROMPT,
   │                    messages=[user(task)], tools=CoordinatorAgent's allowlist)
   ├── client.send(request)                       # HTTP POST to Anthropic
   │     └── (transient failure path)             # 429/5xx → backoff + retry
   ├── response.message.stop_reason == "tool_use"
   ├── _execute_tool_calls(assistant)
   │     └── tool: Task(subagent_type="explore",
   │                    prompt="Find every place AnthropicClient is used
   │                            in src/. Return paths + brief context.",
   │                    description="Audit AnthropicClient usage")
   │           └── TaskTool.execute():
   │                 ├── factory("explore")       # fresh ExploreAgent
   │                 ├── subagent.run(prompt)     # sub-loop (steps 5–6 ↓)
   │                 └── return ToolResult(text=subagent_response)
   └── append ToolResultMessage → continue

5. Subagent (ExploreAgent) — internal turn cycle
   ├── stop_reason == "tool_use" → Read/Grep/Glob
   ├── stop_reason == "end_turn" → return AssistantMessage(text=summary)
   └── subagent.messages is a fresh list — coordinator's history does NOT leak

6. Back in the Coordinator loop — turn N
   ├── Coordinator receives ExploreAgent's summary as a tool result
   ├── stop_reason == "end_turn" → return AssistantMessage(text=synthesis)
   └── PipelineRunner builds PipelineResult(task, text, stop_reason, turn_count)

7. PipelineRunner renders
   ├── text mode → just `text`
   └── json mode → {"task": ..., "text": ..., "stop_reason": ..., "turn_count": ...}

8. CLI writes rendered output to stdout and exits 0
```

**Key invariants**

- The subagent in step 5 has its **own** message list. There is no
  `coordinator.messages` passed to it. The only input is the `prompt` string.
- The Coordinator's `Task` tool call is the **only** way it talks to a
  subagent. It cannot directly call `subagent.run(...)` — it has to go
  through the agentic loop and the model.
- If the API returns `stop_reason == "tool_use"` with no `ToolUseBlock`s
  (a malformed response), the loop injects a structured failure as a
  placeholder rather than panicking. From `loop/engine.py`:

  ```python
  def _no_tool_calls_placeholder(self) -> ToolResult:
      return ToolResult.failure(
          tool_use_id="synthetic-no-calls",
          message="Model returned stop_reason=tool_use with no tool_use blocks.",
          category="validation",
          retryable=False,
      )
  ```

---

## Component map

| Module | Responsibility | Key types |
|---|---|---|
| `agents/base.py` | Declarative agent bundle: system prompt + allowlist + model. | `BaseAgent`, `build_agent`, `with_model`, `with_tools`, `with_prompt` |
| `agents/coordinator_agent.py` | Hub: registers `Task` tool, dispatches subagents, synthesises results. | `CoordinatorAgent`, `TaskTool`, `SUBAGENT_TYPES` |
| `agents/explore_agent.py` | Read-only investigator (Read/Grep/Glob). | `ExploreAgent` |
| `agents/generate_agent.py` | Code writer (Write only). | `GenerateAgent` |
| `agents/automate_agent.py` | Shell runner (Bash only). | `AutomateAgent` |
| `api/client.py` | Anthropic SDK wrapper: retry, backoff, error classification. | `AnthropicClient`, `APIError`, `RateLimitError`, `AuthError` |
| `loop/engine.py` | `stop_reason`-driven turn cycle. | `AgenticLoop`, `MaxTurnsExceeded` |
| `models/api.py` | API request/response/error types. | `APIRequest`, `APIResponse`, `ErrorCategory` |
| `models/messages.py` | Wire message types and `StopReason` enum. | `UserMessage`, `AssistantMessage`, `ToolResultMessage`, `StopReason` |
| `models/tools.py` | Tool schema and structured result. | `ToolDefinition`, `ToolCall`, `ToolResult`, `AgentConfig` |
| `tools/registry.py` | Dispatch table for tools. | `Tool`, `ToolRegistry` |
| `tools/read_tool.py` | Read file contents. | `ReadTool` |
| `tools/grep_tool.py` | Search file contents (regex). | `GrepTool` |
| `tools/glob_tool.py` | Find files by pattern. | `GlobTool` |
| `tools/write_tool.py` | Write new file. | `WriteTool` |
| `tools/edit_tool.py` | Edit existing file. | `EditTool` |
| `tools/bash_tool.py` | Run shell command (shlex + timeout). | `BashTool` |
| `mcp/loader.py` | Parse `.mcp.json`, expand `${ENV_VAR}`. | `MCPConfig`, `MCPConfigLoader`, `MCPServerConfig`, `MCPToolSpec` |
| `mcp/discovery.py` | Produce tool specs and adapters from a config. | `MCPToolDiscovery`, `MCPToolAdapter` |
| `prompts/*.py` | Per-agent system prompts. | `COORDINATOR_SYSTEM_PROMPT`, `EXPLORE_SYSTEM_PROMPT`, `GENERATE_SYSTEM_PROMPT`, `AUTOMATE_SYSTEM_PROMPT` |
| `pipeline/runner.py` | Non-interactive, fresh-agent-per-run pipeline. | `PipelineRunner`, `PipelineResult`, `OutputFormat`, `make_agent_factory` |
| `pipeline/multi_pass.py` | Per-file pass + integration pass. | `run_multi_pass`, `run_pass`, `PassResult`, `MultiPassResult` |
| `pipeline/cli.py` | `scenario-4 run` / `scenario-4 compact` console script. | `main`, `build_parser` |
| `context/scratchpad.py` | Markdown scratchpad (cross-context persistence). | `ScratchpadManager`, `ScratchpadEntry` |
| `context/compact.py` | `/compact` equivalent (preserves task + recent turns). | `ContextCompactor`, `compact_messages`, `summarise_tool_result` |
| `context/hooks.py` | PostToolUse hook protocol + bundled hooks. | `PostToolUseHook`, `TrimReadOutputHook`, `LoggingHook`, `run_hooks` |
| `config.py` | Env-var configuration singleton. | `Config`, `config` |

---

## Risks and mitigations

| Risk | Severity | Mitigation in this codebase |
|---|---|---|
| **API key exposure** | High | `.env` is git-ignored; `.env.example` carries placeholders. `Config.validate()` rejects placeholder values. |
| **Rate limiting (429)** | Medium | `AnthropicClient` retries with exponential backoff + jitter; honours the `Retry-After` header. |
| **Network timeouts / connection errors** | Medium | Same retry path; `APITimeoutError` → `APIError(category=TRANSIENT, retryable=True)`. |
| **Bash injection** | High | `BashTool` uses `shlex.split` + `subprocess.run(argv, shell=False)`; never `shell=True`. Hard wall-clock timeout (default 30s, max 600s). |
| **Context window exhaustion** | Medium | `TrimReadOutputHook` trims large Read results; `ContextCompactor` summarises old turns; `max_turns` cap on the loop. |
| **Subagent context leak** | Medium | Fresh subagent per Task call; explicit `prompt` argument; no shared `messages` list. |
| **Tool overload** | Medium | Per-agent `allowed_tools` allowlist; structurally impossible for Explore to write or Automate to read. |
| **Non-determinism** | Low | Same task can produce different outputs; tests mock the SDK at the boundary and assert on structure, not text. |
| **Cost** | Medium | Per-agent model tier: Sonnet for Coordinator/Generate, Haiku for Explore/Automate. `ANTHROPIC_MAX_RPM` env-var for visibility. |
| **Unbounded loops** | Medium | `max_turns=15` default cap; `MaxTurnsExceeded` raised and surfaced. |
| **Structured error loss** | Medium | `ToolRegistry.execute` never raises; always returns a `ToolResult` (success or structured failure). |
| **MCP server unavailable** | Low | `.mcp.json` is config-only; tools are registered from a static manifest when the server can't be reached. |

---

## Scaling

### Adding a new subagent

1. **Create the system prompt** in `src/scenario_4_dev_productivity/prompts/<name>.py`.
2. **Subclass `BaseAgent`** in `src/scenario_4_dev_productivity/agents/<name>_agent.py`,
   pinning the allowlist and model.
3. **Add the new type to `SUBAGENT_TYPES`** in `coordinator_agent.py`.
4. **Update the Task tool's description** so the coordinator's model knows
   when to dispatch to the new agent.
5. **Update the default subagent factory** in `coordinator_agent.py`.
6. **Add tests** under `tests/test_agents/`.

### Adding a new tool

1. **Implement `execute(tool_use_id, arguments) -> ToolResult`** in
   `src/scenario_4_dev_productivity/tools/<name>_tool.py`. The method
   **must never raise** — return `ToolResult.failure(...)` instead.
2. **Provide a `definition: ToolDefinition`** with input example + boundary
   conditions (see [`anti-patterns.md`](anti-patterns.md) #3).
3. **Add to `BUILTIN_TOOLS`** in `src/scenario_4_dev_productivity/tools/base.py`.
4. **Decide which agents' allowlists include it.**
5. **Add tests** under `tests/test_tools/`.

### Adding a new MCP server

1. **Add the server entry to `.mcp.json`** with `${ENV_VAR}` placeholders
   for any secrets.
2. **Add the static tool manifest** (name, description, input schema) under
   the server's `tools` key. This lets the discovery layer register typed
   tools without needing to launch the server.
3. **The loader picks it up automatically** — no code change required.
4. **Add a test** in `tests/test_mcp/` that loads the config and asserts
   the expected tool names.

### Adding a new context hook

1. **Implement the `PostToolUseHook` protocol** in
   `src/scenario_4_dev_productivity/context/<name>.py`:
   `def __call__(self, tool_name, tool_use_id, result) -> ToolResult`.
2. **Wire it into the agent** (currently the registry constructs tools
   without hooks — extend the wiring to accept a hook list).
3. **Make it idempotent** — hooks run in order, the result of one feeds
   the next.
4. **Make it safe** — `run_hooks` catches exceptions and logs at WARNING;
   a buggy hook must not break the loop.

---

## Boundaries — what the system does NOT do

This is a reference implementation for the exam demo. It is **not** a
production deployment. The following are explicitly out of scope:

- **Real MCP transport.** The `.mcp.json` config is parsed and the static
  tool manifest is used to register typed tool specs, but the `npx
  @anthropic/mcp-github` server is **not** launched at runtime. The tool
  adapter returns a structured failure explaining this. To enable real
  transport, replace the adapter with an stdio/HTTP client.
- **Container sandboxing.** The Bash tool uses `subprocess.run` with a
  timeout, but it does **not** run in a Docker container, cgroup, or
  namespace. The threat model is "model emits something stupid", not
  "untrusted code execution".
- **Persistent sessions.** The CLI runs each task in a fresh process. There
  is no `--resume` equivalent. The `ScratchpadManager` is the cross-session
  persistence surface: a long-running job writes its state to the
  scratchpad, a subsequent job reads from it.
- **Authentication for the CLI.** The CLI uses `ANTHROPIC_API_KEY` from the
  environment. There is no OAuth flow, no user login, no team management.
- **Multi-tenant isolation.** All agents in a single process share the
  `AnthropicClient` instance. If you need per-tenant isolation, build a
  client pool.
- **Streaming responses.** The `AgenticClient` waits for the full response.
  The API supports streaming, but this implementation does not consume it.
- **Distributed execution.** Subagents run in the same Python process as
  the coordinator. There is no RPC, no queue, no worker pool.
- **Hosting / deployment infrastructure.** The package is a library + a
  CLI. There is no FastAPI server, no gRPC, no Dockerfile, no Helm chart.
  Production deployment is the integrator's job.

---

## Trace: one full coordinator turn

Pseudocode for a single turn of the coordinator loop, with the
`stop_reason` dispatch and tool execution:

```python
# Inside AgenticLoop.run (loop/engine.py)
while self._turn_count < cap:
    # 1. Build a wire-format request from the current conversation.
    request = APIRequest(
        model=self._model,
        system=active_system,
        messages=[m.to_user_message() if isinstance(m, ToolResultMessage) else m
                  for m in self._messages],
        tools=active_tools,
    )

    # 2. Send it. The client retries 429/5xx with backoff.
    response = self._client.send(request)
    self._turn_count += 1

    # 3. Append the assistant's message to the conversation.
    assistant = response.message
    self._messages.append(assistant)

    # 4. Branch on stop_reason — NOT on text content.
    if assistant.stop_reason is StopReason.END_TURN:
        return assistant
    if assistant.stop_reason is StopReason.TOOL_USE:
        tool_results = self._execute_tool_calls(assistant)
        self._messages.append(ToolResultMessage(results=tool_results))
        continue
    return assistant  # anything else is terminal

# 5. Out of turns — surface it.
raise MaxTurnsExceeded(max_turns=cap, last_response=...)
```

The single most important line is the `if assistant.stop_reason is
StopReason.END_TURN` branch. The loop **never** inspects `assistant.text`.
That is the canonical exam-guide rule: `stop_reason` is the only reliable
signal, text can lie, the API cannot.

---

## See also

- [`README.md`](README.md) — quick path, usage, project layout.
- [`anti-patterns.md`](anti-patterns.md) — the ten mistakes this design avoids.
- [`quiz.md`](quiz.md) — exam-style questions on this architecture.
