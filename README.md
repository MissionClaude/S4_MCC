# Scenario 4 — Developer Productivity with Claude

A reference implementation of a hub-and-spoke multi-agent developer productivity
toolkit built on the **Claude Agent SDK** (`anthropic` + Pydantic). Engineers
explore unfamiliar codebases, generate boilerplate, and automate shell workflows
through specialised subagents — all driven by a `stop_reason`-based agentic loop.

> **Scenario 4 of the Claude Certified Architect — Foundations exam.**
> Primary domains: Tool Design & MCP Integration, Claude Code Configuration &
> Workflows, Agentic Architecture & Orchestration. Secondary: Context Management
> & Reliability, Prompt Engineering & Structured Output.

---

## Quick path

```bash
# 1. Install (Python 3.13, uv)
uv sync

# 2. Configure credentials
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# 3. Run a task in pipeline mode (non-interactive, = `claude -p`)
uv run scenario-4 run "Find every place we use AnthropicClient and summarise the retry policy."

# 4. Run a task with JSON output (CI-friendly)
uv run scenario-4 run "Audit the test coverage in tests/" --output-format json

# 5. Inspect the scratchpad
uv run scenario-4 compact .scratchpad.md

# 6. Run the test suite (304 tests, no API key required)
uv run pytest tests/ -q
```

---

## What this project is

| Layer | What it does |
|---|---|
| **Agents** | Four concrete agents: `CoordinatorAgent` (hub), `ExploreAgent` (read-only), `GenerateAgent` (write), `AutomateAgent` (bash). |
| **Loop** | `AgenticLoop` drives a `stop_reason`-dispatched turn cycle: `tool_use` → execute → loop, `end_turn` → return final. |
| **Tools** | Six built-ins (`Read`, `Grep`, `Glob`, `Write`, `Edit`, `Bash`) registered in a `ToolRegistry`. Each subagent gets a narrow allowlist. |
| **API** | `AnthropicClient` wraps the SDK with exponential backoff + jitter, 401/403/429/5xx mapping, and structured error propagation. |
| **MCP** | `MCPConfigLoader` parses `.mcp.json` and resolves `${ENV_VAR}` placeholders. Discovery produces typed tool specs and adapters. |
| **Pipeline** | `PipelineRunner` runs a task in a fresh agent and renders `text` or `json`. `run_multi_pass` does per-file + integration analysis. |
| **Context** | `ScratchpadManager` (Markdown round-trip), `ContextCompactor` (preserves task + recent turns), `TrimReadOutputHook` (PostToolUse). |

See [`architecture.md`](architecture.md) for the full system map and
[`anti-patterns.md`](anti-patterns.md) for the decisions this design deliberately
rejects.

---

## Key decisions

- **Hub-and-spoke topology.** The Coordinator is the only agent that talks to
  the user; subagents are reached exclusively via the `Task` tool. There is no
  peer-to-peer agent routing.
- **Per-agent model tiers.** `claude-sonnet` for the Coordinator and the code
  generator; `claude-haiku` for the read-only investigator and the shell
  executor. Configurable per agent via env (`COORDINATOR_MODEL`, `EXPLORE_MODEL`,
  `GENERATE_MODEL`, `AUTOMATE_MODEL`).
- **`stop_reason` as the only loop-termination signal.** The loop never
  inspects assistant text for completion markers. `end_turn` stops; `tool_use`
  continues; everything else is treated as terminal.
- **Structured tool errors.** Every tool returns a `ToolResult` with
  `is_error`, `error_category` (`transient` / `validation` / `permission`), and
  `is_retryable`. Failures are fed back to the model as ordinary tool results
  so it can adapt.
- **Explicit context passing.** Subagents do not inherit the coordinator's
  conversation history. The Task tool's `prompt` argument carries every piece
  of context the subagent needs (paths, prior findings, constraints).
- **Fresh agent per pipeline run.** `PipelineRunner` calls the agent factory
  once per task. `run_multi_pass` builds a new per-file agent each iteration.
  Session isolation is enforced by construction, not by convention.
- **Feature-branch chain delivery.** The change landed across 5 chained PRs
  (foundation → core infra → agents+MCP → pipeline+context → docs). Each PR
  was a reviewable work unit.

---

## Architecture overview

```
                ┌──────────────────────┐
                │  User / CLI / API    │
                └──────────┬───────────┘
                           │ task
                           ▼
                ┌──────────────────────┐
                │   CoordinatorAgent   │  (claude-sonnet)
                │   Task + built-ins   │
                └──────────┬───────────┘
                           │ Task(subagent_type, prompt)
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ ExploreAgent  │  │ GenerateAgent │  │ AutomateAgent │
│ Read Grep Glob│  │ Write only    │  │ Bash only     │
│ (claude-haiku)│  │(claude-sonnet)│  │ (claude-haiku)│
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                ┌──────────────────────┐
                │   AgenticLoop        │
                │ stop_reason dispatch │
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ToolRegistry       AnthropicClient      PostToolUse hooks
   (Read/Write/       (retry/backoff)      (Trim + Logging)
    Bash/Grep/Glob)   + RateLimitError
```

Full diagram, data flow, and component map in [`architecture.md`](architecture.md).

---

## Setup

### Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) (handles venv + dependencies)
- An Anthropic API key from <https://console.anthropic.com/settings/keys>

### Install

```bash
# 1. Clone and enter the repo
git clone <repo> s4_mcc && cd s4_mcc

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Copy and edit the environment file
cp .env.example .env
# Replace the ANTHROPIC_API_KEY placeholder with your real key
```

The `.env` file is git-ignored. **Never commit credentials.**

### Configuration

All configuration is read from environment variables (loaded via
`python-dotenv`). See [`.env.example`](.env.example) for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(none)* | **Required.** API key. |
| `COORDINATOR_MODEL` | `claude-sonnet` | Model for the Coordinator (full tier). |
| `EXPLORE_MODEL` | `claude-haiku` | Model for ExploreAgent (lean tier). |
| `GENERATE_MODEL` | `claude-sonnet` | Model for GenerateAgent (balanced tier). |
| `AUTOMATE_MODEL` | `claude-haiku` | Model for AutomateAgent (lean tier). |
| `ANTHROPIC_MAX_RPM` | `50` | Rate limit (informational). |
| `ANTHROPIC_TIMEOUT_SECONDS` | `120` | Per-request API timeout. |
| `MAX_CONVERSATION_TURNS` | `15` | Loop turn cap (`max_turns`). |
| `SCRATCHPAD_PATH` | `.scratchpad.md` | Scratchpad file location. |
| `MCP_CONFIG_PATH` | `.mcp.json` | MCP config file path. |
| `PIPELINE_MODE` | `false` | Hint for non-interactive runners. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

---

## Usage

### CLI

The project installs a `scenario-4` console script. Two subcommands:

```bash
# Run a task in pipeline mode (the `claude -p` equivalent)
uv run scenario-4 run "Summarise the retry policy in api/client.py"

# JSON output for CI / downstream tooling
uv run scenario-4 run "List every tool in the registry" --output-format json

# Override the model or turn cap
uv run scenario-4 run "..." --model claude-haiku --max-turns 8

# Inspect the scratchpad
uv run scenario-4 compact .scratchpad.md
uv run scenario-4 compact .scratchpad.md --output-format json
```

Exit code is `0` on success, `1` if config is missing or the task fails.
Pipeline mode never blocks on stdin — it processes the prompt and exits.

### Python API

```python
from scenario_4_dev_productivity import (
    AnthropicClient,
    CoordinatorAgent,
    default_registry,
)
from scenario_4_dev_productivity.pipeline.runner import (
    PipelineRunner,
    JSONOutputFormat,
    make_agent_factory,
)

# Wire the components
client = AnthropicClient(api_key="sk-ant-...", max_retries=3)
registry = default_registry()
factory = make_agent_factory(CoordinatorAgent, registry=registry, client=client)

# Run a task
runner = PipelineRunner(client=client, agent_factory=factory)
result = runner.run(
    "Find every place we use AnthropicClient and summarise the retry policy.",
    output=JSONOutputFormat,
    metadata={"ci_job": "1234"},
)
print(result.text)                       # Final assistant text
print(result.to_json_dict())             # JSON-serialisable dict
print(result.turn_count)                 # Number of API calls
```

### Spawning a subagent explicitly

```python
from scenario_4_dev_productivity import (
    AnthropicClient,
    ExploreAgent,
    default_registry,
)

client = AnthropicClient(api_key="sk-ant-...")
explore = ExploreAgent(registry=default_registry(), client=client)

# The agent has only Read/Grep/Glob — no Write, no Bash
response = explore.run("Map the test layout under tests/ and list the fixtures.")
print(response.text)
print(response.stop_reason)              # StopReason.END_TURN
```

### Persisting findings to the scratchpad

```python
from scenario_4_dev_productivity.context import ScratchpadManager

pad = ScratchpadManager(".scratchpad.md")
pad.append_finding(
    topic="Retry policy summary",
    body="- 429: server hint + exponential backoff\n- 5xx: retryable\n- 401/403: not retryable",
    source="coordinator",
)
entries = pad.read_entries()             # Parse back into typed objects
```

### Compressing context (`/compact` equivalent)

```python
from scenario_4_dev_productivity.context import ContextCompactor

compactor = ContextCompactor(keep_lines=20, recent_turns=4)
trimmed = compactor.compact(messages)     # Preserves first user message + last 4 turns
```

### Loading MCP config

```python
from scenario_4_dev_productivity.mcp import MCPConfigLoader

loader = MCPConfigLoader()
config = loader.load(".mcp.json")
for server in config.server_names():
    print(server)
```

---

## Testing

The test suite has **304 tests** across 7 packages. **No API key is required** —
every test mocks the Anthropic client at the boundary.

```bash
# Run the full suite
uv run pytest tests/ -q

# Run a specific package
uv run pytest tests/test_integration -v
uv run pytest tests/test_agents -v
uv run pytest tests/test_loop -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### What the tests cover

| Package | Tests | Focus |
|---|---|---|
| `test_models/` | 50 | Pydantic validation, field defaults, message serialisation. |
| `test_tools/` | 61 | Each tool's execution, error paths, timeouts. |
| `test_api/` | 28 | Retry logic, error classification, backoff with jitter. |
| `test_loop/` | 14 | `stop_reason` dispatch, `max_turns` cap, mock state transitions. |
| `test_agents/` | 58 | Subagent spawning, context isolation, allowlist enforcement. |
| `test_mcp/` | 43 | `.mcp.json` parsing, env-var expansion, tool discovery. |
| `test_integration/` | 50 | Full workflow, pipeline, context compression, error scenarios. |
| **Total** | **304** | — |

### Linting and type-checking

```bash
uv run ruff check .            # Lint
uv run mypy src/               # Strict mypy on the package
```

Both must pass with zero issues — `mypy` runs in `strict` mode
with `warn_unreachable = True`.

---

## Project layout

```
s4_mcc/
├── src/scenario_4_dev_productivity/
│   ├── __init__.py              # Public API surface
│   ├── config.py                # Env-var config loader
│   ├── agents/                  # BaseAgent + 4 concrete agents
│   │   ├── base.py
│   │   ├── coordinator_agent.py # Hub + TaskTool
│   │   ├── explore_agent.py     # Read/Grep/Glob
│   │   ├── generate_agent.py    # Write only
│   │   └── automate_agent.py    # Bash only
│   ├── api/                     # AnthropicClient (retry + error mapping)
│   ├── loop/                    # AgenticLoop (stop_reason dispatch)
│   ├── models/                  # Pydantic: messages, tools, API types
│   ├── tools/                   # 6 built-in tools + ToolRegistry
│   ├── mcp/                     # .mcp.json loader + tool discovery
│   ├── prompts/                 # Per-agent system prompts
│   ├── pipeline/                # PipelineRunner + CLI + multi-pass
│   └── context/                 # Scratchpad, compactor, PostToolUse hooks
├── tests/                       # 304 tests across 8 packages
├── .env.example                 # Configuration template
├── .mcp.json                    # Reference MCP config
├── pyproject.toml               # uv + pytest + ruff + mypy config
└── README.md                    # ← you are here
```

---

## CI / pipeline-mode integration

The pipeline module is the `claude -p` analogue. To run from a CI runner:

```yaml
# .github/workflows/agent.yml (example)
- name: Run developer agent
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    uv sync
    uv run scenario-4 run "$TASK" --output-format json > result.json
    jq '.turn_count, .stop_reason' result.json
```

Key properties:
- **No stdin reads.** The CLI processes the prompt and exits — it never blocks
  on a TTY.
- **Structured output.** `--output-format json` gives you a `PipelineResult`
  with `task`, `text`, `stop_reason`, `turn_count`, and any metadata you attach.
- **Exit codes.** `0` on success, `1` on config or runtime failure.

---

## Exam-domain coverage

| Domain | Where it lives in this project |
|---|---|
| **D1 — Agentic Architecture & Orchestration** | `loop/engine.py` (stop_reason dispatch), `agents/coordinator_agent.py` (Task tool, hub-and-spoke), `pipeline/runner.py` (fresh agent per run), `pipeline/multi_pass.py` (per-file isolation). |
| **D2 — Tool Design & MCP Integration** | `tools/registry.py` (structured `ToolResult`), `tools/*.py` (descriptions with input examples + boundary conditions), `mcp/loader.py` (config + env expansion), `mcp/discovery.py` (tool adapters). |
| **D3 — Claude Code Configuration & Workflows** | `.mcp.json` (project-scoped MCP), `.env.example` (per-agent model tier), `pipeline/cli.py` (`--output-format` analogous to `--output-format json`), `pyproject.toml` (`[project.scripts]`). |
| **D4 — Prompt Engineering & Structured Output** | `prompts/*.py` (per-agent system prompts with explicit role + scope + rules), `models/messages.py` (typed `AssistantContent` blocks). |
| **D5 — Context Management & Reliability** | `context/scratchpad.py` (Markdown round-trip across `/compact`), `context/compact.py` (preserves task + recent turns), `context/hooks.py` (`TrimReadOutputHook` + `LoggingHook`), `api/client.py` (exponential backoff + jitter, structured 429/5xx mapping). |

---

## Further reading

- [`architecture.md`](architecture.md) — full system diagram, data flow, risks, scaling.
- [`anti-patterns.md`](anti-patterns.md) — 10 mistakes and how this code avoids them.
- [`quiz.md`](quiz.md) — 10 exam-style questions with explanations.
- Exam reference: `docs/guide_en.md` and the instructor exam guide in `docs/`.

---

## License

This project is a reference implementation for the Claude Certified Architect
— Foundations certification exam. Use freely for exam preparation.
