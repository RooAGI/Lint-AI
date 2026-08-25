# Agent Integration Benchmark

This benchmark measures how Lint-AI behaves when integrated with an agent
client. It is not a benchmark of the standalone Lint-AI retrieval engine and
it is not an academic corpus benchmark. It measures the complete client path:

- the agent's setup turn;
- lifecycle hook execution and durable capture;
- a fresh continuation turn;
- model tokens and tool calls;
- memory recall and forbidden-memory leakage; and
- client-visible interaction latency.

The suite currently supports Claude Code, Codex, and Antigravity CLI (AGY). Each client has a
provider-specific launcher and scenario directory, while both use the shared
orchestration in `benchmark/codex_code/src/runner.py`.

## Shared metric contract

Every setup and continuation phase is parsed into the same privacy-preserving
record before report aggregation. The contract is:

```json
{
  "schema_version": 1,
  "provider": "claude|codex|agy",
  "parent_tokens": {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
    "total": 0
  },
  "all_model_tokens": { "...": "same token fields" },
  "subagent_tokens": null,
  "tool_calls": 0,
  "repeated_tool_calls": 0,
  "retrieved_documents": 0,
  "hook_events": 0,
  "hook_latency_ms": 0
}
```

`input_tokens` includes uncached and cached input only in the report's
provider-normalized accounting; the raw cache components remain available for
inspection. `total` is populated only when all required components are known.
The report's primary comparison values are continuation-turn input tokens,
cached input tokens, output tokens, tool calls, recall, and continuation
latency. Setup and validator time are diagnostics, not substitutes for token
usage. Provider-specific parsers may accept alternate names such as AGY's
`promptTokenCount`, `candidatesTokenCount`, and `totalTokenCount`, but they must
leave unavailable telemetry as `null` rather than estimate it.

## Goals

The integration benchmark answers these questions:

1. Can the client recall facts established in an earlier session?
2. Does Lint-AI reduce repeated repository exploration?
3. Does Lint-AI preserve the latest decision when earlier decisions are
   superseded?
4. Does bounded transcript parsing preserve memory after oversized tool
   results?
5. Does a negative-control query avoid injecting unrelated memory?
6. What token, tool-call, hook, setup, and continuation costs are added?
7. Does the MCP server work independently of whether the model elects to call
   it?

The benchmark is intended for engineering comparisons and regression
detection. It is not sufficient for general claims about all repositories,
models, or workloads without repeated randomized runs.

## Directory Layout

```text
benchmark/
  integration/
    README.md                 This cross-client methodology
  claude_code/
    scenarios/                Claude scenario definitions
    schemas/                  Scenario and result schemas
    scripts/run_benchmark.py  Claude launcher
    src/parse_run.py          Claude transcript metrics parser
    tests/                    Parser tests
    results/                  Generated Claude reports
  codex_code/
    scenarios/                Codex scenario definitions
    schemas/                  Scenario and result schemas
    scripts/run_benchmark.py  Codex launcher
    src/runner.py             Shared orchestration and runner
    src/parse_run.py          Codex transcript metrics parser
    src/scorer.py             Fact scoring
    src/report.py             Report aggregation and comparisons
    tests/                    Runner, parser, scorer, and report tests
    results/                  Generated Codex reports
  agy/
    scenarios/                AGY scenario definitions
    schemas/                  Scenario and result schemas
    scripts/run_benchmark.py  AGY launcher
    src/parse_run.py          AGY stream-JSON metrics parser
    tests/                    Parser tests
    results/                  Generated AGY reports
```

Client-specific details and command-line options remain in:

- [`benchmark/claude_code/README.md`](../claude_code/README.md)
- [`benchmark/codex_code/README.md`](../codex_code/README.md)
- [`benchmark/agy/README.md`](../agy/README.md)
- [`docs/claude-code-performance-tests.md`](../../docs/claude-code-performance-tests.md)
- [`docs/codex-performance-tests.md`](../../docs/codex-performance-tests.md)

## Test Model

Every run has two logical phases.

### Setup phase

The runner creates a disposable worktree at the exact scenario revision and
starts a new client session. The setup messages establish facts that should be
captured into durable memory. The setup turn is allowed to perform normal
client work and lifecycle capture.

The setup phase measures setup and capture behavior, but it is not the primary
latency comparison. Setup cost can include initial repository indexing,
authentication, model startup, hook execution, and session persistence.

### Continuation phase

The runner starts a fresh client session against the same disposable worktree
and asks the continuation question. The client must answer without asking the
user to repeat the setup discussion. The continuation phase is the primary
interaction comparison because it measures the cost of retrieving and using
previously captured memory.

The reported interaction latency starts when the continuation request is sent
and ends when the continuation run completes. Setup latency and validator
latency are retained as diagnostics and must not be substituted for the
continuation metric.

### Isolation

Each arm receives:

- a disposable Git worktree;
- an isolated client home/configuration directory;
- the same repository revision;
- the same scenario prompts;
- the same model setting;
- the same validator commands; and
- a fresh session for setup and continuation.

The host checkout is not used as the test worktree. Host sessions, caches,
Memories, plugins, and unrelated configuration must not leak into a run.
Authentication may be copied into the isolated client home when required by
the provider launcher.

AGY is the exception: its documented launcher must reuse the authenticated host
profile/keychain because a fresh AGY profile does not provide the same runtime
behavior. The AGY launcher temporarily replaces the host AGY settings, hooks,
and MCP configuration, resets them before each arm, and restores the originals
on exit. AGY benchmark runs therefore require exclusive use of AGY, and the
launcher must not be force-killed before cleanup.

## Comparison Arms

The standard hooks-only comparison uses three logical roles per client:
**Native** (the client's own built-in memory, no Lint-AI at all), **Lint-AI
only** (Lint-AI hooks, client-native memory explicitly disabled where the
client has one), and **Both** (client-native memory and Lint-AI hooks
enabled together). Each client maps these roles onto its own `--arms` names:

| Role | Claude arm | Codex arm | AGY arm | Native client memory | Lint-AI hooks | Lint-AI MCP |
| --- | --- | --- | --- | --- | --- | --- |
| Native | `claude-native` | `codex-native` | `agy-native` | Enabled (Claude auto-memory / Codex Memories) | Disabled | Disabled |
| Lint-AI only | `claude-lint-ai` | `lint-ai` | `agy-lint-ai` | Disabled | Enabled | Disabled |
| Both | `claude-both` | `lint-ai-with-codex-memory` | n/a | Enabled | Enabled | Disabled |
| Matched-install disabled control | n/a | n/a | `agy-lint-ai-disabled` | Disabled | Installed but off (`disable_lint_ai`) | Disabled |
| MCP-only track | (`--profile mcp-only`) | n/a | `agy-mcp-only` | Disabled | Disabled (hooks removed) | Enabled |

AGY has no built-in "native memory" feature of its own to combine with
Lint-AI, so it has no `Both` arm; `agy-native` is Lint-AI-absent entirely
(`--agy-install` is not even run), and `agy-lint-ai-disabled` is a stricter
control than `agy-native` — Lint-AI is installed identically to `agy-lint-ai`
but toggled off via `disable_lint_ai`, isolating "installed but inactive"
from "never installed."

The MCP server is disabled for the hooks-only table intentionally. MCP adds
workspace code indexing and explicit code-search capability, which is a
different measurement from lifecycle memory overhead and can dominate
startup latency; it is measured separately by the MCP-only track (see "MCP
Test Track" below).

Concretely, each launcher configures its arms as follows:

- `claude-native`: Claude auto-memory enabled, no Lint-AI configuration at all.
- `claude-lint-ai`: Claude auto-memory disabled, Lint-AI hooks enabled, MCP excluded from `hooks-only` profile.
- `claude-both`: Claude auto-memory enabled *and* Lint-AI hooks enabled together.
- `codex-native`: `memories = true` in Codex's own `config.toml`, no Lint-AI hook installation (`--codex-install` is skipped entirely).
- `lint-ai`: `memories = false`, Lint-AI hooks installed and restricted to the Claude-equivalent lifecycle (see below).
- `lint-ai-with-codex-memory`: `memories = true` *and* Lint-AI hooks installed — the Codex analogue of `claude-both`.
- `agy-native`: no `--agy-install` step at all; pure AGY with nothing Lint-AI-related present.
- `agy-lint-ai`: Lint-AI installed and explicitly enabled (`enable_lint_ai`), MCP server config emptied so only hooks are active.
- `agy-lint-ai-disabled`: Lint-AI installed identically to `agy-lint-ai` but explicitly disabled (`disable_lint_ai`) — a matched-installation control.
- `agy-mcp-only`: Lint-AI installed with hook settings emptied (no hooks fire) and the MCP server config left in place — isolates the MCP path.

For the fair cross-client comparison, Lint-AI uses the Claude-equivalent
lifecycle events on Codex (`SessionStart`, `UserPromptSubmit`,
`UserPromptExpansion`, `PreCompact`, `Stop`, `SessionEnd`). Codex-only tool
and subagent hook events are excluded from that comparison so they do not
create extra retrieval opportunities that Claude structurally cannot have.

## Single-Turn vs. Multi-Turn Setup

A scenario's `setup_messages` array controls this, and the same rule applies
identically across Claude, Codex, and AGY:

- **One entry** → single-turn setup: one request/response establishes all
  facts before the continuation phase.
- **More than one entry** → multi-turn setup: the messages run as sequential
  turns inside the *same* session, one after another, before the
  continuation phase starts as its own fresh, separate session.

The optional `continuation_messages` array follows the identical rule for
the continuation phase (not currently used by any real scenario; only
exercised by a throwaway mechanism-verification scenario during development).

The delivery mechanism differs by client because their CLIs differ, but the
scenario-authoring contract is identical — a scenario author never needs to
know which mechanism a client uses:

- **Claude**: a single live process fed multiple JSON user-turn lines via
  `--input-format stream-json`, so all turns share one real `session_id`.
- **Codex**: no equivalent streaming-input mode exists, so each turn is a
  separate `codex exec` invocation, threaded together by extracting the
  prior turn's `thread_id` and passing `codex exec resume <thread_id>` to
  the next one. See `run_resume_chain_phase` in
  `benchmark/codex_code/src/runner.py`.
- **AGY**: sequential turns executed via `agy` CLI in stream-JSON mode, chained
  together by passing the emitted conversation ID via `--conversation <id>` to
  subsequent turns in `run_resume_chain_phase`.

Both dispatch strategies live in the same shared `run_turn_phase` function in
`benchmark/codex_code/src/runner.py`, which every provider's launcher calls
for both the setup and continuation phases — the choice between them is
purely `metrics_mode` (provider) plus message-count, not separate code paths
per scenario.

Current real (non-throwaway) multi-turn scenarios:

- `routing-decision-supersession` (Claude, Codex, and AGY): 2 setup messages
  (initial proposal, then a superseding decision) — see "Current Scenarios"
  below.

All other current scenarios (`index-store-segmented-routing`,
`oversized-transcript-recovery`, `unrelated-query-negative-control`,
`codex-tool-use-retrieval`) are single-turn setup.

## Hooks Versus MCP

Hooks and MCP are different integration mechanisms. They can be enabled at the
same time, but a benchmark must identify which mechanism produced the memory
context.

| Concern | Lifecycle hooks | MCP server |
| --- | --- | --- |
| Invocation | Client lifecycle event invokes Lint-AI automatically | Model or client explicitly calls an MCP tool |
| Model choice | No model decision is required for retrieval/capture | Tool selection is model-controlled unless the prompt/client requires it |
| Retrieval timing | At prompt/session lifecycle events before or around model work | When `search` is selected during the model turn |
| Capture timing | Stop, compaction, and session-end lifecycle events | MCP itself does not automatically capture a conversation |
| Primary purpose | Durable session memory across agent turns | Explicit workspace and memory search |
| Typical data | Bounded conversation/session records | Current repository index plus persisted memory |
| Persistence | Writes provider-specific memory under `.lint-ai/` | Opens/synchronizes the provider-specific persistent index |
| Failure behavior | Should fail open and not block the client | Server/tool failure is reported to the MCP client |
| Benchmark track | Hooks-only A/B memory comparison | MCP availability, tool-use, and search performance |

### Hook lifecycle

The hook path is event-driven:

1. A client lifecycle event starts a short-lived Lint-AI hook process.
2. Retrieval hooks build a bounded query from the event payload and read the
   provider memory store.
3. Lint-AI returns bounded additional context to the client.
4. Capture hooks extract bounded conversational content, exclude tool-result
   blocks, and persist a compact structured memory record.
5. The next session can retrieve that record without requiring the model to
   call a tool.

This path is the correct one for measuring automatic memory overhead. The
hooks-only three-arm comparison disables the MCP server so repository indexing
and explicit code search do not contaminate the lifecycle-memory measurement.

### MCP lifecycle

The MCP path is a long-running JSON-RPC server:

1. The client starts the configured provider-specific Lint-AI process.
2. The server completes `initialize` and `tools/list`.
3. The persistent index is opened or initialized before store-dependent tools
   are served.
4. The model or client may call `search` with a query and `top_k`.
5. The server synchronizes newly captured provider memory, queries the shared
   retrieval stack, and returns ranked results with diagnostics.

MCP search is not automatic merely because the server is connected. A Claude
run can show a connected MCP server and still make zero `search` calls. The
installed skill guides Claude toward MCP search, but skills are guidance and
cannot guarantee tool selection. A direct MCP client test is required to prove
that the server's `search` implementation works.

### Why the tracks stay separate

Hooks and MCP have different cost centers:

- hooks add short-lived process launches, memory retrieval, capture, and
  additional context;
- MCP adds a long-running server, repository indexing or loading, JSON-RPC
  transport, explicit tool-call turns, and returned search results; and
- the model can add additional latency or tokens when it chooses local tools
  or MCP tools.

Consequently, the hooks-only table answers “what is the lifecycle memory cost?”
The MCP track answers “can the server start and search, and what does an agent
pay when it actually uses that tool?” Combining both into one percentage would
not identify which mechanism caused the difference.

## MCP Test Track

MCP must be measured separately from the hooks-only table.

### Agent MCP-only run

The Claude `mcp-only` profile keeps the Lint-AI MCP server enabled and removes
Lint-AI lifecycle hooks. It tests whether the client can discover and call the
server without automatic hook retrieval.

```bash
python3 benchmark/claude_code/scripts/run_benchmark.py \
  --profile mcp-only \
  --arms claude-lint-ai \
  --scenario index-store-segmented-routing \
  --results-dir benchmark/claude_code/results/mcp-only
```

MCP tool selection remains model-controlled. The installed
`.claude/skills/lint-ai-memory/SKILL.md` guides Claude to call
`mcp__lint-ai__search` for project-memory questions, but a skill cannot force
the model to select that tool.

Therefore, a report must distinguish these outcomes:

- **MCP server connected**: `initialize` and `tools/list` succeeded.
- **MCP tool called**: the transcript contains a call such as
  `mcp__lint-ai__search` or `mcp__lint-ai__info`.
- **MCP search performed**: the transcript contains an actual `search` call.
- **MCP server functional**: a direct MCP client successfully calls `search`.

A client can have a connected and functional MCP server while making zero
`search` calls. That is a model tool-selection result, not a server failure.

### Direct MCP validation

Use the provider health command to validate process startup, `initialize`, and
`tools/list`:

```bash
./lint-ai --claude-code-verify-mcp /path/to/repo --mcp-timeout-ms 30000
./lint-ai --codex-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

To validate the search path, use an MCP client to call:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "project memory and prior decisions",
      "top_k": 3
    }
  }
}
```

Record server startup, index initialization, query latency, document count,
result count, and returned document IDs. Do not combine these direct-server
measurements with agent continuation latency.

## Scenario Contract

Each scenario is a JSON document validated against the provider schema. The
contract includes:

| Field | Meaning |
| --- | --- |
| `schema_version` | Scenario format version |
| `id` | Stable lowercase scenario identifier |
| `category` | Workload class used in reports |
| `description` | Human-readable workload description |
| `negative_control` | Whether unrelated-memory suppression is the primary signal |
| `repository` | Fixture path and exact revision selector |
| `setup_messages` | Prompts that establish durable facts |
| `continuation_prompt` | Fresh-session question |
| `expected_facts` | Fact patterns required in the answer |
| `forbidden_facts` | Fact patterns that invalidate or weaken the answer |
| `validators` | Commands run inside the disposable worktree |
| `limits` | Timeout, turn, budget, and context limits |

Facts are scored using `match_any` text patterns. A scenario succeeds only when
all expected facts are found, no forbidden facts are found, and the client run
and validators complete successfully. A valid answer can still be marked
invalid if it includes a forbidden superseded or unrelated fact.

Validators check the repository implementation independently of the model
answer. They are not run against the developer's checkout.

## Current Scenarios

### `index-store-segmented-routing` (Claude, Codex, and AGY)

Category: `decision-recall`. **Single-turn setup** (1 `setup_messages`
entry).

Setup establishes that adapters use `IndexStore`, segmented memory uses
`SegmentRoutingStrategy::LocalDistinctiveness`, and a global fallback index is
retained. The continuation asks which API and routing strategy are current and
why the global index remains.

Expected facts:

- `IndexStore` owns the memory snapshot and adapter operations.
- `LocalDistinctiveness` is the selected routing strategy.
- The global index supports persistence and fallback query paths.

Forbidden fact: the adapter should construct or own a bare `MemoryIndex`.

Files: `benchmark/claude_code/scenarios/index-store-segmented-routing.json`,
`benchmark/codex_code/scenarios/codex-index-store-segmented-routing.json`,
`benchmark/agy/scenarios/index-store-segmented-routing.json`.

This is the primary scenario for comparing memory recall, continuation
latency, and token use across clients.

### `oversized-transcript-recovery`

Category: `failure-avoidance`.

Setup establishes the failure caused by reading only the final 64 KiB of a
JSONL transcript when the tail begins inside an oversized tool-result record.
It also establishes the bounded recovery design.

Expected facts:

- A fixed tail can begin inside a partial JSONL record.
- The scan expands backward from 64 KiB up to 4 MiB.
- Only six useful conversation messages are retained.
- Tool-result blocks are excluded.

Forbidden fact: ingesting the entire transcript without a bound.

This scenario tests whether memory captures remain bounded while still
recovering useful conversation context.

### `routing-decision-supersession` (Claude, Codex, and AGY)

Category: `temporal-correction`. **Multi-turn setup** (2 sequential
`setup_messages`) — see "Single-Turn vs. Multi-Turn Setup" above.

Setup first establishes `SparseOverlap` in one turn, then supersedes it with
`LocalDistinctiveness` and `query_top_n = 3` in a second, sequential turn
within the same setup session. The continuation asks for the current
configuration and requires the answer to distinguish the old proposal from
the current decision.

Expected facts:

- `LocalDistinctiveness` is current.
- `query_top_n` is 3.
- `SparseOverlap` is explicitly superseded.

Forbidden fact: presenting `SparseOverlap` as the current strategy.

Files: `benchmark/claude_code/scenarios/routing-decision-supersession.json`,
`benchmark/codex_code/scenarios/codex-routing-decision-supersession.json`,
`benchmark/agy/scenarios/routing-decision-supersession.json`.

This scenario tests recency, revision, and conflict handling rather than
simple keyword recall.

### `unrelated-query-negative-control` (Claude)

Category: `negative-control`.

Setup establishes routing memory, but the continuation asks only for the path
of the Python smoke benchmark.

Expected fact: `benchmark/python_smoke_benchmark.py`.

Forbidden facts: unrelated routing terms such as `LocalDistinctiveness` or
`query_top_n`.

The primary signal is that Lint-AI does not inject unrelated memory into an
unrelated answer. The scenario's `max_injected_context_bytes` is zero.

### `codex-tool-use-retrieval` (Codex)

Category: Codex tool-use retrieval.

The continuation asks which Codex hook events retrieve memory and which request
fields form the retrieval query. It checks Codex-specific `PreToolUse` and
`PostToolUse` behavior, bounded excerpts, and tool input metadata. It is not a
direct Claude comparison because Claude and Codex expose different hook
payloads.

## Metrics

### Recall and validity

- `expected_fact_ids_found`: expected facts matched in the final answer.
- `forbidden_fact_ids_found`: forbidden facts matched in the final answer.
- `recall`: expected facts found divided by expected fact count.
- `success`: all expected facts found, no forbidden facts found, and the run
  completed validly.
- `invalid_reason`: infrastructure, timeout, validator, parsing, or scoring
  failure reason.

Recall is answer-level fact recall. It is not the same as retrieval-engine
recall from the LongMemEval benchmark.

### Latency

Report these separately:

- setup duration;
- continuation or interaction-round duration;
- hook retrieval duration;
- hook capture duration;
- MCP startup and initialize duration;
- MCP index initialization duration;
- MCP query duration; and
- validator duration.

For apples-to-apples memory-layer comparisons, use continuation or interaction
round duration. Do not compare setup time from one arm with continuation time
from another arm.

### Tokens and tools

Record at least:

- input tokens;
- cached input tokens or cache-read tokens;
- output tokens;
- total tokens, without double-counting cached input;
- local tool calls by name;
- MCP tool calls by name;
- delegated-agent count and token usage; and
- injected context bytes.

The model may use repository tools instead of MCP. Tool-call totals must
therefore be split into local tools and actual `mcp__...` calls. A connected
MCP server with zero MCP calls is a valid observation.

## Running Tests

Validate scenario schemas and runner behavior before launching a live client:

```bash
python3 -m unittest discover -s benchmark/claude_code/tests -p 'test_*.py'
python3 -m unittest discover -s benchmark/codex_code/tests -p 'test_*.py'
python3 -m unittest discover -s benchmark/agy/tests -p 'test_*.py'
python3 benchmark/codex_code/src/runner.py \
  --root benchmark/codex_code \
  --validate-only \
  --repetitions 3
```

Run a Claude hooks-only smoke comparison:

```bash
python3 benchmark/claude_code/scripts/run_benchmark.py \
  --arms claude-native claude-lint-ai claude-both \
  --scenario index-store-segmented-routing \
  --repetitions 1 \
  --results-dir benchmark/claude_code/results/smoke
```

Run a Codex hooks-only smoke comparison:

```bash
python3 benchmark/codex_code/scripts/run_benchmark.py \
  --scenario codex-index-store-segmented-routing \
  --repetitions 1 \
  --results-dir benchmark/codex_code/results/smoke
```

Run an AGY hooks-only smoke comparison:

```bash
python3 benchmark/agy/scripts/run_benchmark.py \
  --arms agy-native agy-lint-ai \
  --scenario index-store-segmented-routing \
  --repetitions 1 \
  --results-dir benchmark/agy/results/smoke
```

For performance claims, use at least five repetitions per scenario and arm;
use ten or more before making release claims. Keep paired arms on the same
machine and use the same model, model settings, repository revision, and
network policy.

## Reports and Artifacts

Each run produces a report containing:

- benchmark and repository roots;
- selected scenarios and repetition count;
- per-run records;
- per-scenario scores;
- per-arm summary counts;
- token and tool metrics; and
- final comparison data when a baseline and candidate arm are both present.

The stable report locations are:

```text
benchmark/claude_code/results/<run-name>/.../report.json
benchmark/codex_code/results/<run-name>/report.json
```

Use `report.json` for summaries and the per-run `run-record.json`,
`continuation.metrics.json`, transcripts, and logs for diagnosis. A report
with only one arm can show raw metrics but cannot produce a valid percentage
comparison.

Do not commit raw transcripts, isolated client homes, memory stores, generated
patches, or temporary worktrees. Review aggregate reports for secrets and
secret-canary leakage before committing them.

## Interpreting Results

Interpret results in this order:

1. Check `status`, success count, invalid count, and timeout errors.
2. Check expected and forbidden fact scoring.
3. Confirm that the compared arms used the intended memory and MCP settings.
4. Compare continuation latency, not setup latency.
5. Compare uncached input, cached input, output, and total token fields
   separately.
6. Break down local tool calls, MCP calls, hooks, and delegated agents.
7. Inspect transcripts when a model selected repository tools instead of MCP.

Do not attribute a model's failure to call MCP to the server without a direct
MCP-client check. Do not attribute a latency difference to Lint-AI without
separating client startup, model generation, tool execution, hook work, MCP
startup, and index initialization.

## Reproducibility Checklist

Record the following with every performance run:

- client name and version;
- model identifier and model settings;
- Lint-AI revision and feature flags;
- repository revision and dirty-diff digest;
- operating system and architecture;
- scenario IDs and repetition count;
- benchmark profile and arm configuration;
- enabled hooks, MCP servers, skills, and plugins;
- authentication and network policy;
- warm or cold filesystem-cache state;
- timeout and budget limits; and
- result directory and report path.

If any of these differ between paired arms, label the comparison accordingly
and do not present it as an apples-to-apples result.
