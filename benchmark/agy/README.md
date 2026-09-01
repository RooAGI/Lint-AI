# Antigravity CLI Integration Benchmark

This directory contains test-only A/B scenarios for measuring Lint-AI as an
Antigravity (AGY) memory layer. The methodology and release gates follow the
cross-client integration benchmark contract defined in
[`docs/benchmark-integration.md`](../../docs/benchmark-integration.md).

---

## Layout

```text
benchmark/agy/
  scenarios/       Versioned workload definitions
  schemas/         JSON schemas for scenarios and results
  fixtures/        Fixture setup metadata and scripts
  src/             Run parser and AGY-specific metrics normalizer
  scripts/         Local orchestration entry point
  results/         Generated results; ignored except for .gitkeep
```

---

## Quick Start

Run the shared-harness AGY arms with:

```bash
python3 benchmark/agy/scripts/run_benchmark.py \
  --arms agy-native agy-lint-ai agy-lint-ai-disabled agy-mcp-only \
  --scenario index-store-segmented-routing \
  --repetitions 1 \
  --timeout-scale 0.5 \
  --results-dir benchmark/agy/results/shared
```

To run a multi-turn temporal-correction scenario:

```bash
python3 benchmark/agy/scripts/run_benchmark.py \
  --arms agy-native agy-lint-ai \
  --scenario routing-decision-supersession \
  --repetitions 1 \
  --timeout-scale 0.5 \
  --results-dir benchmark/agy/results/multi-turn
```

### Available Arms

* **`agy-native`**: Baseline AGY without Lint-AI hooks or MCP configuration.
* **`agy-lint-ai`**: Full integration with Lint-AI memory injection, lifecycle hooks, and durable recording.
* **`agy-lint-ai-disabled`**: Hooks and session recording active, but memory injection toggled off via `disable_lint_ai` (isolates hook execution overhead).
* **`agy-mcp-only`**: Lint-AI MCP server registered with lifecycle hooks removed (isolates tool discovery/calling overhead).

---

## Model & Environment Configuration

* **Model**: **Gemini 3.7 Flash** (managed via the authenticated AGY host profile / keychain).
* **CLI Runtime**: **Antigravity CLI (`agy`)** using `--output-format stream-json` and `--disable-slash-commands`.
* **Provider Token Accounting**: Token counts are parsed directly from stream-JSON `usage` / `usageMetadata` records (`input_tokens`, `output_tokens`, `total_tokens`, `thinking_tokens`, `cache_read_tokens`) via [`src/parse_run.py`](src/parse_run.py).

### AGY configuration isolation

The launcher intentionally uses the authenticated host AGY profile and
keychain, because a fresh AGY profile does not provide the same runtime
behavior. It temporarily replaces the host AGY configuration files used by
the benchmark:

* `~/.gemini/config/hooks.json`; and
* `~/.gemini/config/mcp_config.json`.

The files are restored when the launcher exits, including normal failures.
Run the benchmark without using AGY concurrently, and do not force-kill the
launcher; a hard process kill can prevent restoration. The launcher resets the
temporary hook and MCP configuration before every arm, so stale Lint-AI
configuration cannot contaminate `agy-native` or another arm.

The benchmark does not modify AGY permissions or bypass permission checks.
This keeps the run representative of a normal AGY installation and avoids
granting host-wide command permission solely for benchmarking.

---

## Single-Turn vs. Multi-Turn Setup

A scenario's `setup_messages` array controls the continuous chat lifecycle:

* **Single-Turn Setup**: One message runs as a single request/response.
* **Multi-Turn Setup**: More than one message chains sequential turns into the same interactive AGY session by passing the reported conversation ID via `--conversation <conversation_id>` between per-message processes (handled by `run_resume_chain_phase` in `benchmark/codex_code/src/runner.py`).

### Cross-Session Continuation Boundary
To strictly measure cross-session memory retrieval (matching Claude Code and Codex harness methodology):
* **Setup Phase**: Runs the setup message sequence (single-turn or multi-turn). When setup terminates, Lint-AI's `Stop` hook extracts and indexes the full conversation transcript into `.lint-ai/agy-memory`.
* **Continuation Phase (Fresh Session)**: Continuation always executes as a **brand new session** with **no prior conversation history** passed in its prompt context.
  * In `agy-native`, the model receives only the continuation prompt without conversation history, requiring it to answer from pre-training or hallucinate.
  * In `agy-lint-ai`, the `PreInvocation` hook intercepts the continuation turn, queries `.lint-ai/agy-memory`, and injects the indexed setup decision as an `EPHEMERAL_MESSAGE`.

---

## Benchmark Results

### 1. Single-Scenario Result (`index-store-segmented-routing`, Single-Turn)

One repetition each on AGY with Gemini 3.7 Flash:

| Metric | `agy-native` | `agy-lint-ai` | Impact / Advantage |
| :--- | :---: | :---: | :--- |
| **Success** | true | true | Both passed validators |
| **Recall** | 1/3 (33.3%) | **3/3 (100.0%)** | **+66.7% Recall Improvement** |
| **Setup Time** | 6.7 s | 6.0 s | Comparable |
| **Continuation Time** | 7.4 s | **6.4 s** | **~1.0 s faster** |
| **Continuation Input Tokens** | 14,596 | **13,965** | **-631 input tokens** |
| **Continuation Output Tokens** | 351 | **100** | **-71.5% output tokens (concise & factual)** |

**Analysis**: Lacking prior conversation history in a fresh session, `agy-native` hallucinated memory APIs (`record_session`, `list_memories`) and routing architectures (`hierarchical, domain-partitioned routing`), only matching `global fallback` by coincidence. `agy-lint-ai` successfully injected the stored `IndexStore` and `SegmentRoutingStrategy::LocalDistinctiveness` decisions via `PreInvocation`, achieving 100% recall.

---

### 2. Multi-Turn Result (`routing-decision-supersession`, Multi-Turn)

One repetition each on AGY with Gemini 3.7 Flash. `setup_messages` has 2 entries (initial `SparseOverlap` proposal, then a superseding `LocalDistinctiveness` decision with `query_top_n = 3`), running as two chained turns via `--conversation <id>`:

| Metric | `agy-native` | `agy-lint-ai` | Impact / Advantage |
| :--- | :---: | :---: | :--- |
| **Success** | true | true | Both passed validators |
| **Recall** | 1/3 (33.3%) | **3/3 (100.0%)** | **+66.7% Recall Improvement** |
| **Forbidden Facts Triggered** | none (0) | none (0) | Clean boundary |
| **Setup Time** | 5.5 s | 6.0 s | Full 2-turn execution |
| **Continuation Time** | 8.5 s | **5.9 s** | **~2.6 s faster** |
| **Continuation Input Tokens** | 14,583 | 14,601 | Normalized usage |
| **Continuation Output Tokens** | 990 | **99** | **-90.0% output tokens** |

**Analysis**: In multi-turn continuous chat, `agy-lint-ai` accurately distinguished the active `LocalDistinctiveness` strategy and `query_top_n = 3` parameter from the superseded `SparseOverlap` proposal using only 99 output tokens, whereas `agy-native` required 990 output tokens and could not identify the current routing parameters across the session boundary.

---

## Memory Store Inspection

Each scenario/arm worktree persists after the run so its `.lint-ai/agy-memory` store remains inspectable. Inspect a run's captured memory with:

```bash
target/debug/lint-ai \
  --inspect-index benchmark/agy/results/<arm>/<scenario>/rep-001/worktree/.lint-ai/agy-memory \
  --inspect-view source-documents
```

`--inspect-view` also accepts `summary`, `records`, and `segments`.

---

## Local Protocol Latency Track

Measures raw overhead of the local MCP and hook execution paths:

| Measurement | Result |
| :--- | ---: |
| MCP requests / responses | 6 / 6 |
| MCP roundtrip | 1,013.2 ms |
| `PreInvocation` hook | 16.6 ms |
| `PreToolUse` hook | 17.0 ms |
| `PostToolUse` hook | 6.8 ms |
| `Stop` hook | 6.0 ms |
| Recorded lifecycle events | 4 |

---

## Current Scenarios

* **`index-store-segmented-routing.json`**: Architectural decision recall (single-turn setup).
* **`routing-decision-supersession.json`**: Temporal correction over a superseded proposal (multi-turn setup).

---

## Troubleshooting & Quota Handling

* **Authentication**: AGY authenticates through the user's host profile. The benchmark preserves user credentials while isolating configuration files.
* **Quota Exhaustion**: If AGY returns `Individual quota reached`, runs will exit before emitting telemetry. These runs are recorded as unavailable rather than integration failures. Retry once account quota resets.
