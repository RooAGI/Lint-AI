# Codex Performance Benchmarks

This directory contains test-only A/B scenarios for measuring Lint-AI as a
Codex memory layer. The methodology and release gates are defined in
[`docs/codex-performance-tests.md`](../../docs/codex-performance-tests.md).
The cross-client integration benchmark methodology is defined in
[`docs/benchmark-integration.md`](../../docs/benchmark-integration.md).

## Layout

```text
benchmark/codex_code/
  scenarios/       Versioned workload definitions
  schemas/         JSON schemas for scenarios and results
  fixtures/        Fixture setup metadata and scripts
  src/             Run parser and future runner, scorer, and reporter
  scripts/         Local orchestration entry points
  results/         Generated results; ignored except for .gitkeep
```

The initial scenarios should use the Lint-AI repository itself as the fixture.
A runner must resolve `repository.revision` to an exact commit and record a
dirty diff digest before execution. It must use a disposable worktree rather
than the developer's checkout.

## Scenario Contract

Each scenario should contain setup messages that establish durable memory and a
continuation prompt that runs in a fresh Codex session. Expected and forbidden
facts provide deterministic scoring targets. Commands in `validators` should
run only inside the disposable fixture worktree.

`negative_control` scenarios intentionally contain no relevant memory. Their
primary signal is whether Lint-AI avoids injecting unrelated context.

## Current Scenarios

- `codex-index-store-segmented-routing.json`: architectural decision recall (single-turn).
- `codex-routing-decision-supersession.json`: temporal correction over a superseded proposal (multi-turn setup).
- `codex-tool-use-retrieval.json`: Codex tool-use hook retrieval.
- `codex-oversized-transcript-recovery.json`: transcript parsing failure and fix recall.

The overlapping architecture and transcript scenarios use the same scenario IDs
and prompts as the Claude suite, while keeping adapter-specific Rust validators.

The Codex benchmark scaffold now mirrors the Claude benchmark shape closely:

- schema files exist so Codex scenarios can be validated before the runner is
  implemented
- the layout matches the Claude benchmark structure
- these scenarios provide the initial smoke set for Codex-specific behavior

The launcher performs three comparisons: `codex-native`, with Codex Memories
enabled and no Lint-AI configuration; `lint-ai`, with Codex Memories disabled
and Lint-AI memory hooks enabled; and `lint-ai-with-codex-memory`, with both
memory layers enabled.
Both arms use the same model, prompts, fresh sessions, and validators.
For this comparison, the Lint-AI arm limits its Codex hook configuration to
the Claude-equivalent lifecycle: `SessionStart`, `UserPromptSubmit`,
`UserPromptExpansion`, `PreCompact`, `Stop`, and `SessionEnd`. Codex-only
tool and subagent hook events are excluded so they do not bias the comparison.
The launcher also removes Lint-AI's MCP server in this test configuration: it
builds a repository code index at process startup, which is a separate
code-search capability and would dominate memory interaction latency. Normal
Lint-AI installations retain the MCP server.

## Parse Run Metrics

Extract parent, all-model, and delegated-agent token usage plus hook context
measurements after a run. This does not change Codex prompts or tool policy:

```bash
python3 benchmark/codex_code/src/parse_run.py \
  --result /path/to/codex-result.json \
  --transcript /path/to/session.jsonl \
  --out /path/to/metrics.json
```

## Runner Scaffold

Discover and validate scenario files, then emit a normalized run plan:

```bash
python3 benchmark/codex_code/src/runner.py \
  --root benchmark/codex_code \
  --validate-only \
  --repetitions 3
```

The runner currently stops at validation and run-plan generation. It does not
yet launch Codex sessions or score live runs.

When `--execute` is provided, the runner also writes a summarized
`report.json` to the results directory unless `--report-out` is supplied.

To run the live Codex suite and keep the temp tree for inspection, use the
launcher script:

```bash
python3 benchmark/codex_code/scripts/run_benchmark.py \
  --repetitions 1 \
  --results-dir benchmark/codex_code/results
```

For an isolated diagnostic run, select the Lint-AI arm and one scenario:

```bash
python3 benchmark/codex_code/scripts/run_benchmark.py \
  --arms lint-ai \
  --scenario index-store-segmented-routing \
  --results-dir benchmark/codex_code/results/diagnostic
```

This preserves the temp worktree and writes a stable copy of the final report
to `benchmark/codex_code/results/report.json`. The report's
`summary.final_results` section is the top-level result to use when reviewing a
run: its primary latency field is `interaction_round_latency_ms`, measured
only for the continuation prompt. Setup and validator timings are retained as
diagnostics and must not be used for latency comparisons. `runner.json`
remains available for detailed raw run records.

The launcher intentionally does not use `codex exec --ephemeral`: the setup
turn's transcript must persist for the `Stop` and `SessionEnd` hooks to store
memory for the continuation. Each arm uses a disposable `CODEX_HOME`, so this
persistence cannot affect normal user data.

Only `~/.codex/auth.json` is copied into that home. The benchmark creates its
own config and hook settings and does not import host sessions, Memories,
caches, or plugins.

Each executed continuation writes `continuation.metrics.json`. The report
consumes this normalized artifact, whose token, delegation, tool, hook, and
retrieval fields match the Claude parser. Codex cached input is converted to
separate uncached-input and cache-read fields so totals do not double-count it.

Each scenario/arm/repetition worktree persists after the run instead of being
deleted, so its `.lint-ai/codex-memory` store stays inspectable. A worktree
left over from a prior run of the same scenario/arm/repetition is removed
automatically right before the next run of it starts. Inspect a run's
captured memory with:

```bash
target/release/lint-ai \
  --inspect-index benchmark/codex_code/results/<arm>/<scenario>/rep-001/worktree/.lint-ai/codex-memory \
  --inspect-view source-documents
```

`--inspect-view` also accepts `summary`, `records`, and `segments`.

## Single-Turn vs. Multi-Turn Setup

A scenario's `setup_messages` array controls this: one message runs as a
single request/response (single-turn); more than one message chains
sequential turns into the same Codex session by threading `codex exec resume
<thread_id>` between per-message processes (multi-turn) — see
`run_resume_chain_phase` in `benchmark/codex_code/src/runner.py`. The same
applies to the optional `continuation_messages` array. This dispatch is
shared across all three provider harnesses (Claude, Codex, AGY); Claude uses
a single live `--input-format stream-json` process instead of resume-chaining
since it supports that natively, but the scenario-authoring contract
(message-array length) is identical.

`codex-index-store-segmented-routing.json` currently defines a single setup
message, so the results below are single-turn. A multi-turn variant has not
yet been added as a real scenario with scored facts.

### Latest single-scenario result (`index-store-segmented-routing`, single-turn)

One repetition each, `hooks-only`-equivalent restricted hook configuration
(see above), run 2026-08-16 after fixing the `UUID_RE` regex (previously
rejected Codex's UUIDv7 thread IDs) and loosening the `index-store-owner`
`match_any` patterns (previously too brittle to match either arm's phrasing):

| Metric | codex-native | lint-ai | lint-ai-with-codex-memory |
|---|---|---|---|
| Success | true | true | true |
| Recall | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Setup time | 10.2 s | 14.6 s | 16.9 s |
| Continuation time | 19.8 s | 12.0 s | 10.4 s |
| Input tokens (cached) | 63,451 (45,312) | 13,990 (11,008) | 14,515 (11,008) |
| Output tokens | 593 | 99 | 117 |
| Tool calls | 2 | 0 | 0 |

All three arms recalled all 3 expected facts with identical accuracy.
`lint-ai` and `lint-ai-with-codex-memory` both answered directly with 0 tool
calls versus native's 2, using roughly a fifth of native's input tokens and
noticeably lower continuation latency. Unlike Claude, where enabling both
memory layers together (`claude-both`) beat every other arm on every metric,
Codex's combined arm only marginally improved continuation latency over
plain `lint-ai` and was slightly worse on setup time and tokens — within the
noise of a single repetition, not a clear win. This is a
single-repetition, single-scenario result, not a release benchmark — treat
it as directional pending a multi-scenario, multi-repetition run.

### Latest multi-turn result (`routing-decision-supersession`, multi-turn)

One repetition each, run 2026-08-16. `setup_messages` has 2 entries (initial
proposal, then a superseding decision), so setup runs as two chained turns
via `codex exec resume <thread_id>` (see "Single-Turn vs. Multi-Turn Setup"
above). This scenario also required loosening the `query-top-three` and
`old-sparse-superseded` `match_any` patterns for the same reason as the
single-turn fix: models phrase facts (e.g. `query_top_n: 3` with a colon)
differently than the original literal-substring patterns expected.

| Metric | codex-native | lint-ai | lint-ai-with-codex-memory |
|---|---|---|---|
| Success | true | true | true |
| Recall | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Setup time | 8.5 s | 7.8 s | 6.9 s |
| Continuation time | 43.7 s | 6.9 s | 7.0 s |
| Input tokens (cached) | 216,750 (179,200) | 14,376 (11,008) | 14,411 (11,008) |
| Output tokens | 1,507 | 52 | 54 |
| Tool calls | 6 | 0 | 0 |

All three arms recalled all 3 expected facts, but the gap between native and
Lint-AI is far larger here than in the single-turn scenario: to correctly
distinguish the current decision from the superseded one, `codex-native`
needed 6 tool calls and 216k input tokens re-deriving the temporal history
from source, versus 0 tool calls and ~14k tokens for either `lint-ai` arm
answering directly from injected memory — roughly 15x fewer input tokens and
6x lower continuation latency. Temporal-correction/multi-turn scenarios
expose Lint-AI's advantage far more sharply than simple single-turn recall.
This is a single-repetition, single-scenario result, not a release
benchmark — treat it as directional pending a multi-scenario,
multi-repetition run.

## Generated Data

Do not commit Codex transcripts, isolated configuration, memory stores,
generated patches, or per-run result files. Aggregate reports may be committed
later after privacy review and secret-canary validation.
