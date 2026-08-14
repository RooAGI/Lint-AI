# Codex Performance Benchmarks

This directory contains test-only A/B scenarios for measuring Lint-AI as a
Codex memory layer. The methodology and release gates are defined in
[`docs/codex-performance-tests.md`](../../docs/codex-performance-tests.md).
The cross-client integration benchmark methodology is defined in
[`benchmark/integration/README.md`](../integration/README.md).

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

- `codex-index-store-segmented-routing.json`: architectural decision recall.
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

## Generated Data

Do not commit Codex transcripts, isolated configuration, memory stores,
generated patches, or per-run result files. Aggregate reports may be committed
later after privacy review and secret-canary validation.
