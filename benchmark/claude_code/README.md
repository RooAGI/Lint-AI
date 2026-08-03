# Claude Code Performance Benchmarks

This directory contains test-only A/B scenarios for measuring Lint-AI as a
Claude Code memory layer. The methodology and release gates are defined in
[`docs/claude-code-performance-tests.md`](../../docs/claude-code-performance-tests.md).

## Layout

```text
benchmark/claude_code/
  scenarios/       Versioned workload definitions
  schemas/         JSON schemas for scenarios and results
  fixtures/        Fixture setup metadata and scripts
  src/             Run parser and future runner, scorer, and reporter
  scripts/         Local orchestration entry points
  results/         Generated results; ignored except for .gitkeep
```

The initial scenarios use the Lint-AI repository itself as the fixture. A
runner must resolve `repository.revision` to an exact commit and record a dirty
diff digest before execution. It must use a disposable worktree rather than the
developer's checkout.

A one-pair smoke result and its limitations are recorded in the performance
test design. It validated cross-session recall but is not a release benchmark.

## Parse Run Metrics

Extract parent, all-model, and delegated-agent token usage plus hook context
measurements after a run. This does not change Claude's prompts or tool policy:

```bash
python3 benchmark/claude_code/src/parse_run.py \
  --result /path/to/claude-result.json \
  --transcript /path/to/session.jsonl \
  --out /path/to/metrics.json
```

Delegation records retain the Agent description and a SHA-256 prompt digest,
not the raw delegated prompt. When no Agent launch is observed, token usage
outside the parent session is reported as unattributed rather than subagent
usage.

## Scenario Contract

Each scenario contains setup messages that establish durable memory and a
continuation prompt that runs in a fresh Claude session. Expected and forbidden
facts provide deterministic scoring targets. Commands in `validators` run only
inside the disposable fixture worktree.

`negative_control` scenarios intentionally contain no relevant memory. Their
primary signal is whether Lint-AI avoids injecting unrelated context.

## Current Scenarios

- `index-store-segmented-routing.json`: architectural decision recall.
- `oversized-transcript-recovery.json`: failure and fix recall.
- `routing-decision-supersession.json`: temporal correction.
- `unrelated-query-negative-control.json`: irrelevant-memory suppression.

## Generated Data

Do not commit Claude transcripts, isolated configuration, memory stores,
generated patches, or per-run result files. Aggregate reports may be committed
later after privacy review and secret-canary validation.
