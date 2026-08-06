# Claude Code Performance Benchmarks

This directory contains test-only A/B scenarios for measuring Lint-AI as a
Claude Code memory layer. The methodology and release gates are defined in
[`docs/claude-code-performance-tests.md`](../../docs/claude-code-performance-tests.md).
The cross-client integration benchmark methodology is defined in
[`benchmark/integration/README.md`](../integration/README.md).

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
developer's checkout. The Claude launcher reuses the shared orchestration in
`benchmark/codex_code/src/runner.py`.

Run one or more matching scenarios and memory arms with:

```bash
python3 benchmark/claude_code/scripts/run_benchmark.py \
  --arms claude-native claude-lint-ai claude-both \
  --scenario index-store-segmented-routing \
  --results-dir benchmark/claude_code/results/shared-orchestration
```

The report's `average_interaction_round_latency_ms` measures only the
continuation prompt. The launcher runs three arms: Claude native memory only,
Lint-AI memory with Claude auto-memory disabled, and both memory layers
enabled. The Lint-AI arms use hooks and durable memory without the MCP
code-index server, matching the Codex hooks-only latency comparison.

To validate MCP with Lint-AI lifecycle hooks disabled, use the dedicated
MCP-only profile:

```bash
python3 benchmark/claude_code/scripts/run_benchmark.py \
  --profile mcp-only \
  --arms claude-lint-ai \
  --scenario index-store-segmented-routing \
  --results-dir benchmark/claude_code/results/mcp-only
```

This keeps the Lint-AI MCP server enabled and removes only Lint-AI hook
commands from the isolated Claude settings. MCP tool selection remains
model-controlled. The installed `lint-ai-memory` skill guides Claude to call
`mcp__lint-ai__search` for prior project context, but it does not guarantee
that Claude will select the tool. In the recorded MCP-only run Claude called
`info` but did not call `search`; use an explicit MCP-required prompt or a
direct MCP client when measuring search/retrieval performance.

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
