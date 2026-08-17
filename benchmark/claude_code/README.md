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
code-index server, matching the Codex hooks-only latency comparison. The
`hooks-only` profile no longer installs the `lint-ai-memory` skill, since that
skill tells Claude to call `mcp__lint-ai__search`, a tool that does not exist
once MCP is stripped out of the isolated config — installing it there only
confused the model and inflated tokens/latency without helping recall.

Each scenario/arm/repetition worktree persists after the run instead of being
deleted, so its `.lint-ai/claude-memory` store stays inspectable. A worktree
left over from a prior run of the same scenario/arm/repetition is removed
automatically right before the next run of it starts. Inspect a run's
captured memory with:

```bash
target/release/lint-ai \
  --inspect-index benchmark/claude_code/results/<arm>/<scenario>/rep-001/worktree/.lint-ai/claude-memory \
  --inspect-view source-documents
```

`--inspect-view` also accepts `summary`, `records`, and `segments`.

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

### Latest single-scenario result (`index-store-segmented-routing`)

One repetition each, `hooks-only` profile, run 2026-08-16 after the SKILL.md
and worktree-cleanup fixes above:

| Metric | claude-native | claude-lint-ai | claude-both |
|---|---|---|---|
| Success | true | true | true |
| Recall | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Setup time | 32.8 s | 26.3 s | 21.8 s |
| Continuation time | 26.9 s | 11.9 s | 5.8 s |
| Input tokens (cached) | 164,242 (164,230) | 40,930 (40,926) | 16,654 (16,652) |
| Output tokens | 1,944 | 793 | 227 |
| Tool calls | 9 | 2 | 0 |

All three arms recalled all 3 expected facts with identical accuracy, but the
amount of work needed to get there differed sharply. Native Claude re-derived
the architecture decision via repeated file reads/searches (9 tool calls).
`claude-lint-ai` (Lint-AI memory, native auto-memory disabled) answered
directly from 2 tool calls, using about a quarter of native's input tokens
and roughly half its continuation latency. `claude-both` (Lint-AI memory and
Claude's native auto-memory both enabled) went further still: 0 tool calls,
the lowest latency, and the fewest tokens of the three — having two
independent memory sources available appears to reinforce recall rather than
add overhead or conflict. This is a single-repetition, single-scenario
result, not a release benchmark — treat it as directional pending a
multi-scenario, multi-repetition run.

### Latest multi-turn result (`routing-decision-supersession`, multi-turn)

One repetition each, `hooks-only` profile, run 2026-08-16. `setup_messages`
has 2 entries (initial proposal, then a superseding decision), so setup runs
as two sequential turns inside one live `claude --input-format stream-json`
process (see "Single-Turn vs. Multi-Turn Setup" in
[`benchmark/integration/README.md`](../integration/README.md)).

Getting a clean result here required fixing two shared scoring bugs (both in
`benchmark/codex_code/src/scorer.py` and `src/runner.py`, so they apply to
every provider, not just Claude): Claude's harness never wrote a
final-message-only file the way Codex's `--output-last-message` does, so
scoring fell back to the full raw transcript — including tool-result text
that could contain scenario/rubric wording verbatim and produce false
matches. `run_command` now extracts the final assistant message from
Claude's stream-json output into a `.last` file, mirroring Codex. Separately,
the `match_any` substring scorer now strips Markdown emphasis markers
(`` ` ``, `**`, `_`) before comparing, since model answers formatted as
`` **`query_top_n`:** `3` `` weren't matching a literal `query_top_n: 3`
pattern.

| Metric | claude-native | claude-lint-ai | claude-both |
|---|---|---|---|
| Success | true | true | true |
| Recall | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Forbidden fact leaked | none | none | none |
| Setup time | 24.5 s | 17.7 s | 19.9 s |
| Continuation time | 27.3 s | 8.4 s | 9.2 s |
| Input tokens (cached) | 159,336 (159,159) | 40,889 (40,885) | 42,910 (42,906) |
| Output tokens | 1,897 | 266 | 442 |
| Tool calls | 7 | 1 | 1 |

All three arms reach identical, perfect accuracy — no arm is "smarter" at
recall. The gap is entirely in how much work it takes to get there:
`claude-lint-ai` uses about a quarter of native's input tokens, is roughly
3x faster on continuation, and needs 7x fewer tool calls than
`claude-native`. `claude-both` performs almost identically to
`claude-lint-ai` here, suggesting Lint-AI's injected memory is doing the
work and Claude's own auto-memory adds little on top of it for this
scenario. This is a single-repetition, single-scenario result, not a
release benchmark — treat it as directional pending a multi-scenario,
multi-repetition run.

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

- `index-store-segmented-routing.json`: architectural decision recall (single-turn).
- `oversized-transcript-recovery.json`: failure and fix recall.
- `routing-decision-supersession.json`: temporal correction (multi-turn setup).
- `unrelated-query-negative-control.json`: irrelevant-memory suppression.

## Generated Data

Do not commit Claude transcripts, isolated configuration, memory stores,
generated patches, or per-run result files. Aggregate reports may be committed
later after privacy review and secret-canary validation.
