# Codex Performance Evaluation

## Overview

Lint-AI integrates with Codex as a persistent memory layer. It captures
useful outcomes from one session and retrieves relevant memories when a later
session needs them. The expected benefit is less repeated repository
exploration without reducing answer quality.

This document defines the benchmark required before making general performance
claims about the Codex integration. The current Codex implementation is
functionally complete. A latest three-arm smoke result is recorded below;
additional repetitions are required before release claims.

## Latest Validated Smoke Result

The current three-arm orchestration was run on the
`index-store-segmented-routing` scenario with one repetition. Metrics below
cover the continuation session only; setup is the memory-seeding phase.

| Arm | Continuation | Input tokens | Cached input | Output tokens | Tool calls | Recall | Hook time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Codex native memory | 38.13 s | 151,000 | 99,840 | 1,505 | 5 | 2/3 | 0 ms |
| Lint-AI only | 18.53 s | 62,478 | 43,264 | 604 | 2 | 2/3 | 1.33 s |
| Both layers | 13.98 s | 31,684 | 25,088 | 454 | 1 | 2/3 | 1.39 s |

Compared with native memory, Lint-AI-only was `51%` faster and the combined
arm was `63%` faster in this run. Recall was equal across all three arms.
Lint-AI hook execution added approximately `1.3-1.4 s` to continuation time.

The setup phase must not be used as a latency comparison. Codex native and
combined setup consumed approximately `342k` and `347k` input tokens, while
Lint-AI-only consumed `921k`; the model performed more repository exploration
when Codex native memory was disabled. The native Codex memory file contained
no relevant consolidated rollout memory, while Lint-AI injected the captured
architecture decision during continuation.

This is one repetition and is diagnostic evidence, not a release benchmark.
Preserved report:

- `benchmark/codex_code/results/index-store-three-arms/report.json`

## MCP Validation Run

The Codex MCP path was validated separately with the same
`index-store-segmented-routing` scenario and one `lint-ai` arm:

| Arm | MCP status | Continuation | Input tokens | Cached input | Output tokens | Reported tool calls | Recall | Hook time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Codex + Lint-AI MCP | Connected | 9.38 s | 13,246 | 12,032 | 77 | 0 | 2/3 | 1.35 s |

The MCP server completed initialization and `tools/list`. The scenario used
Codex lifecycle hooks for memory retrieval and did not invoke an MCP
`tools/call`, so the reported zero tool calls is expected. This is a separate
MCP production-path validation, not a direct replacement for the three-arm
hooks-only comparison above.

Preserved report:

- `benchmark/codex_code/results/mcp-production-fixed/report.json`

## Benchmark Goals

## MCP Performance Track

The current three-arm results intentionally remove the Lint-AI MCP server so
they measure lifecycle memory-hook behavior only. MCP should be measured in a
separate full-integration track because it adds repository code-search
functionality and may build or load a repository index at startup.

Use this additional matrix:

| Arm | Codex memory | Lint-AI hooks | Lint-AI MCP |
| --- | --- | --- | --- |
| Native | Enabled | Disabled | Disabled |
| Lint-AI hooks | Disabled | Enabled | Disabled |
| Lint-AI full | Disabled | Enabled | Enabled |
| Combined full | Enabled | Enabled | Enabled |

Record these MCP-specific metrics independently from continuation latency:

- cold MCP startup and index-build time
- first MCP query latency
- warm MCP query latency
- MCP tool-call count and injected result bytes
- continuation latency after MCP use
- memory recall and repository code-search accuracy

Report cold and warm runs separately. Do not combine MCP-enabled results with
the hooks-only tables above, because MCP measures additional repository search
capability rather than only memory-layer overhead.

The benchmark is designed to answer four questions:

1. Does retrieved memory improve task accuracy across Codex sessions?
2. Does it reduce repeated repository exploration and user restatement?
3. Does it reduce total model tokens after accounting for injected context?
4. What latency does capture, retrieval, indexing, and MCP synchronization add?

## Non-Goals

- Comparing Codex models or Codex releases.
- Comparing Lint-AI with unrelated coding agents.
- Using retrieval benchmark scores alone as evidence of agent improvement.
- Optimizing Rust microbenchmarks without measuring end-to-end Codex tasks.
- Persisting private transcripts as benchmark artifacts.

## Experiment Arms

The current launcher runs three memory arms and can be extended with the
diagnostic arms below.

| Arm | Hooks | MCP | Persisted memory |
| --- | --- | --- | --- |
| Codex-native | No Lint-AI hook commands | No Lint-AI MCP | Codex Memories enabled |
| Lint-AI | Enabled | Excluded from hooks-only benchmark | Seeded by the setup session |
| Lint-AI with Codex memory | Enabled | Excluded from hooks-only benchmark | Both layers enabled |
| Baseline | Disabled | Disabled | None |
| Oracle | Disabled | Disabled | Relevant facts inserted into the prompt |

The oracle arm estimates the best result available when the correct memory is
known. It separates retrieval failures from model reasoning failures.

Run order must be randomized. Each arm uses a fresh Codex session, fresh
working copy, and isolated Codex configuration and Lint-AI store. Do not reuse
model context or generated files across arms.

## Apples-to-Apples Comparison

The current apples-to-apples comparison is `Codex-native` versus `Lint-AI`.
Both use the same model, prompts, validators, fresh sessions, and isolated
configuration. Codex-native enables Codex Memories and has no Lint-AI hook
commands or MCP server. Lint-AI disables Codex Memories and enables only the
Lint-AI hooks and MCP server. This measures the two competing memory layers
without stacking them.

For direct comparison with the Claude benchmark, the Lint-AI Codex arm uses
only the Claude-equivalent lifecycle events: `SessionStart`,
`UserPromptSubmit`, `UserPromptExpansion`, `PreCompact`, `Stop`, and
`SessionEnd`. Tool and subagent hooks are excluded from this benchmark profile.

The clean comparison matrix is:

| Arm | Codex hooks | Codex Memories | Lint-AI |
| --- | --- | --- | --- |
| Baseline | Disabled | Disabled | Disabled |
| Codex-only | Disabled | Enabled | Disabled |
| Lint-AI only | Enabled | Disabled | Enabled |
| Combined | Enabled | Enabled | Enabled |

Use Codex-native against Lint-AI to compare the two memory implementations. A
no-memory baseline and combined arm remain useful diagnostic measurements, but
are not the primary comparison in the current launcher.

## Workload

### Scenario Structure

Every scenario has two phases:

1. A setup session establishes durable facts through normal repository work.
2. A continuation session asks Codex to act without restating those facts.

Each scenario defines:

- repository and pinned revision
- setup prompt sequence
- expected memory facts
- continuation prompt
- deterministic validation command
- forbidden assumptions
- relevant files and expected tool operations
- maximum wall-clock time and turn count

### Scenario Categories

The initial suite should contain at least three scenarios in each category:

- Decision recall: preserve an architectural choice and its rationale.
- Unresolved work: continue from a known blocker or incomplete task.
- Failure avoidance: remember a failed approach and avoid repeating it.
- Repository navigation: recall relevant modules and reduce rediscovery.
- Temporal correction: prefer a newer decision over superseded memory.
- Segment isolation: retrieve the right session without leaking unrelated
  session facts.

Add Codex-specific tool-use coverage that the Claude suite does not need:

- Tool-use recall: use `PreToolUse`, `PermissionRequest`, or `PostToolUse`
  context to recover the immediately relevant implementation detail.
- Subagent handoff: verify that `SubagentStart` retrieval and `SubagentStop`
  capture do not leak unrelated session memory.

Include negative queries where no memory is relevant. Lint-AI should inject no
context or only context below a defined relevance threshold.

## Controlled Environment

Pin and record:

- Codex version
- model identifier
- Lint-AI revision and feature set
- operating system and architecture
- repository revision
- Codex settings and permission mode
- enabled tools and MCP servers
- network policy
- warm or cold filesystem-cache state

Use the same machine for paired runs when possible. Disable unrelated hooks,
plugins, and MCP servers. Runs that encounter rate limiting, service errors, or
interactive human intervention are invalid and must be repeated.

The recommended launcher keeps the temp tree on disk for inspection instead of
deleting it on exit:

```bash
python3 benchmark/codex_code/scripts/run_benchmark.py \
  --repetitions 1 \
  --results-dir benchmark/codex_code/results
```

That command preserves the temporary worktree, logs, and `report.json`, and it
also copies the final report into `benchmark/codex_code/results/report.json`.

Model output is nondeterministic. Run each scenario and arm at least five times.
Use ten or more repetitions before making release claims.

## Metrics

### Primary Metrics

**Task success rate**

The deterministic validator passes. Examples include tests, exact output
checks, patch assertions, and structured answer keys. Human judgment may be a
secondary score but cannot be the only validator.

**Memory fact accuracy**

Score expected facts as correct, missing, contradicted, or unsupported. Report
precision, recall, and exact scenario success. Unsupported details count
against precision.

**Total model tokens**

Sum input, cache-read, cache-write, and output tokens across the continuation
session. Report each component separately and report injected Lint-AI context
bytes and estimated tokens. Missing telemetry is `null`, never zero.

Report parent-session and all-model token components separately. Derive
subagent tokens as the non-negative all-model minus parent-session difference
only when the transcript contains a subagent launch. Otherwise report the
difference as unattributed non-parent usage.

**End-to-end latency**

Measure from continuation prompt submission until the final answer or task
completion. Report median, p90, and p95 across repetitions.

### Secondary Metrics

- time to first assistant response
- number of model turns
- tool calls by tool name
- files read and total bytes read
- repeated file reads
- repeated shell commands
- user clarification requests
- hook capture and retrieval latency
- MCP search and memory synchronization latency
- retrieved result count and selected segment count
- additional-context bytes
- exact-revision memories retrieved
- subagent count, type, purpose, resolved model, and token usage
- persistent store size and document count

Repeated exploration is a tool operation with the same normalized target and
equivalent arguments more than once in a continuation session. The parser must
retain both raw and normalized counts.

## Instrumentation

### Codex Transcript Parser

Parse Codex hook and session data into a versioned neutral event model:

```text
RunEvent
  timestamp
  session_id
  event_type
  model
  token_usage?
  tool_name?
  tool_input_digest?
  file_paths[]
  duration_ms?
```

The parser must validate the Codex version and fixture schema. Unknown events
are retained as raw metadata and counted. If token fields are absent or change
shape, token metrics for that run are unsupported.

Do not copy transcript text into benchmark results. Store digests, counts,
timestamps, tool names, paths relative to the fixture repository, and scored
fact identifiers.

The initial parser can be implemented under a Codex-specific benchmark
directory such as `benchmark/codex_code/src/parse_run.py`. It should recognize
Codex hook events, store only a digest of delegated prompts, count
`hookSpecificOutput.additionalContext` bytes once, and count memories marked
with an exact-revision status. Its measurements are post-run observations and
do not alter benchmark tool availability or Codex behavior.

The live runner persists the parser output as a per-continuation metrics
artifact, and reports consume that artifact rather than independently parsing
raw Codex output. Its field names and token accounting match the Claude parser;
Codex cached input is normalized into uncached input plus cache-read input
before totals are calculated.

### Lint-AI Measurements

Add structured measurements around these boundaries:

- hook JSON received
- transcript extraction completed
- `IndexStore::refresh` completed
- retrieval query completed
- MCP memory synchronization completed
- hook response written

Each measurement includes run ID, session ID hash, event name, elapsed
microseconds, document count, result count, context bytes, and segment count.
Write measurements to a harness-selected JSONL path. Production logging remains
off unless explicitly configured.

### Wall-Clock Measurement

The harness records monotonic start and completion times around the Codex CLI
process. Transcript timestamps are used for event ordering, not as the sole
latency source.

## Harness Layout

The proposed test-only layout is:

```text
benchmark/codex_code/
  README.md
  scenarios/
    <scenario-id>.json
  fixtures/
    <repository metadata or setup scripts>
  src/
    runner
    transcript_parser
    scorer
    report
  schemas/
    scenario.schema.json
    result.schema.json
  results/
    .gitkeep
```

Large fixture repositories, raw Codex transcripts, generated patches, and
result files containing project content must not be committed.

## Execution Protocol

For each scenario repetition:

1. Create an isolated repository worktree at the pinned revision.
2. Create isolated Codex configuration and Lint-AI memory directories.
3. Run the setup session and validate that required facts were established.
4. For the Lint-AI arm, verify expected capture documents and segments.
5. Reset generated repository changes required by the scenario definition.
6. Start a fresh continuation session.
7. Run the continuation prompt with a fixed timeout.
8. Execute the deterministic task validator.
9. Parse Codex and Lint-AI event data.
10. Redact and write one structured result record.
11. Destroy the isolated configuration, transcript, and memory directories.

Baseline setup sessions still run so that elapsed time and repository effects
match, but their captured memory is unavailable to the continuation session.

## Analysis

Use paired comparisons by scenario and repetition. Report raw per-scenario
results in addition to aggregate values.

- Accuracy: paired success-rate difference with a bootstrap confidence
  interval.
- Tokens and latency: median paired difference and percentile distributions.
- Tool calls: paired difference and repeated-operation rate.
- Retrieval: recall at the injected context boundary and irrelevant-injection
  rate.

Do not average successful and timed-out latency without reporting timeout rate.
Do not discard failed Lint-AI runs. Infrastructure failures are classified
separately from task failures using predefined rules.

## Initial Release Gates

The first release candidate should satisfy all of the following on the fixed
suite:

- no statistically credible reduction in task success
- at least a 10 percentage-point success improvement on memory-dependent tasks,
  or the confidence interval supports a positive improvement
- median total continuation tokens do not increase by more than 5%
- median repeated exploration decreases by at least 15%
- median retrieval-hook latency is below 100 ms
- p95 retrieval-hook latency is below 300 ms
- p95 capture-hook latency is below 1 second
- irrelevant context is injected in fewer than 5% of negative-query runs
- zero secret-canary values appear in persisted documents or result artifacts

These thresholds are provisional. Change them only before viewing a release
candidate's results, and record the rationale in this document.

## Security and Privacy Validation

Every run includes synthetic canaries for API keys, bearer tokens, passwords,
and PEM private keys. After capture, scan:

- `.lint-ai/codex-memory`
- Lint-AI measurement JSONL
- benchmark result records
- generated reports

The test fails if a canary appears in any durable artifact. Raw Codex
transcripts stay in the isolated Codex directory and are deleted after metric
extraction.

## Reporting

The generated report includes:

- environment and exact revisions
- scenario inventory and repetition count
- final pass/fail status, success rate, phase timing averages, and per-scenario results
- invalid and timed-out runs
- per-arm primary and secondary metrics
- paired differences and confidence intervals
- retrieval misses and irrelevant injections
- hook latency distributions
- secret-canary scan result

Reports must clearly separate measured results from interpretation. A report
must not claim that Lint-AI reduces tokens or latency when the corresponding
telemetry was missing.

## Implementation Phases

1. Implement schemas, transcript fixtures, and the versioned parser.
2. Add hook/MCP measurement output behind a test-only environment variable.
3. Implement three deterministic smoke scenarios and local reporting.
4. Add randomized repeated A/B execution.
5. Expand to the full category matrix and establish baseline distributions.
6. Run release candidates and publish only reproducible aggregate reports.
