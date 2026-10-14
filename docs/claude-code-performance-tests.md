# Claude Code Performance Evaluation

## Overview

Membrick integrates with Claude Code as a persistent memory layer. It captures
useful outcomes from one session and retrieves relevant memories when a later
session needs them. The expected benefit is less repeated repository
exploration without reducing answer quality.

This page reports a three-run comparison of one Claude memory-recall scenario.
Treat it as a result for this task and setup, not as a guarantee across
projects or workloads.

## Latest result: automatic Membrick context reduced token use

Claude recalled all three expected facts in every run. With Membrick providing
saved project context automatically, Claude used fewer tokens and made fewer
tool calls than with Claude's native memory alone.

!!! success "Claude native memory compared with automatic Membrick context"

    - **Fresh input tokens:** 128 → 32 (75.0% fewer)
    - **Overall tokens:** 2,884,478 → 605,686 (79.0% fewer)
    - **Overall tokens per fresh input token:** 22,535 → 18,928 (16.0% fewer)
    - **Agent tool calls:** 65 → 14 (78.5% fewer)
    - **Claude response turns:** 131 → 36 (72.5% fewer)

    Claude found 9/9 expected facts across the three runs in both setups. The
    response-turn count is the total number of Claude replies across those
    runs. It is not the number of benchmark repetitions.

Membrick retrieves relevant saved project context during the session and
adds it automatically. Claude can use that context while answering; it does
not need to call a Membrick tool. The benchmark's hooks are the mechanism that
triggers this automatic context. “Claude native memory only” uses Claude's
built-in memory, while “automatic Membrick context only” uses Membrick for
retrieval.

Overall tokens include fresh input, cache creation, cached input reads, and
output. Cached input reads count each time Claude processes them. “Overall
tokens per fresh input token” is overall tokens divided by fresh input tokens.
This was one scenario repeated three times in each setup.

## Raw reports for the three setups

The third setup enabled both Claude's native memory and automatic Membrick
context. It used 587,723 overall tokens, 13 agent tool calls, and found all
three facts in all three runs. The comparison with the native-only setup is
shown on the [Performance overview](performance.md#overview).

- Native: [machine-readable report](https://github.com/RooAGI/Membrick/blob/main/benchmark/claude_code/results/v0.5.0-provider-2026-10-14/claude-native/report.json)
- Membrick hooks only: [machine-readable report](https://github.com/RooAGI/Membrick/blob/main/benchmark/claude_code/results/v0.5.0-claude-hooks-only-rerun-2026-10-14/claude-membrick/report.json)
- Claude + Membrick: [machine-readable report](https://github.com/RooAGI/Membrick/blob/main/benchmark/claude_code/results/v0.5.0-claude-both-hooks-only-three-rounds-2026-10-14/claude-both/report.json)

## Benchmark Methodology

## MCP Performance Track

The current results intentionally exclude the Membrick MCP server so they
measure lifecycle memory-hook behavior only. MCP should be measured separately
because it adds repository code-search functionality and may build or load a
repository index at startup.

Use a separate full-integration matrix:

| Arm | Claude memory | Membrick hooks | Membrick MCP |
| --- | --- | --- | --- |
| Native | Enabled | Disabled | Disabled |
| Membrick hooks | Disabled | Enabled | Disabled |
| Membrick full | Disabled | Enabled | Enabled |
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

1. Does retrieved memory improve task accuracy across Claude sessions?
2. Does it reduce repeated repository exploration and user restatement?
3. Does it reduce total model tokens after accounting for injected context?
4. What latency does capture, retrieval, indexing, and MCP synchronization add?

The primary comparison is Claude Code with Membrick disabled versus the same
Claude Code version, model, repository state, prompts, and permissions with
Membrick enabled.

### Non-Goals

- Comparing Claude models or Claude Code releases.
- Comparing Membrick with unrelated coding agents.
- Using retrieval benchmark scores alone as evidence of agent improvement.
- Optimizing Rust microbenchmarks without measuring end-to-end Claude tasks.
- Persisting private transcripts as benchmark artifacts.

### Experiment Arms

Each scenario runs in two required arms and one optional diagnostic arm.

| Arm | Hooks | MCP | Persisted memory |
| --- | --- | --- | --- |
| Baseline | Disabled | Disabled | None |
| Membrick | Enabled | Enabled | Seeded by the setup session |
| Oracle | Disabled | Disabled | Relevant facts inserted into the prompt |

The oracle arm estimates the best result available when the correct memory is
known. It distinguishes retrieval failures from Claude reasoning failures.

Run order must be randomized. Each arm uses a fresh Claude session, fresh
working copy, and isolated `CLAUDE_CONFIG_DIR` and Membrick store. Do not reuse
model context or generated files across arms.

## Workload

### Scenario Structure

Every scenario has two phases:

1. A setup session establishes durable facts through normal repository work.
2. A continuation session asks Claude to act without restating those facts.

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

- **Decision recall:** preserve an architectural choice and its rationale.
- **Unresolved work:** continue from a known blocker or incomplete task.
- **Failure avoidance:** remember a failed approach and avoid repeating it.
- **Repository navigation:** recall relevant modules and reduce rediscovery.
- **Temporal correction:** prefer a newer decision over superseded memory.
- **Segment isolation:** retrieve the right session without leaking unrelated
  session facts.

Include negative queries where no memory is relevant. Membrick should inject no
context or only context below a defined relevance threshold.

## Controlled Environment

Pin and record:

- Claude Code version
- Claude model identifier
- Membrick revision and feature set
- operating system and architecture
- repository revision
- Claude settings and permission mode
- enabled tools and MCP servers
- network policy
- warm or cold filesystem-cache state

Use the same machine for paired runs when possible. Disable unrelated hooks,
plugins, and MCP servers. Runs that encounter rate limiting, service errors, or
interactive human intervention are invalid and must be repeated.

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
session. Report each component separately and report injected Membrick context
bytes and estimated tokens. Missing telemetry is `null`, never zero.

Report parent-session and all-model token components separately. Derive
subagent tokens as the non-negative all-model minus parent-session difference
only when the transcript contains an Agent delegation. Otherwise report the
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

### Claude Transcript Parser

Parse Claude's project JSONL into a versioned neutral event model:

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

The parser must validate the Claude Code version and fixture schema. Unknown
events are retained as raw metadata and counted. If token fields are absent or
change shape, token metrics for that run are unsupported.

Do not copy transcript text into benchmark results. Store digests, counts,
timestamps, tool names, paths relative to the fixture repository, and scored
fact identifiers.

The initial parser is `benchmark/claude_code/src/parse_run.py`. It recognizes
Agent tool-use events and their result metadata, stores only a digest of each
delegated prompt, counts `hook_additional_context` bytes once, and counts
memories marked `Revision status: exact-match`. Its measurements are post-run
observations and do not alter benchmark tool availability or Claude behavior.

### Membrick Measurements

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

The harness records monotonic start and completion times around the Claude CLI
process. Transcript timestamps are used for event ordering, not as the sole
latency source.

## Harness Layout

The proposed test-only layout is:

```text
benchmark/claude_code/
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

Large fixture repositories, raw Claude transcripts, generated patches, and
result files containing project content must not be committed.

## Execution Protocol

For each scenario repetition:

1. Create an isolated repository worktree at the pinned revision.
2. Create isolated Claude configuration and Membrick memory directories.
3. Run the setup session and validate that required facts were established.
4. For the Membrick arm, verify expected capture documents and segments.
5. Reset generated repository changes required by the scenario definition.
6. Start a fresh continuation session.
7. Run the continuation prompt with a fixed timeout.
8. Execute the deterministic task validator.
9. Parse Claude and Membrick event data.
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
Do not discard failed Membrick runs. Infrastructure failures are classified
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

- `.membrick/memory`
- Membrick measurement JSONL
- benchmark result records
- generated reports

The test fails if a canary appears in any durable artifact. Raw Claude
transcripts stay in the isolated Claude directory and are deleted after metric
extraction.

## Reporting

The generated report includes:

- environment and exact revisions
- scenario inventory and repetition count
- invalid and timed-out runs
- per-arm primary and secondary metrics
- paired differences and confidence intervals
- retrieval misses and irrelevant injections
- hook latency distributions
- secret-canary scan result

Reports must clearly separate measured results from interpretation. A report
must not claim that Membrick reduces tokens or latency when the corresponding
telemetry was missing.

## Implementation Phases

1. Implement schemas, transcript fixtures, and the versioned parser.
2. Add hook/MCP measurement output behind a test-only environment variable.
3. Implement three deterministic smoke scenarios and local reporting.
4. Add randomized repeated A/B execution.
5. Expand to the full category matrix and establish baseline distributions.
6. Run release candidates and publish only reproducible aggregate reports.
