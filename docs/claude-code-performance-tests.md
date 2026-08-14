# Claude Code Performance Evaluation

## Overview

Lint-AI integrates with Claude Code as a persistent memory layer. It captures
useful outcomes from one session and retrieves relevant memories when a later
session needs them. The expected benefit is less repeated repository
exploration without reducing answer quality.

This document reports the evidence collected so far and defines the benchmark
required before making general performance claims. The latest result is a
single-run, three-arm comparison of one memory-recall scenario. It demonstrates
that the integration works, but does not establish statistically reliable
improvements across projects or workloads.

## Latest Validated Smoke Result

The current three-arm orchestration was validated on the
`index-store-segmented-routing` scenario with one repetition. Metrics below
cover the continuation session only; setup is the memory-seeding phase.

| Arm | Continuation | Input tokens | Cached input | Output tokens | Tool calls | Recall | Hook time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude native memory | 22.47 s | 123,536 | 123,526 | 1,239 | 4 | 2/3 | 0 ms |
| Lint-AI only | 7.05 s | 15,914 | 15,912 | 376 | 0 | 3/3 | 1.40 s |
| Both layers | 7.17 s | 15,914 | 15,912 | 271 | 0 | 3/3 | 1.36 s |

The native-versus-combined comparison was `15.30 s` lower for the combined
arm in this run. The Lint-AI-only arm disabled Claude auto-memory with
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`; the native arm used an isolated
auto-memory directory; and both Lint-AI arms excluded the MCP code-index
server. These are smoke results, not statistically reliable product claims.

Preserved reports:

- `benchmark/claude_code/results/native-vs-both-fixed/comparison.json`
- `benchmark/claude_code/results/lint-ai-only-fixed/comparison.json`

## MCP-Only Validation Run

The Claude MCP-only path was validated with the same
`index-store-segmented-routing` scenario and one `claude-lint-ai` arm. Lint-AI
lifecycle hooks were disabled; the MCP server remained enabled:

| Arm | MCP status | Continuation | Input tokens | Cached input | Output tokens | All tool calls | Lint-AI MCP calls | Recall | Hook time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude MCP-only | Connected | 47.09 s | 189,022 | 189,006 | 3,678 | 10 | 1 (`info`) | 3/3 | 0 ms |

The MCP server completed initialization and Claude called the Lint-AI
`info` tool. It did not call `search`; Claude selected local repository tools
for the architecture question. The installed `lint-ai-memory` skill guides
Claude to prefer `mcp__lint-ai__search`, but skills are model-invoked guidance
and do not force a tool call. This validates MCP availability with hooks
disabled, but is not a measurement of MCP search/retrieval performance.
The run also reported the `bare-memory-index` forbidden-fact match, despite
scoring all three expected facts; this should be treated as a scenario-quality
issue rather than a successful clean recall result.

A direct MCP-client check against the same server separately confirmed that
the `search` implementation is functional: initialization and `tools/list`
completed, `search` returned three results from 29 indexed documents, and the
query completed in approximately 220 ms. Therefore, the missing search call in
the Claude run is a model tool-selection outcome, not an MCP server failure.

Preserved report:

- `benchmark/claude_code/results/mcp-only/claude-lint-ai/report.json`

## Current Findings

The most representative comparison left Claude Code's normal repository tools
and Agent delegation available. Both runs received the same prompt against the
same Lint-AI repository revision. The only intended difference was whether
Lint-AI's hooks, MCP server, and stored memory were enabled.

Both configurations produced a correct answer. Claude alone rediscovered the
architectural decision by inspecting the repository. With Lint-AI enabled,
Claude received three memories from the exact repository revision and required
less exploration.

| Metric | Claude only | Claude + Lint-AI | Difference |
| --- | ---: | ---: | ---: |
| Answer accuracy | 3/3 | 3/3 | Equal |
| End-to-end wall time | 25.570 s | 21.272 s | 16.8% lower |
| Model turns | 7 | 4 | 42.9% fewer |
| Tool calls | 6 | 3 | 50.0% fewer |
| Parent tokens | 164,156 | 108,974 | 33.6% lower |
| All-model tokens | 164,727 | 108,974 | 33.8% lower |
| Output tokens | 1,891 | 1,061 | 43.9% lower |
| Reported cost | USD 0.1262 | USD 0.1218 | 3.4% lower |
| Retrieved memories | 0 | 3 | +3 |
| Exact-revision memories | 0 | 3 | +3 |
| Injected context | 0 | 3,465 bytes | +3,465 bytes |
| Hook latency | 0 | 189 ms | +189 ms |
| Subagents | 0 | 0 | Equal |

In this run, Lint-AI preserved answer accuracy while reducing tool use, model
turns, elapsed time, and total token consumption. Its hooks accounted for less
than 1% of the Lint-AI run's elapsed time.

### What Was Remembered

The scenario asked Claude to recall an earlier architectural decision without
the user repeating it. The expected facts were:

- Claude integration code should use `IndexStore` instead of constructing a
  `MemoryIndex` directly.
- Segmented memory should use `SegmentRoutingStrategy::LocalDistinctiveness`.
- A global index remains available as a fallback when segment routing does not
  select enough relevant content.

Lint-AI retrieved all three facts from memories associated with the exact Git
revision under test. Claude still verified the retrieved information against
the repository before answering.

### Test Conditions

- Date: 2026-07-14
- Claude Code: `2.1.209`
- Model: `sonnet`, medium effort
- Repository revision: `9d4b751`
- Permission mode: `dontAsk`
- Claude tools and Agent delegation: available in both configurations
- Continuation sessions: fresh and isolated
- Maximum budget: USD 1 per invocation

## Controlled Smoke Test

An earlier diagnostic run denied repository file and shell tools in both
configurations. This tested whether Lint-AI could recover the decision when
Claude could not rediscover it directly. Claude Code was version `2.1.143`; all
other major model and repository settings matched the realistic comparison.

| Metric | Claude only | Claude + Lint-AI | Difference |
| --- | ---: | ---: | ---: |
| Expected facts answered correctly | 0/3 | 3/3 | +3 facts |
| End-to-end time | 38.550 s | 37.811 s | 1.9% lower |
| Model turns | 3 | 2 | 33.3% fewer |
| Tool calls | 3 | 5 | 66.7% more |
| Parent-session tokens | 81,504 | 47,651 | 41.5% lower |
| Total model tokens | 184,353 | 211,982 | 15.0% higher |
| Reported cost | USD 0.2868 | USD 0.2545 | 11.3% lower |
| Memories supplied by Lint-AI | 0 | 7 | +7 |
| Lint-AI context added | 0 bytes | 5,568 bytes | +5,568 bytes |
| Lint-AI hook time | 0 ms | 207 ms | +207 ms |
| Subagents launched | 1 | 1 | Equal |

Claude alone declined to state the decision because it had neither memory nor
repository access. Lint-AI recovered the facts, but its Explore agent consumed
enough additional tokens that total model usage increased. This controlled run
shows the value of memory for otherwise unavailable facts; it does not show a
token-efficiency improvement.

Revision-provenance labels had not yet been implemented, so exact-revision
retrieval cannot be reconstructed for this historical run.

## Reading the Results

**Total model tokens** include the parent Claude session and any delegated
agent work. This is the primary token measure because delegated work still
contributes to usage and cost. Parent-session tokens are reported separately
when they help explain delegation behavior.

**End-to-end time** covers the complete continuation transcript, including
tool calls, hooks, and asynchronous agents. Claude's top-level `duration_ms`
can cover only the final resumed portion of an asynchronous session, so it is
not used as the benchmark wall-clock value.

**Exact-revision memory** means the memory was captured from the same Git
revision being queried. This gives Claude evidence that the retrieved context
matches the checked-out code, although Claude may still choose to verify it.

## Limitations

- Each table represents one run of one scenario.
- Claude output and tool-use choices are nondeterministic.
- The two tables used different Claude Code versions and different tool
  policies, so they are separate experiments and must not be combined.
- The fixture contained the committed `IndexStore` change while the Claude
  adapter and benchmark code came from the working integration build.
- No confidence interval or variance estimate is available yet.

These results are directional engineering evidence, not a general product
benchmark. Release or README claims require the repeated randomized suite and
release gates defined below.

## Benchmark Methodology

## MCP Performance Track

The current results intentionally exclude the Lint-AI MCP server so they
measure lifecycle memory-hook behavior only. MCP should be measured separately
because it adds repository code-search functionality and may build or load a
repository index at startup.

Use a separate full-integration matrix:

| Arm | Claude memory | Lint-AI hooks | Lint-AI MCP |
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

1. Does retrieved memory improve task accuracy across Claude sessions?
2. Does it reduce repeated repository exploration and user restatement?
3. Does it reduce total model tokens after accounting for injected context?
4. What latency does capture, retrieval, indexing, and MCP synchronization add?

The primary comparison is Claude Code with Lint-AI disabled versus the same
Claude Code version, model, repository state, prompts, and permissions with
Lint-AI enabled.

### Non-Goals

- Comparing Claude models or Claude Code releases.
- Comparing Lint-AI with unrelated coding agents.
- Using retrieval benchmark scores alone as evidence of agent improvement.
- Optimizing Rust microbenchmarks without measuring end-to-end Claude tasks.
- Persisting private transcripts as benchmark artifacts.

### Experiment Arms

Each scenario runs in two required arms and one optional diagnostic arm.

| Arm | Hooks | MCP | Persisted memory |
| --- | --- | --- | --- |
| Baseline | Disabled | Disabled | None |
| Lint-AI | Enabled | Enabled | Seeded by the setup session |
| Oracle | Disabled | Disabled | Relevant facts inserted into the prompt |

The oracle arm estimates the best result available when the correct memory is
known. It distinguishes retrieval failures from Claude reasoning failures.

Run order must be randomized. Each arm uses a fresh Claude session, fresh
working copy, and isolated `CLAUDE_CONFIG_DIR` and Lint-AI store. Do not reuse
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

Include negative queries where no memory is relevant. Lint-AI should inject no
context or only context below a defined relevance threshold.

## Controlled Environment

Pin and record:

- Claude Code version
- Claude model identifier
- Lint-AI revision and feature set
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
session. Report each component separately and report injected Lint-AI context
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
2. Create isolated Claude configuration and Lint-AI memory directories.
3. Run the setup session and validate that required facts were established.
4. For the Lint-AI arm, verify expected capture documents and segments.
5. Reset generated repository changes required by the scenario definition.
6. Start a fresh continuation session.
7. Run the continuation prompt with a fixed timeout.
8. Execute the deterministic task validator.
9. Parse Claude and Lint-AI event data.
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

- `.lint-ai/claude-memory`
- Lint-AI measurement JSONL
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
must not claim that Lint-AI reduces tokens or latency when the corresponding
telemetry was missing.

## Implementation Phases

1. Implement schemas, transcript fixtures, and the versioned parser.
2. Add hook/MCP measurement output behind a test-only environment variable.
3. Implement three deterministic smoke scenarios and local reporting.
4. Add randomized repeated A/B execution.
5. Expand to the full category matrix and establish baseline distributions.
6. Run release candidates and publish only reproducible aggregate reports.
