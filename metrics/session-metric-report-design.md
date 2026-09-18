# Session metric report design

## Purpose

The report answers four questions:

1. Did the agent complete the task?
2. How long and how many resources did it use?
3. Did Lint-AI memory change the session?
4. Was the recording complete and trustworthy?

The report is derived data. The recorded session (`manifest.json` and
`events.jsonl`) remains the immutable evidence source.

## Report shape

```text
report
scenario
sessions.baseline
sessions.replay?        # present for A/B or replay reports
comparison?             # derived baseline-to-replay deltas
provenance
```

## Metrics

### Quality — primary user outcome

- `task_success`: pass, fail, or unavailable
- `expected_facts_passed` / `expected_facts_total`
- `validators_passed` / `validators_total`
- `forbidden_facts_found`
- `answer_quality_notes`

Quality must not be reduced to a composite score. A report can be faster while
still being worse if task success or validator results decline.

### Performance — speed and resource use

- `duration_ms`
- `time_to_first_response_ms`
- `input_tokens`, `output_tokens`, `cached_input_tokens`, `total_tokens`
- `estimated_cost_usd`
- `tool_calls`, `tool_errors`, `tool_retry_count`

### Memory impact

- `memory_enabled`
- `retrieval_calls`
- `retrieved_documents`
- `injected_context_bytes` and `injected_context_tokens`

### Reliability and recording health

- `session_status`: complete, interrupted, failed, or active
- `recorded_events` / `expected_events`
- `recording_completeness`
- `provider_errors` and `hook_errors`
- `redaction_count`

## User-facing report

The first view should show decision-relevant metrics only:

```text
Scenario: routing-memory                         PASS

                         Baseline       Replay       Change
Task success            PASS           PASS         —
Expected facts          2 / 3          3 / 3         +1
Validators              1 / 2          2 / 2         +1
Total tokens            18,420         15,870       -13.8%
Duration                4m 12s         3m 48s         -9.5%
Tool calls              14             10            -4
Injected context        0 B            6,240 B       +6,240 B
Redactions              2              3             +1
```

Detailed event timing, usage sources, and validator logs belong behind an
expandable detail view. Every displayed summary should link to the supporting
session event when one exists.

## Comparison rules

For numeric metrics:

```text
delta          = replay - baseline
relative_delta = (replay - baseline) / baseline
```

Use relative change only when the baseline is non-zero. For quality metrics,
show outcome changes (`PASS → FAIL`, `2/3 → 3/3`) rather than pretending they
are ordinary numeric deltas. Label each value as observed, estimated, or
user-rated where applicable.

## Immutability and privacy

- Generate a new report revision for every evaluation.
- Never rewrite the session archive or scenario definition.
- Preserve provider, model, repository revision, settings root, and scenario ID.
- Keep redaction counts visible while never exposing the redacted value.
- Treat cost as estimated unless the provider supplies authoritative usage data.
