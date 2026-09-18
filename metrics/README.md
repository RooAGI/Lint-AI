# Session metrics

This directory defines the user-facing metric report for recorded Claude and
Codex sessions.

## Files

- `session-metric-report-design.md` — metric definitions and review UI contract
- `session-metric-report.schema.json` — JSON Schema for generated reports
- `generate_session_metric_report.py` — CLI parser and report generator
- `reports/` — generated report instances; session archives remain the source evidence

Reports support both a single recorded session and a baseline/replay
comparison. Quality and efficiency are intentionally reported separately: a
faster session is not an improvement if it fails the task.

The CLI reads `manifest.json` and `events.jsonl`. It tolerates incomplete or
malformed JSONL lines, preserves unavailable measurements as `null`, and does
not modify the recorded session.

## Generate a report

The report generator should consume immutable session manifests and events,
then write a new report revision. It must not modify the session archive.

```text
metrics/
├── session-metric-report-design.md
├── session-metric-report.schema.json
└── reports/
    └── <report-id>.json
```

Missing provider measurements are represented as `null`, never as zero. This
distinguishes “not observed” from “observed as zero.”
