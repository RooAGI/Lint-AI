# Prometheus

Lint-AI exposes a Prometheus text endpoint at `GET /metrics`. Prometheus can
scrape it to collect aggregate query and agent activity metrics.

Lint-AI currently supports Prometheus exposition format. It does not currently
export OpenTelemetry OTLP metrics, traces, or logs.

## Start the Lint-AI server

Start the server for the project you want to monitor. For a local Prometheus
instance, bind Lint-AI to localhost:

```bash
cargo run --release --bin server -- \
  --project-root /path/to/project \
  --bind 127.0.0.1:8080 \
  --server-token "$LINT_AI_SERVER_TOKEN"
```

Replace `/path/to/project` with your project directory. Set
`LINT_AI_SERVER_TOKEN` to a secret value before starting the server. For a
single-user local setup, you can instead use `--allow-unauthenticated`; keep
the server bound to localhost.

## Configure Prometheus

Add a scrape job to `prometheus.yml`. When Lint-AI uses a server token, put the
same token in a file readable by Prometheus and set `bearer_token_file`:

```yaml
scrape_configs:
  - job_name: lint-ai
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8080"]
    bearer_token_file: /path/to/lint-ai-server-token
```

For an unauthenticated localhost-only server, omit `bearer_token_file`. Reload
Prometheus after changing its configuration, then check **Status → Targets** in
the Prometheus UI. The `lint-ai` target should show as **UP**.

You can also inspect the endpoint directly:

```bash
curl -H "Authorization: Bearer $LINT_AI_SERVER_TOKEN" \
  http://127.0.0.1:8080/metrics
```

## Metrics

### Search activity

| Metric | Type | Meaning |
|---|---|---|
| `lint_ai_query_requests_window` | Gauge | Search requests in the retained telemetry window |
| `lint_ai_query_errors_window` | Gauge | Search errors in that window |
| `lint_ai_query_empty_results_window` | Gauge | Searches with no results in that window |
| `lint_ai_query_requests_per_second` | Gauge | Recent search request rate |
| `lint_ai_query_error_rate` | Gauge | Fraction of searches that returned errors |
| `lint_ai_query_empty_result_rate` | Gauge | Fraction of searches with no results |
| `lint_ai_query_latency_ms{quantile="0.5"}` | Gauge | Estimated median search latency in milliseconds |
| `lint_ai_query_latency_ms{quantile="0.95"}` | Gauge | Estimated p95 search latency in milliseconds |

These search values describe the rolling telemetry window (up to ten minutes),
so they can go down as older buckets expire. The latency quantiles are estimates
from Lint-AI's existing latency buckets, not Prometheus histograms.

### Agent activity

Provider metrics use the bounded `provider` label values `claude`, `codex`,
`gemini-cli`, `agy`, `openclaw`, and `hermes`.

| Metric | Type | Meaning |
|---|---|---|
| `lint_ai_provider_compiled{provider}` | Gauge | Whether this server build includes the integration (1 or 0) |
| `lint_ai_provider_observed{provider}` | Gauge | Whether any lifecycle event has been received (1 or 0) |
| `lint_ai_provider_events_total{provider}` | Counter | Lifecycle events recorded for the provider |
| `lint_ai_provider_sessions_started_total{provider}` | Counter | Sessions started |
| `lint_ai_provider_sessions_ended_total{provider}` | Counter | Sessions ended |
| `lint_ai_provider_retrieval_events_total{provider}` | Counter | Events classified as retrieval |
| `lint_ai_provider_capture_events_total{provider}` | Counter | Events classified as capture |
| `lint_ai_provider_sessions_active{provider}` | Gauge | Active sessions observed by Lint-AI |
| `lint_ai_provider_last_seen_timestamp_seconds{provider}` | Gauge | Unix time of the last event, or zero if none was received |
| `lint_ai_provider_recent_events{provider,category}` | Gauge | Events by category in the retained provider ledger |
| `lint_ai_provider_token_usage_recent{provider,kind}` | Gauge | Token counts in retained events that reported usage |

`category` is one of `session`, `retrieval`, `compaction`, `capture`, or
`lifecycle`. It groups event kinds for monitoring; it does not expose each
session or tool event separately. Recent event and token gauges summarize the
provider ledger, which retains at most 500 events per provider. Token values
are only available when the agent reports them.

### Memory index

| Metric | Type | Meaning |
|---|---|---|
| `lint_ai_index_source_documents` | Gauge | Source documents in the project index |
| `lint_ai_index_records` | Gauge | Indexed memory records |
| `lint_ai_index_dirty` | Gauge | Whether there are unpublished index changes (1 or 0) |
| `lint_ai_index_store_revision` | Gauge | Current mutable store revision |
| `lint_ai_index_snapshot_revision` | Gauge | Published search snapshot revision |
| `lint_ai_index_revision_lag` | Gauge | Store revision minus published snapshot revision |
| `lint_ai_index_segments` | Gauge | Segments in the published search snapshot |

For example, graph recent p95 latency and active sessions in Prometheus with:

```promql
lint_ai_query_latency_ms{quantile="0.95"}
lint_ai_provider_sessions_active
```

Prometheus receives aggregate metrics only. It does not receive query text,
memory contents, session identifiers, prompts, tool arguments, or the
dashboard's individual event timeline. See [Telemetry details](telemetry.md)
for the retained telemetry data and [Observability](observability.md) for the
dashboard.

## Network access

The metrics endpoint uses the server's configured authentication. Keep Lint-AI
on localhost unless you have secured the network path and token distribution.
See the [HTTP server guide](server.md) for authentication and bind-address
behavior.
