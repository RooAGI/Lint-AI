# Prometheus

Membrick exposes a Prometheus text endpoint at `GET /metrics`. Prometheus can
scrape it to collect aggregate query and agent activity metrics.

Membrick currently supports Prometheus exposition format. It does not currently
export OpenTelemetry OTLP metrics, traces, or logs.

## Start the Membrick server

Start the server for the project you want to monitor. For a local Prometheus
instance, bind Membrick to localhost:

```bash
cargo run --release --bin server -- \
  --project-root /path/to/project \
  --bind 127.0.0.1:8080 \
  --server-token "$MEMBRICK_SERVER_TOKEN"
```

Replace `/path/to/project` with your project directory. Set
`MEMBRICK_SERVER_TOKEN` to a secret value before starting the server. For a
single-user local setup, you can instead use `--allow-unauthenticated`; keep
the server bound to localhost.

## Configure Prometheus

Add a scrape job to `prometheus.yml`. When Membrick uses a server token, put the
same token in a file readable by Prometheus and set `bearer_token_file`:

```yaml
scrape_configs:
  - job_name: membrick
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8080"]
    bearer_token_file: /path/to/membrick-server-token
```

For an unauthenticated localhost-only server, omit `bearer_token_file`. Reload
Prometheus after changing its configuration, then check **Status → Targets** in
the Prometheus UI. The `membrick` target should show as **UP**.

You can also inspect the endpoint directly:

```bash
curl -H "Authorization: Bearer $MEMBRICK_SERVER_TOKEN" \
  http://127.0.0.1:8080/metrics
```

## Metrics

### Search activity

| Metric | Type | Meaning |
|---|---|---|
| `membrick_query_requests_window` | Gauge | Search requests in the retained telemetry window |
| `membrick_query_errors_window` | Gauge | Search errors in that window |
| `membrick_query_empty_results_window` | Gauge | Searches with no results in that window |
| `membrick_query_requests_per_second` | Gauge | Recent search request rate |
| `membrick_query_error_rate` | Gauge | Fraction of searches that returned errors |
| `membrick_query_empty_result_rate` | Gauge | Fraction of searches with no results |
| `membrick_query_latency_ms{quantile="0.5"}` | Gauge | Estimated median search latency in milliseconds |
| `membrick_query_latency_ms{quantile="0.95"}` | Gauge | Estimated p95 search latency in milliseconds |

These search values describe the rolling telemetry window (up to ten minutes),
so they can go down as older buckets expire. The latency quantiles are estimates
from Membrick's existing latency buckets, not Prometheus histograms.

### Agent activity

Provider metrics use the bounded `provider` label values `claude`, `codex`,
`gemini-cli`, `agy`, `openclaw`, and `hermes`.

| Metric | Type | Meaning |
|---|---|---|
| `membrick_provider_compiled{provider}` | Gauge | Whether this server build includes the integration (1 or 0) |
| `membrick_provider_observed{provider}` | Gauge | Whether any lifecycle event has been received (1 or 0) |
| `membrick_provider_events_total{provider}` | Counter | Lifecycle events recorded for the provider |
| `membrick_provider_sessions_started_total{provider}` | Counter | Sessions started |
| `membrick_provider_sessions_ended_total{provider}` | Counter | Sessions ended |
| `membrick_provider_retrieval_events_total{provider}` | Counter | Events classified as retrieval |
| `membrick_provider_capture_events_total{provider}` | Counter | Events classified as capture |
| `membrick_provider_sessions_active{provider}` | Gauge | Active sessions observed by Membrick |
| `membrick_provider_last_seen_timestamp_seconds{provider}` | Gauge | Unix time of the last event, or zero if none was received |
| `membrick_provider_recent_events{provider,category}` | Gauge | Events by category in the retained provider ledger |
| `membrick_provider_token_usage_recent{provider,kind}` | Gauge | Token counts in retained events that reported usage |

`category` is one of `session`, `retrieval`, `compaction`, `capture`, or
`lifecycle`. It groups event kinds for monitoring; it does not expose each
session or tool event separately. Recent event and token gauges summarize the
provider ledger, which retains at most 500 events per provider. Token values
are only available when the agent reports them.

### Memory index

| Metric | Type | Meaning |
|---|---|---|
| `membrick_index_source_documents` | Gauge | Source documents in the project index |
| `membrick_index_records` | Gauge | Indexed memory records |
| `membrick_index_dirty` | Gauge | Whether there are unpublished index changes (1 or 0) |
| `membrick_index_store_revision` | Gauge | Current mutable store revision |
| `membrick_index_snapshot_revision` | Gauge | Published search snapshot revision |
| `membrick_index_revision_lag` | Gauge | Store revision minus published snapshot revision |
| `membrick_index_segments` | Gauge | Segments in the published search snapshot |

For example, graph recent p95 latency and active sessions in Prometheus with:

```promql
membrick_query_latency_ms{quantile="0.95"}
membrick_provider_sessions_active
```

Prometheus receives aggregate metrics only. It does not receive query text,
memory contents, session identifiers, prompts, tool arguments, or the
dashboard's individual event timeline. See [Telemetry details](telemetry.md)
for the retained telemetry data and [Observability](observability.md) for the
dashboard.

## Network access

The metrics endpoint uses the server's configured authentication. Keep Membrick
on localhost unless you have secured the network path and token distribution.
See the [HTTP server guide](server.md) for authentication and bind-address
behavior.
