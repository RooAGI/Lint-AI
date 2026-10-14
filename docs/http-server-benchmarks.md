# HTTP server benchmarks

These measurements exercise Membrick through its HTTP API. They describe
server throughput for specific workloads, not retrieval quality or a universal
capacity guarantee. The runs used a local release build, a seeded corpus of
23,366 records, and Tantivy 0.25.0 unless noted otherwise.

## Latest diagnostic runs (2026-10-12)

These working-tree diagnostics use the release server built from revision
`95a1d8e` on macOS 26.6.2 arm64, with 23,366 seeded records and Tantivy
0.25.0. They were run before the workspace root allowlist and resource caps
were added, so they do not measure those security changes. The longer 2026-10-05
reference runs remain below for comparison.

### Write-only batch sweep

The server received writes for eight seconds per cell, with one writer and
three repetitions. Publication used the default 250 ms refresh interval and
512-document refresh batch. Every final flush succeeded, and the last written
record was searchable in each repetition.

| Batch records per request | Accepted records/s | Published records/s after flush | Ack latency p50 / p95 / p99 |
|---:|---:|---:|---:|
| 1 | 190.85 | 129.86 | 3.95 / 4.24 / 6.92 ms |
| 8 | 1,481.81 | 1,030.43 | 3.94 / 4.12 / 5.46 ms |
| 32 | 4,279.65 | 2,188.10 | 3.99 / 4.59 / 6.93 ms |
| 128 | 4,494.46 | 1,902.33 | 0.52 / 4.95 / 6.11 ms |

The 128-record cell returned many HTTP 429 responses (2,609–4,115 per
repetition). Its accepted rate is burst admission throughput, not sustainable
write capacity. The [write-only artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/server-throughput-write-only-2026-10-12.json)
contains per-repetition details.

### Direct single-record writes

With the default publication schedule, `/add` accepted 196.27 records/s and
published 186.97 records/s after flush across three 65-second repetitions.
Acknowledgement p50/p95/p99 was 3.88/4.11/6.02 ms, there were no rejections,
and the last record was searchable in every run. The
[single-add artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/server-single-add-investigation-2026-10-12.json)
has the full results.

A separate one-repetition configuration experiment used a 10-second refresh
interval and a 32,768-document batch. It reached 259.46 accepted and 249.34
published records/s, with 3.60/4.08/5.00 ms acknowledgement p50/p95/p99.
Treat this as a schedule experiment, not the default result. See the
[low-publication-frequency artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/server-single-add-low-publication-frequency-2026-10-12.json).

### Mixed search and write load

This three-repetition run used ten search clients and one writer sending
eight-record batches as quickly as possible. The reported 2,568.63–2,961.99
search requests/s is not a useful retrieval-throughput result: 36,694–43,510
responses per repetition were empty because queries did not match the
workload's changing write nonce. In addition, all three final flushes timed
out with HTTP 408, and the final written records were not searchable. Treat
this as a saturated queue/backlog diagnostic only. The
[mixed-load artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/server-throughput-mixed-2026-10-12.json)
preserves the raw results.

## Repeated matched HTTP and IPC benchmark (2026-10-14)

This benchmark sent the same `/provider-memory/mcp/tools/call` requests through
loopback HTTP and the resident Unix socket. It used three 10-second runs per
transport, four search callers, and one writer adding one memory every 500 ms.
Each isolated workspace started with 100 matching Markdown documents. Every
request opened a fresh connection. All six runs had zero errors, and every
acknowledged write was searchable immediately afterward.

The table reports the median of the three per-run measurements. Latency values
are the median of the corresponding per-run percentiles.

| Transport | Search requests/s | Search p50 / p95 / p99 | Writes/s | Write p50 / p95 / p99 | Startup to ready |
|---|---:|---:|---:|---:|---:|
| HTTP over loopback TCP | 2,008.6 | 1.660 / 3.520 / 3.998 ms | 1.8 | 59.59 / 62.18 / 63.40 ms | 650.5 ms |
| Local IPC over Unix socket | 1,808.5 | 1.596 / 3.831 / 4.183 ms | 1.8 | 59.37 / 66.84 / 66.87 ms | 605.2 ms |

Search throughput varied substantially between repetitions: HTTP ranged from
1,981.6 to 2,142.4 requests/s, while IPC ranged from 1,333.1 to 2,557.4.
These results do not establish a throughput advantage for either transport.
They cover a local MCP workload, not Windows named pipes or large production
corpora. The candidate was commit `18339c0a675cb57a196a5677cd0b7a681ec8aadb`
on macOS 26.6.2 arm64, using Python 3.9.6 and the optimized
`agent-integrations` server binary. The benchmark script and complete raw
measurements are in [`comparison/ipc_http_mcp_benchmark.py`](https://github.com/RooAGI/Membrick/blob/main/comparison/ipc_http_mcp_benchmark.py)
and [`comparison/results/release-v0.5.0-ipc-http-mcp-18339c0.json`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/release-v0.5.0-ipc-http-mcp-18339c0.json).
See the [local IPC architecture](local-ipc-architecture.md) for design details.

## Earlier reference results (through 2026-10-05)

| Workload | HTTP request and load | Rate | Result |
|---|---|---:|---|
| Search only | `POST /search`, 10 concurrent clients | **2,246.69 searches/s** | Five-run median with 23,366 records; no writes during measurement. |
| Single add | `POST /add`, one record per request, one writer | **222.52 accepted records/s**; **212.36 published records/s** | Three 65-second runs; no rejected requests. Ack latency p50/p95/p99 was 3.996/4.336/6.964 ms. |
| Batch add | `POST /add/batch`, 128 records per request, one writer | **2,528.46 accepted records/s**; **1,977.12 published records/s** | Three 65-second runs; about 25,500–26,700 requests were rejected per run. This is a saturated burst result, not sustainable capacity. |
| Search while adding | 10 search clients and one `POST /add` per second | **1,248.69 searches/s during writes**; **1,209.01 searches/s in the paired read-only phase** | Three paired 65-second runs; all 65 writes per run were accepted and searchable after flush. |

“Accepted” means the write was acknowledged after durable staging. “Published”
includes the final flush, so the measured records were visible to search before
the run was considered complete. This distinction matters most for batch add:
the high accepted rate came with extensive admission rejections, and the final
published rate accounts for the time needed to flush accepted work.

The search-only, write-only, and mixed-load rows use different workloads. Do
not compare them as if they were measured simultaneously. In particular,
single-add throughput is requests per second because each request contains one
record; batch-add throughput is records per second, not batch requests per
second.

## Per-endpoint details

### `POST /search`

The current read-only reference is the five-repetition Tantivy 0.25.0 single
index run. It used 23,366 records, 1,000 measured requests and 100 warm-up
requests per concurrency cell, `top_k: 20`, and a keyword query. At concurrency
10, median throughput was 2,246.69 requests/s. The same report includes the
routed-segment measurement and Tantivy 0.26.2 comparison.

[Read throughput artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/throughput-tantivy-upgrade-2026-10-04.json)

### `POST /add`

The direct endpoint test sent one memory per request for three 65-second
repetitions. Median durable acknowledgement throughput was 222.52 records/s;
after final flush, median published throughput was 212.36 records/s. There were
no HTTP 429 responses, and the final written record was searchable in every
run.

[Single-add artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/throughput-add-endpoint-2026-10-05.json)

### `POST /add/batch`

Each successful request contained 128 memories. Across three 65-second runs,
the median was 19.75 successful batch requests/s, or 2,528.46 accepted
records/s. Including final flush, published throughput was 1,977.12
records/s. The server returned 25,501–26,673 HTTP 429 responses per run, so
the accepted rate is a saturated burst rate. Use the published rate and the
rejection count when evaluating this workload.

[Batch-add artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/throughput-write-publication-temporal-incremental-2026-10-05.json)

An additional publication-schedule diagnostic reached 2,583.54 accepted and
2,077.75 published records/s with a custom one-second/4,096-document trigger.
It also rejected 26,760–27,128 requests per run. That schedule was a tuning
experiment, not the default configuration, so it is not the headline result.

[Publication-schedule diagnostic](https://github.com/RooAGI/Membrick/blob/main/comparison/results/throughput-write-publication-coalesced-2026-10-05.json)

### Search while adding

This paired test ran ten concurrent search clients. In the write phase, one
writer sent a single `/add` request each second for 65 seconds. Median search
throughput was 1,248.69 searches/s while writing and 1,209.01 searches/s in
the read-only control phase. The three runs had no rejected requests, and the
final write was confirmed searchable after flushing.

The small difference between the control and write phases is an observation
from this workload, not evidence that writes improve search performance.

[Mixed-load artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/throughput-write-publication-mixed-concentrated-2026-10-05.json)

## How to read these numbers

- Throughput depends on the machine, corpus, query, concurrency, and publication
  schedule. These are local measurements on an Apple M5 Pro MacBook Pro running
  macOS 26.6.2.
- The search-only benchmark measures `POST /search`; the LongMemEval retrieval
  benchmark measures answer-session recall and is documented separately in
  [Benchmark overview](benchmark.md).
- For the cross-system comparison and older runs, see
  [Comparison](comparison.md). The benchmark scripts and raw results are in
  the repository's [`comparison/` directory](https://github.com/RooAGI/Membrick/tree/main/comparison).
