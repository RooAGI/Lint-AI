# HTTP server benchmarks

These measurements exercise Lint-AI through its HTTP API. They describe
server throughput for specific workloads, not retrieval quality or a universal
capacity guarantee. The runs used a local release build, a seeded corpus of
23,366 records, and Tantivy 0.25.0 unless noted otherwise.

## Current results

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

[Read throughput artifact](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-tantivy-upgrade-2026-10-04.json)

### `POST /add`

The direct endpoint test sent one memory per request for three 65-second
repetitions. Median durable acknowledgement throughput was 222.52 records/s;
after final flush, median published throughput was 212.36 records/s. There were
no HTTP 429 responses, and the final written record was searchable in every
run.

[Single-add artifact](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-add-endpoint-2026-10-05.json)

### `POST /add/batch`

Each successful request contained 128 memories. Across three 65-second runs,
the median was 19.75 successful batch requests/s, or 2,528.46 accepted
records/s. Including final flush, published throughput was 1,977.12
records/s. The server returned 25,501–26,673 HTTP 429 responses per run, so
the accepted rate is a saturated burst rate. Use the published rate and the
rejection count when evaluating this workload.

[Batch-add artifact](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-temporal-incremental-2026-10-05.json)

An additional publication-schedule diagnostic reached 2,583.54 accepted and
2,077.75 published records/s with a custom one-second/4,096-document trigger.
It also rejected 26,760–27,128 requests per run. That schedule was a tuning
experiment, not the default configuration, so it is not the headline result.

[Publication-schedule diagnostic](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-coalesced-2026-10-05.json)

### Search while adding

This paired test ran ten concurrent search clients. In the write phase, one
writer sent a single `/add` request each second for 65 seconds. Median search
throughput was 1,248.69 searches/s while writing and 1,209.01 searches/s in
the read-only control phase. The three runs had no rejected requests, and the
final write was confirmed searchable after flushing.

The small difference between the control and write phases is an observation
from this workload, not evidence that writes improve search performance.

[Mixed-load artifact](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-mixed-concentrated-2026-10-05.json)

## How to read these numbers

- Throughput depends on the machine, corpus, query, concurrency, and publication
  schedule. These are local measurements on an Apple M5 Pro MacBook Pro running
  macOS 26.6.2.
- The search-only benchmark measures `POST /search`; the LongMemEval retrieval
  benchmark measures answer-session recall and is documented separately in
  [Benchmark overview](benchmark.md).
- For the cross-system comparison and older runs, see
  [Comparison](comparison.md). The benchmark scripts and raw results are in
  the repository's [`comparison/` directory](https://github.com/RooAGI/Lint-AI/tree/main/comparison).
