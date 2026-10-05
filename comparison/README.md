# Comparison: Lint-AI and AgentMemory

This directory contains reproducible retrieval and HTTP latency comparisons.
The measurements were taken on 2026-09-01 on the same local machine.

## Latency comparison

The normalized run used 23,366 records, 100 requests per concurrency level,
`top_k`/`limit` 20, and a keyword query. Lint-AI used `POST /search` and
AgentMemory used `POST /agentmemory/smart-search`. AgentMemory was running in
keyless BM25 mode (no embeddings provider).

Start each server, then run the client:

```bash
python3 comparison/http_latency.py \
  --url http://127.0.0.1:8080/search \
  --payload '{"query":"deployment configuration system decision","user_id":"bench-user","top_k":20}'

python3 comparison/http_latency.py \
  --url http://127.0.0.1:3111/agentmemory/smart-search \
  --payload '{"query":"deployment configuration system decision","limit":20}'
```

For a fresh Lint-AI corpus, use the bounded batch seeder (it refreshes once per
batch; the API caps each request at 1,024 messages):

```bash
python3 comparison/seed_lint_ai.py --count 23366 --batch-size 1024 --bulk
```

### Layout throughput comparison

The reusable `throughput_harness.py` builds the release server, runs all three
layouts five times, and reports median results. It uses a cold-start C=1/C=10
HTTP workload of 100 requests per cell by default. Warm-up is opt-in with
`--warmup-requests`; increase the measured workload with `--requests` when
measuring steady-state behavior.

Run the full cold-start harness and optionally save its JSON report:

```bash
mkdir -p .benchmark-tmp
TMPDIR="$PWD/.benchmark-tmp" uv run --no-project python comparison/throughput_harness.py \
  --output comparison/results/throughput-layout-latest.json
```

Use `--skip-build` when the release binary is already built. Use
`--repetitions`, `--requests`, and `--warmup-requests` to control the workload.
The lower-level `throughput.py` remains available for testing one layout.

On macOS, run the Python benchmark through `uv` and provide a real temporary
directory. Some macOS environments expose `/tmp` through a symlink, while the
server deliberately rejects symlinked index paths.

```bash
mkdir -p .benchmark-tmp
TMPDIR="$PWD/.benchmark-tmp" uv run python comparison/throughput.py \
  --mode single --no-cache
```

Repeat the command with `--mode global` and `--mode segment`. The benchmark
does not require third-party Python packages, but `uv run` keeps the execution
environment reproducible.

```bash
python3 comparison/throughput.py --mode single --no-cache
python3 comparison/throughput.py --mode global --no-cache
python3 comparison/throughput.py --mode segment --no-cache
```

`single` uses one global Tantivy index, `global` stores segmented data but
queries every segment, and `segment` uses routed segmented execution. The
three modes use separate temporary indexes and never share persisted state.

### Mixed HTTP read/write load

`mixed_load.py` seeds a temporary 23,366-record corpus, measures a read-only
control, then runs `/search` workers alongside `/add/batch` writers. It reports
read and write throughput and latency percentiles, counts rejected writer-gate
responses, and checks whether a successfully written record is searchable once
the mixed phase ends. Search queries vary by nonce to avoid repeated-query
cache reuse in either phase. Each repetition starts with a fresh server and
index.

```bash
python3 comparison/mixed_load.py \
  --seconds 15 --readers 10 --writers 1 --write-batch 8 \
  --repetitions 3 \
  --output comparison/results/throughput-mixed-read-write-latest.json
```

The temporary index is created under `.benchmark-tmp` because macOS commonly
resolves `/tmp` through a symlink, which the server rejects. This workload is
distinct from the read-only layout benchmark and should be compared against
its paired control from the same run.

### Write-only HTTP throughput

`write_only.py` measures one sequential writer with no concurrent searches.
Every cell starts with a fresh index seeded with 23,366 records across 23
sessions. Batch size is the number of add requests sent in each `/add/batch`
HTTP call, up to the server limit of 128. The run uses a release server, five
warm-up calls, an eight-second timed phase, and three repetitions per batch
size. A final search checks that the last added record is visible.

```bash
mkdir -p .benchmark-tmp
TMPDIR="$PWD/.benchmark-tmp" uv run python comparison/write_only.py \
  --batch-sizes 1 8 32 128 --seconds 8 --repetitions 3 \
  --output comparison/results/throughput-write-only-latest.json
```

The 2026-10-05 Apple M5 Pro diagnostic measured medians of 3.42, 13.76, 26.58,
and 94.20 records/s at batch sizes 1, 8, 32, and 128, respectively. All writes
succeeded and final-record visibility checks passed. See the
[`throughput-write-only-2026-10-05.json`](results/throughput-write-only-2026-10-05.json)
artifact and [server performance notes](../docs/server.md#write-only-throughput).
This is a working-tree diagnostic, not a release capacity claim.

The [follow-up profile](../docs/comparison.md#write-path-optimization-profile-2026-10-05)
records the compact-persistence and receipt-journal changes. It preserves the
documented lifecycle sidecar and includes a full three-repetition rerun.

#### 2026-10-04 mixed-load diagnostic

Three paired 10-second runs used 23,366 seeded records, 10 readers, one writer,
and eight `/add/batch` requests per write. Median read-only control throughput
was 6,053 req/s (p50 1.563 ms). During writes, search throughput fell to 23.94
req/s (p50 425.041 ms, p95 450.154 ms). The writer completed 2.35 batches/s,
or about 18.8 records/s, with a 424.568 ms median batch latency. No writes were
rejected, and the record visibility check passed in all three repetitions.

This diagnostic was run against the modified working tree on the Apple M5 Pro.
It is not a release benchmark. Searches and writes had nearly identical
latencies, consistent with requests contending on the server's service lock:
`write_mutation` holds its write lock while `add_batch` refreshes the snapshot,
while `/search` takes the corresponding read lock. The measurement captures
the user-visible effect; it does not split time spent waiting for the lock from
time spent building and publishing the new snapshot. See
[`throughput-mixed-read-write-2026-10-04.json`](results/throughput-mixed-read-write-2026-10-04.json)
and [server performance notes](../docs/server.md#mixed-read-write-load).

The original Linux layout run is recorded in
[`results/throughput-layout-v0.2.0.json`](results/throughput-layout-v0.2.0.json).
The latest macOS rerun is recorded in
[`results/throughput-layout-watcher-rerun-2026-09-17.json`](results/throughput-layout-watcher-rerun-2026-09-17.json).
It used release mode, 23,366 records, 100 requests per cell, `top_k: 20`, the
query above, and cache disabled on a MacBook Pro with an Apple M5 Pro chip and
24 GB RAM.

The latest published five-run cold-start layout comparison is recorded in
[`results/throughput-layout-latest.json`](results/throughput-layout-latest.json).
It used no warm-up requests. The table below reports the median of those five
runs:

| Mode | C=1 req/s | C=10 req/s | C=10 p50 |
|---|---:|---:|---:|
| Single index | 148.42 | 997.62 | 8.87 ms |
| Global segmented | 237.82 | 1,306.82 | 6.30 ms |
| Routed segment | 237.84 | 1,512.31 | 5.91 ms |

The earlier five-run cold-start experiment is preserved in
[`results/throughput-layout-cold-start-2026-09-17.json`](results/throughput-layout-cold-start-2026-09-17.json).
Its different medians describe a separate run; do not combine its rows with
the latest artifact. Both artifacts are uncached local measurements and are
not a measurement of the exact v0.2.1 release binary.

### Current single-index optimization diagnostic

The one-run 2026-10-03 optimized single-index diagnostic is recorded in
[`results/throughput-single-index-2026-10-03.json`](results/throughput-single-index-2026-10-03.json).
It used 23,366 records, 100 requests per concurrency cell, 10 warm-up requests
per cell, cache disabled, and the standard `deployment configuration system
decision` query. It measured 322.59 req/s at C=1 and 1,723.67 req/s at C=10.
This was a dirty working-tree build on a Darwin arm64 host whose exact hardware
model was not captured. Treat it as a diagnostic for that source state, not a
release result or a direct apples-to-apples comparison with the five-run,
cold-start v0.2.0 artifact.

Before the one-segment filter fix, global-index mode measured 115.61 req/s at
C=1 and 786.29 req/s at C=10. The seeded snapshot had exactly one segment.
Profiling found unnecessary document-ID allow-list construction in that path.
After switching it to the single-index bitmap filter, a rerun measured 321.60
req/s at C=1 and 1,775.21 req/s at C=10. Each run used 23,366 records, 100
requests and 10 warm-ups per concurrency cell, disabled query caching, and ran
once on a dirty working tree. These are diagnostic results, not release
claims. The pre-fix server feature set was not recorded. See the
[server performance notes](../docs/server.md#why-the-one-segment-path-is-slower)
and the raw artifacts:
[`pre-fix`](results/throughput-global-segmented-2026-10-03.json),
[`after-fix`](results/throughput-global-one-segment-2026-10-03.json).

A follow-up routed profile compared one segment with 23 session segments. It
found the same filter allow-list cost in the multi-segment route: filter setup
was 0.034 ms versus 2.959 ms median, with throughput moving from 304 to 124
req/s at C=1 and 1,310 to 453 req/s at C=10. This was a single-run diagnostic
with 25 requests and 3 warm-ups per cell. See
[`throughput-routed-multisegment-profile-2026-10-04.json`](results/throughput-routed-multisegment-profile-2026-10-04.json)
and the [server profile notes](../docs/server.md#multi-segment-routed-filter-cost).

The multi-segment filter fix uses per-segment bitmaps for route eligibility
when semantic supersession does not need an ID allow-list. In a follow-up
three-run diagnostic with 100 requests and 10 warm-ups per cell, filter setup
fell to 0.046 ms median across 23 segments. Throughput was 410.72 req/s at C=1
and 1,041.78 req/s at C=10. See
[`throughput-routed-multisegment-filter-fix-2026-10-04.json`](results/throughput-routed-multisegment-filter-fix-2026-10-04.json).

### Routed query profile and concurrency experiments (2026-10-04)

The matched one-segment versus routed run used the same 23,366-record corpus,
query (`deployment configuration system decision`), `top_k: 20`, cache
behavior, 1,000 measured requests and 250 warm-ups per concurrency cell, and
five repetitions. The segmented corpus had 23 session segments; routing
selected five. Results are medians across repetitions on a MacBook Pro with an
Apple M5 Pro, 24 GB RAM, macOS 26.6.2, and arm64. This was a modified working
tree diagnostic, not a release build claim.

| Layout | C=1 req/s | C=1 p50 | C=10 req/s | C=10 p50 |
|---|---:|---:|---:|---:|
| One segment | 392.81 | 2.481 ms | 2,477.54 | 3.726 ms |
| 23 segments, route 5 | 484.43 | 2.035 ms | 1,274.95 | 7.686 ms |

At C=10, five routed searches are slower than a single search even though the
five segment tasks run in parallel. Route scoring itself is small. With
`LINT_AI_QUERY_TIMINGS=1`, route scoring had a 0.066 ms median (0.088 ms p95)
at C=10, while selected-segment execution had a 6.003 ms median (7.832 ms
p95). The query-term preparation median was 0.012 ms. These stage timings are
instrumented diagnostics and are not throughput results; their instrumentation
adds overhead. This also corrects two earlier labels: the old `filter_ms`
timer measures document filtering, not route scoring, and the old
`default_segment_execution_ms` timer included coordinator setup.

We swept the per-request Rayon fan-out cap for five routed segments. The shared
Rayon pool remains bounded by its worker count; concurrent HTTP requests submit
work to that pool rather than creating a thread per segment task.

| Per-request cap | C=1 req/s | C=10 req/s |
|---|---:|---:|
| 1 | 205.43 | 1,011.25 |
| 2 | 297.45 | 1,060.68 |
| Current cap 8 | 484.43 | 1,274.95 |

Lowering the cap reduced throughput in this workload, so these measurements do
not support tying Rayon worker count to HTTP concurrency or reducing each
request's fan-out. The pool should remain sized for CPU capacity; requests can
queue tasks in the shared pool.

We also swept route width. Narrower routing improved C=10 throughput but reduced
retrieval quality on a diagnostic 100-question LongMemEval-S subset:

| Segments searched | C=10 req/s | Recall@5 | Any-hit Recall@5 |
|---:|---:|---:|---:|
| 1 | 2,336.67 | 0.534 | 0.68 |
| 2 | 1,565.83 | 0.723 | 0.85 |
| 3 | 1,424.67 | 0.811 | 0.89 |
| 5 (current default) | 1,274.95 | 0.886 | 0.94 |

The recall sample is small and does not justify changing the default by itself.
The next design to evaluate is starting with two segments and expanding toward
five when the initial evidence is weak, then validating it on the full
500-question set. The current default route width (5) and fan-out cap (8) were
restored after the compile-time sweeps.

Earlier investigation fixed a corpus-wide document-ID allow-list that was
duplicating per-segment bitmap filtering when semantic visibility did not
require that list. Filter setup fell from 2.959 ms to 0.046 ms in the routed
profile. An Instruments capture then pointed toward candidate scoring and
allocation work. Partial ranking reduced measured per-shard rank time from
0.075 ms to 0.017 ms, but improved C=10 throughput only modestly (1,024 to
1,054 req/s in that paired diagnostic). Removing the candidate-ID vector
lowered one internal timer but gave mixed end-to-end throughput, so it was not
kept. The [server performance notes](../docs/server.md#multi-segment-routed-filter-cost)
record these iterations, including their workload differences and artifacts.

Detailed reports:

- [one segment vs five routed segments](results/throughput-routed-one-vs-five-2026-10-04.json)
- [fan-out and route-width sweep](results/throughput-routed-concurrency-breadth-sweep-2026-10-04.json)
- [separate query-stage profile](results/throughput-routed-query-stage-profile-2026-10-04.json)
- LongMemEval-S route-width reports: [1](results/retrieval-longmemeval-route-width-1-100-2026-10-04.json.gz), [2](results/retrieval-longmemeval-route-width-2-100-2026-10-04.json.gz), [3](results/retrieval-longmemeval-route-width-3-100-2026-10-04.json.gz), [5](results/retrieval-longmemeval-route-width-5-100-2026-10-04.json.gz)

The watcher dependency rerun is recorded in
[`results/throughput-layout-watcher-rerun-2026-09-17.json`](results/throughput-layout-watcher-rerun-2026-09-17.json).
It used the same 23,366-record workload and `uv` runner. Because this is the
standalone HTTP server benchmark, provider MCP watchers are not initialized;
the artifact validates the current server build but does not measure watcher
overhead in an MCP process.

AgentMemory's official load harness seeds one record per request and can be
run with `BENCH_N=23366 BENCH_C=1,10 BENCH_OPS=100 npx tsx
benchmark/load-100k.ts` from that repository. It is intentionally not hidden
behind this repository's scripts.

## Recorded results

The current v0.2.0 post-refactor Lint-AI-only rerun is in
[`results/latency-23366-v0.2.0.json`](results/latency-23366-v0.2.0.json):

| System | C | p50 (ms) | p90 (ms) | p99 (ms) | req/s |
|---|---:|---:|---:|---:|---:|
| Lint-AI v0.2.0 | 1 | 10.39 | 10.90 | 12.21 | 95.00 |
| Lint-AI v0.2.0 | 10 | 23.71 | 38.02 | 45.89 | 386.55 |

The older artifact below is the 0.1.9 single-index baseline used for the
cross-system comparison:

[`results/latency-23366.json`](results/latency-23366.json)

| System | C | p50 (ms) | p90 (ms) | p99 (ms) | req/s |
|---|---:|---:|---:|---:|---:|
| Lint-AI | 1 | 6.96 | 7.65 | 28.17 | 139.52 |
| AgentMemory | 1 | 7.04 | 10.53 | 16.94 | 130.34 |
| Lint-AI | 10 | 10.23 | 11.87 | 12.52 | 952.07 |
| AgentMemory | 10 | 58.81 | 60.35 | 61.33 | 171.26 |

The earlier 5,000-record Lint-AI run is preserved in
[`results/latency-5000.json`](results/latency-5000.json).
The pre-optimization baseline is preserved in
[`results/latency-5000-before-fix.json`](results/latency-5000-before-fix.json),
and the larger in-process corpus benchmark is in
[`results/corpus-scale-19829.json`](results/corpus-scale-19829.json).

## Retrieval quality

The shared scorer is available as [`score_retrieval.py`](score_retrieval.py)
(a wrapper around the canonical [`../benchmark/score_retrieval.py`](../benchmark/score_retrieval.py)).
It uses the same any-hit Recall@K, MRR, and NDCG@10 formulas for both result
formats. The recorded 500-question LongMemEval comparison is in
[`results/retrieval-longmemeval-500.json`](results/retrieval-longmemeval-500.json).

The canonical retrieval headline is:

| System | Any-hit Recall@5 | Any-hit Recall@10 | Any-hit Recall@20 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Lint-AI | 92.4% | 95.6% | 97.0% | 84.0% | 81.8% |
| AgentMemory | 87.0% | 94.8% | 98.4% | 71.6% | 73.0% |

These comparison values are the numbers used on the Home page and in the
benchmark overview. Standalone benchmark runs may also report fractional
recall; those are diagnostic results and should not be mixed with this table.

Latency results measure service behavior only; they are not a quality
comparison. Corpus contents, endpoint implementations, and response formats
remain different, so reproduce and report them with those caveats.

### Staged write measurements

The primary HTTP server stages durable adds by default. Both write-only and
mixed runners flush seeded and warm-up writes before measurement, then flush
again after measurement. Reports distinguish acknowledgement throughput from
published throughput, which includes the final checkpoint duration. Write-only
supports `--wait-for-visibility` for the synchronous response contract.

The mixed runner searches the seeded `bench-user` and writes as
`mixed-bench-user`; `empty_search_responses` detects an accidentally empty read
workload. The October 4 mixed artifact used the wrong reader user and must not
be used as a populated-corpus read baseline.
