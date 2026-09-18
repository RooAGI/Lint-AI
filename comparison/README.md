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

The original Linux layout run is recorded in
[`results/throughput-layout-v0.2.0.json`](results/throughput-layout-v0.2.0.json).
The latest macOS rerun is recorded in
[`results/throughput-layout-watcher-rerun-2026-09-17.json`](results/throughput-layout-watcher-rerun-2026-09-17.json).
It used release mode, 23,366 records, 100 requests per cell, `top_k: 20`, the
query above, and cache disabled on a MacBook Pro with an Apple M5 Pro chip and
24 GB RAM.

The five-run cold-start summary is recorded in
[`results/throughput-layout-cold-start-2026-09-17.json`](results/throughput-layout-cold-start-2026-09-17.json).
The table below reports the median of those five runs; no warm-up requests were
used.

| Mode | C=1 req/s | C=10 req/s | C=10 p50 |
|---|---:|---:|---:|
| Single index | 136.47 | 927.96 | 9.47 ms |
| Global segmented | 229.87 | 1,480.30 | 6.16 ms |
| Routed segment | 235.30 | 1,510.15 | 6.16 ms |

These are uncached local measurements; they are not directly comparable with
the historical 0.1.9 result until machine and server-build provenance match.

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
