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

For a fresh Lint-AI corpus, use the bulk seeder (it refreshes once):

```bash
python3 comparison/seed_lint_ai.py --count 23366
```

AgentMemory's official load harness seeds one record per request and can be
run with `BENCH_N=23366 BENCH_C=1,10 BENCH_OPS=100 npx tsx
benchmark/load-100k.ts` from that repository. It is intentionally not hidden
behind this repository's scripts.

## Recorded results

The normalized service-load results are in
[`results/latency-23366.json`](results/latency-23366.json). In summary:

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
