# Benchmark overview

This page is the web version of the repository's benchmark README. It covers
the standalone retrieval, corpus-scale, and LongMemEval benchmark commands.

For the full fair AgentMemory comparison, see the [Comparison](comparison.md)
page.

## Reproduce the published retrieval result

From the repository root, run:

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 5 --k 10 --k 20 \
  --out benchmark/data/lintai_longmemeval_scoped_results.json
```

The raw dataset can be refreshed and verified with:

```bash
python3 benchmark/download_longmemeval_raw.py
```

To compare against AgentMemory with the identical evaluator:

```bash
python3 benchmark/score_retrieval.py \
  --dataset benchmark/data/longmemeval_s_raw.json \
  --lintai benchmark/data/lintai_longmemeval_scoped_results.json \
  --agentmemory /path/to/agentmemory/benchmark/data/longmemeval_results_bm25.json
```

The complete source README, scripts, and recorded JSON artifacts remain in the
repository's [`benchmark/`](https://github.com/RooAGI/Lint-AI/tree/main/benchmark)
and [`comparison/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison)
directories.

## Our verified LongMemEval-S results

These are 500 question-scoped queries using the current heuristic release
backend and no embeddings.

**Any-hit Recall@K** is the percentage of questions where at least one
correct answer session appears in the top *K* results. **Fractional Recall@K**
is the average fraction of all correct answer sessions recovered in the top
*K* results. Any-hit measures whether the search found usable evidence;
fractional recall measures how much of the relevant evidence it found.

| Metric | Result |
|---|---:|
| Any-hit Recall@5 | 92.4% |
| Any-hit Recall@10 | 95.6% |
| Any-hit Recall@20 | 97.0% |
| Fractional Recall@5 | 83.5% |
| Fractional Recall@10 | 89.5% |
| Fractional Recall@20 | 91.1% |
| MRR | 84.0% |
| NDCG@10 | 81.8% |

Lint-AI's fractional recall is 83.5% at 5, 89.5% at 10, and 91.1% at 20.

| Question type | n | Any-hit @5 | Any-hit @10 | Any-hit @20 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Single-session assistant | 56 | 100.0% | 100.0% | 100.0% | 98.2% | 98.7% |
| Single-session user | 70 | 94.3% | 98.6% | 98.6% | 77.5% | 82.7% |
| Single-session preference | 30 | 80.0% | 93.3% | 93.3% | 65.8% | 72.2% |
| Knowledge update | 78 | 98.7% | 100.0% | 100.0% | 94.6% | 92.4% |
| Temporal reasoning | 133 | 86.5% | 91.7% | 96.2% | 81.8% | 78.0% |
| Multi-session | 133 | 93.2% | 94.0% | 94.7% | 81.5% | 73.8% |

| Benchmark detail | Value |
|---|---|
| Dataset | LongMemEval-S |
| Questions | 500 question-scoped queries |
| Backend | Heuristic release backend |
| Embeddings | Disabled |
| Cutoffs | 5, 10, and 20 |
| Average query latency | 1.88 ms |
| Reproduction command | `cargo run --release --bin haystack_scoped_benchmark -- --longmemeval benchmark/data/longmemeval_s_raw.json --k 5 --k 10 --k 20` |

## Corpus-scale and HTTP results

The in-process corpus-scale run used 19,829 sessions and 500 eligible queries:

| Query p50 | Query p95 | Any-hit Recall@10 |
|---:|---:|---:|
| 1.65 ms | 3.42 ms | 85.0% |

The normalized Lint-AI HTTP run used 23,366 records and 100 requests per cell:

| Concurrency | p50 | p90 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 6.96 ms | 7.65 ms | 28.17 ms | 139.52 req/s |
| 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952.07 req/s |

Latency is a separate service-load measurement; it is not a retrieval-quality
metric. See the recorded artifacts in [`comparison/results/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison/results).
