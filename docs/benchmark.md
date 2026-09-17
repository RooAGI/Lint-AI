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

### Reading the two benchmark tracks

The **500-question aggregate** covers the full scoped dataset and is the source
of the website headline. The **133-question segmented comparison** covers only
the multi-session slice and compares fixed segmented, adaptive segmented, and
single-index modes. They must not be merged into one score.

The single-index row in the segmented table is the controlled global baseline
for that same 133-question slice. It is not the 500-question aggregate rerun,
so its recall and latency should not be compared directly with the headline
numbers.

Within either track, the metric name is significant: **Fractional Recall@K**
(regular recall) measures the fraction of all relevant sessions recovered, while
**Any-hit Recall@K** measures whether at least one relevant session was found.
`@5`, `@10`, and `@20` are separate result cutoffs, so each value must retain
both its metric type and cutoff label.

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

### Segmented-index comparison

The latest 133-query multi-session comparison uses the same corpus with the
segmented benchmark's fixed, adaptive, and single-index modes:

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | MRR | Average latency |
|---|---:|---:|---:|---:|
| Segmented (fixed top-5) | **95.49%** | 95.49% | **0.859** | **1.25 ms** |
| Segmented (adaptive 5→12) | **96.24%** | **96.24%** | 0.839 | 4.36 ms |
| Single index | 93.23% | 93.98% | 0.814 | 6.41 ms |

This run explicitly selected fixed top-5; the benchmark CLI and server default
to top-3. Adaptive routing improves any-hit recall at the cost of additional query
latency. These results are scoped to the multi-session slice and should not be
compared directly with the 500-question aggregate above.

Summary artifact:
[`segment-multisession-v0.2.0.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-multisession-v0.2.0.json).

## Corpus-scale and HTTP results

The in-process corpus-scale run used 19,829 sessions and 500 eligible queries:

| Query p50 | Query p95 | Any-hit Recall@10 |
|---:|---:|---:|
| 1.65 ms | 3.42 ms | 85.0% |

The normalized Lint-AI HTTP run used 23,366 records and 100 requests per cell.
It is a 0.1.9 service baseline and predates the 0.2.0 Axum/snapshot refactor:

| Concurrency | p50 | p90 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 6.96 ms | 7.65 ms | 28.17 ms | 139.52 req/s |
| 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952.07 req/s |

Latency is a separate service-load measurement; it is not a retrieval-quality
metric. See the recorded artifacts in [`comparison/results/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison/results).
