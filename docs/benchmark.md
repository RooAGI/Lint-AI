# Benchmark overview

This page is the web version of the repository's benchmark README. It covers
the standalone retrieval, corpus-scale, and LongMemEval benchmark commands.

For the full fair AgentMemory comparison, see the [Comparison](comparison.md)
page.

For explanations of routing, adaptive retrieval, intent retrieval, and fusion,
see [LongMemEval segmented retrieval strategies](longmemeval-strategies.md).

## Reproduce the published retrieval result

From the repository root, run:

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 5 --k 10 --k 20 \
  --ner-provider heuristic \
  --out benchmark/data/lintai_longmemeval_scoped_results.json
```

The `--ner-provider` flag selects the Tier1 NER backend (`heuristic` or `spacy`,
default `spacy`). Pass `--ner-provider heuristic` to reproduce the numbers below,
which were recorded with the heuristic release backend.

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
of the website headline. The segmented benchmark also covers all 500 questions
and includes its own single-index control. Keep its scores and latencies
separate from the standalone headline because it uses a different harness and
timing scope. The earlier 133-question multi-session comparison is retained as
a historical slice.

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
| Any-hit Recall@5 | 94.2% |
| Any-hit Recall@10 | 96.8% |
| Any-hit Recall@20 | 97.6% |
| Fractional Recall@5 | 85.6% |
| Fractional Recall@10 | 92.0% |
| Fractional Recall@20 | 93.1% |
| MRR | 87.0% |
| NDCG@10 | 85.0% |

Lint-AI's fractional recall is 85.6% at 5, 92.0% at 10, and 93.1% at 20.

### Latest standalone rerun (2026-10-04)

A fresh release-mode run evaluated all 500 questions with the heuristic NER
backend, a single index, and no embeddings. It used cutoffs 1, 3, 5, 10, and
20. The detailed per-question report is saved at
`comparison/results/retrieval-longmemeval-current-2026-10-04.json.gz`.

| Metric | Result |
|---|---:|
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| MRR | 87.0% |
| NDCG@10 | 85.11% |
| Search latency, mean / p50 / p95 | 2.68 / 2.15 / 4.47 ms |

Latency here is measured around each `MemoryService::search` call. It excludes
the per-question haystack indexing performed before the query. The previous
13.0 ms figure below comes from the earlier published report and has a
different timing scope, so the two latency values are not directly comparable.
Retrieval quality is close to the published result: any-hit recall and MRR are
unchanged at the displayed precision, while fractional Recall@5 is 0.8
percentage points higher.

| Question type | n | Any-hit @5 | Any-hit @10 | Any-hit @20 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Single-session assistant | 56 | 100.0% | 100.0% | 100.0% | 99.1% | 99.3% |
| Single-session user | 70 | 97.1% | 98.6% | 98.6% | 88.2% | 90.9% |
| Single-session preference | 30 | 80.0% | 90.0% | 93.3% | 66.2% | 71.6% |
| Knowledge update | 78 | 100.0% | 100.0% | 100.0% | 96.2% | 94.3% |
| Temporal reasoning | 133 | 89.5% | 94.0% | 96.2% | 82.5% | 80.0% |
| Multi-session | 133 | 94.7% | 97.0% | 97.0% | 84.9% | 78.4% |

### IndexStore duplicate-index refactor rerun (2026-10-05)

After removing the unused `IndexStore` lexical writer, the same release-mode
500-question scoped benchmark returned identical aggregate retrieval metrics
to the pre-refactor Tantivy 0.26.2 result. Two queries had tie-order changes
within the returned IDs; both retained the same per-query MRR and NDCG. The
measured average search latency was
2.45 ms versus 2.60 ms before the refactor; these are single runs, so treat
that latency difference as directional only. The raw report is
[`retrieval-longmemeval-current-2026-10-05-indexstore-refactor.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current-2026-10-05-indexstore-refactor.json.gz).

| Metric | Before | After |
|---|---:|---:|
| Any-hit Recall@5 / @10 | 92.0% / 96.0% | 92.0% / 96.0% |
| Fractional Recall@5 / @10 | 82.50% / 89.68% | 82.50% / 89.68% |
| MRR | 0.83345 | 0.83345 |
| NDCG@10 | 0.81599 | 0.81599 |
| Average query latency | 2.60 ms | 2.45 ms |

| Benchmark detail | Value |
|---|---|
| Dataset | LongMemEval-S |
| Questions | 500 question-scoped queries |
| Backend | Heuristic release backend |
| Embeddings | Disabled |
| Cutoffs | 5, 10, and 20 |
| Published-run average query latency | 13.0 ms |
| Latest rerun command | `cargo run --release --bin haystack_scoped_benchmark -- --longmemeval benchmark/data/longmemeval_s_raw.json --k 1 --k 3 --k 5 --k 10 --k 20 --ner-provider heuristic --out comparison/results/retrieval-longmemeval-current-2026-10-04.json.gz` |

The published numbers above used heuristic Tier1 NER. Note: this benchmark's
query loop (`analyze_query` → lexical search → aggregation) never invokes
behood, so the numbers are independent of the behood path.

behood (bekind) owns the full tag→chunk→judge pipeline (Luyi 2026-09-30):
the query path sends raw texts to `bekind --serve` and gets per-text
verdicts back. No descriptors cross the process boundary and no Python
process is involved. A 500-question parse→judge A/B (pre-move: spaCy parse
vs heuristic parse, same bekind binary) gave identical scope verdicts on
all 500 questions; entity verdicts matched on 228/500, the gap being
spaCy-NER teacher votes the heuristic backend deliberately does not invent.

### Segmented-index comparison

The latest segmented benchmark covers all 500 eligible LongMemEval-S
questions. The single-index control and segment modes use the same query set,
heuristic backend, and harness:

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | Fractional Recall@5 | MRR | Average latency |
|---|---:|---:|---:|---:|---:|
| Segmented (fixed top-5) | 94.20% | 94.20% | **86.77%** | 0.880 | 4.08 ms |
| Segmented (adaptive 5→12, enriched) | 94.20% | 96.20% | 85.64% | 0.869 | 7.05 ms |
| Intent baseline | 94.20% | 96.20% | 85.54% | 0.863 | 0.77 ms |
| Fused adaptive + global | 95.40% | 97.20% | 87.11% | 0.880 | 29.47 ms |
| Fused temporal + global | **96.00%** | **97.80%** | **88.51%** | **0.885** | 26.98 ms |
| Single-index control | 94.40% | 96.60% | 86.41% | **0.872** | 14.34 ms |

The run used `--segment-top-n 5`, `--adaptive-segment-max-n 12`, and the
`coverage-local` router. Latency is the benchmark's average per-query time for
each variant, excluding haystack indexing. The direct-index control's latency
is not comparable to standalone MemoryService latency above. Fused temporal +
global has the strongest recall and MRR in this table, with higher latency.
The intent baseline is fastest among the listed modes.

Full per-query report:
[`segment-longmemeval-500-2026-10-04.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-longmemeval-500-2026-10-04.json.gz).
The historical 133-query multi-session result remains at
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
