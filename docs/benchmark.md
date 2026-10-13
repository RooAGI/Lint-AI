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
| Fractional Recall@5 | 86.37% |
| Fractional Recall@10 | 91.89% |
| Fractional Recall@20 | 92.93% |
| MRR | 86.97% |
| NDCG@10 | 85.11% |

These headline values are from the current 500-question Tantivy 0.25.0
LongMemEval-S run.

### Standalone session-level rerun (2026-10-12)

Benchmark runs are recorded under their run date. Add a new dated entry for
future runs and leave earlier results intact.

This release-profile working-tree run evaluated all 500 LongMemEval-S questions
with heuristic NER, structured-fact retrieval enabled, and beKIND disabled.
The benchmark scores unique sessions, while `MemoryService` returns document
memories. It therefore requests four times the maximum session cutoff (80
documents for Recall@20), deduplicates by session ID, then scores the first 20
unique sessions. This is a benchmark retrieval-depth change; it does not change
production search behavior.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 89.78% / 95.20% / 98.34% |
| Any-hit Recall@5 / @10 / @20 | 96.6% / 98.8% / 100.0% |
| MRR | 88.85% |
| NDCG@10 | 87.91% |
| Search latency, mean / p50 / p95 | 3.52 / 3.32 / 5.80 ms |

Compared with the same run before session-depth overfetch, fractional
Recall@20 rose from 96.33% to 98.34%, and Any-hit Recall@20 rose from 99.4%
to 100%. Recall@5 and Recall@10, MRR, and NDCG were effectively unchanged.
The three former top-20 misses ranked 19, 14, and 14. The hamster case is an
abstention answer and does not contain a supported hamster-name fact.

The reproducible command was:

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --ner-provider heuristic \
  --structured-fact-retrieval \
  --out scripts/longmemeval-500-misses-session-depth4-2026-10-12.json
```

The [JSON report](../scripts/longmemeval-500-misses-session-depth4-2026-10-12.json)
and [console log](../scripts/longmemeval-500-misses-session-depth4-2026-10-12.txt)
are retained in `scripts/`. This is a working-tree diagnostic, not a tagged
release result. The segmented benchmark was not rerun with this session-depth
protocol.

### Native event retrieval at normal depth (2026-10-12)

A separate full 500-question run used normal result depth with structured
native-event retrieval enabled. It reached Any-hit Recall@5 / @10 / @20 of
96.4% / 98.4% / 99.0%, and Fractional Recall@5 / @10 / @20 of
89.38% / 94.70% / 95.93%. MRR was 88.52%, NDCG@10 was 87.50%, and mean
search time was 3.50 ms. The 96.6% / 100% figures above belong to the
four-times-depth session diagnostic, not this normal-depth run.

The sports-events question retrieved all three relevant sessions in the top
three; the structured-retrieval-off baseline had none in the top 20. This
working-tree diagnostic is recorded separately in
[`benchmark-results.md`](benchmark-results.md#native-event-retrieval-run-2026-10-12),
with reports at
[`structured-native-events`](../comparison/results/retrieval-longmemeval-500-structured-native-events-2026-10-12.json)
and [`structured-retrieval-off`](../comparison/results/retrieval-longmemeval-500-nobekind-2026-10-12.json).

### Standalone rerun (2026-10-05)

A fresh release-profile run evaluated all 500 questions with the heuristic NER
backend, a single index, and no embeddings. It used Tantivy 0.25.0 and cutoffs
1, 3, 5, 10, and 20. The code tree contained local changes, so this is a
working-tree benchmark rather than a tagged-release result. The latest report
is [`retrieval-longmemeval-current-2026-10-05-write-publication-k20.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current-2026-10-05-write-publication-k20.json.gz).

| Metric | Result |
|---|---:|
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| MRR | 87.0% |
| NDCG@10 | 85.11% |
| Search latency, mean | 2.58 ms |

Latency here is measured around each `MemoryService::search` call. It excludes
the per-question haystack indexing performed before the query. The previous
13.0 ms figure below comes from the earlier published report and has a
different timing scope, so the two latency values are not directly comparable.

| Question type | n | Any-hit @5 | Any-hit @10 | Any-hit @20 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Single-session assistant | 56 | 100.0% | 100.0% | 100.0% | 99.1% | 99.3% |
| Single-session user | 70 | 97.1% | 98.6% | 98.6% | 88.2% | 90.9% |
| Single-session preference | 30 | 80.0% | 90.0% | 93.3% | 66.2% | 71.6% |
| Knowledge update | 78 | 100.0% | 100.0% | 100.0% | 96.2% | 94.3% |
| Temporal reasoning | 133 | 89.5% | 94.0% | 96.2% | 82.5% | 80.0% |
| Multi-session | 133 | 94.7% | 97.0% | 97.0% | 84.9% | 78.4% |

### Historical Tantivy 0.26.2 IndexStore refactor experiment (2026-10-05)

During the temporary Tantivy 0.26.2 upgrade, after removing the unused
`IndexStore` lexical writer, the same release-profile 500-question scoped
benchmark returned identical aggregate retrieval metrics
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
| Latest rerun command | `cargo run --release --bin haystack_scoped_benchmark -- --longmemeval benchmark/data/longmemeval_s_raw.json --k 1 --k 3 --k 5 --k 10 --k 20 --ner-provider heuristic --out comparison/results/retrieval-longmemeval-current-2026-10-05-write-publication-k20.json` |

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
questions. It used Tantivy 0.25.0, the heuristic backend, and the experimental
segmented feature. The single-index control and segment modes use the same
query set and harness. The source tree had local changes, so treat it as a
working-tree result rather than a tagged-release measurement.

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | Fractional Recall@5 | MRR | Average latency |
|---|---:|---:|---:|---:|---:|
| Segmented (fixed top-5) | 94.20% | 94.20% | 86.77% | 0.880 | 2.49 ms |
| Segmented (enriched + reranked) | 94.20% | 94.20% | 86.77% | 0.881 | 3.85 ms |
| Adaptive (enriched + reranked) | 94.00% | 96.00% | 86.30% | 0.884 | 7.26 ms |
| Intent baseline | 94.20% | 96.20% | 85.54% | 0.863 | 0.71 ms |
| Fused adaptive + global | 95.40% | 97.20% | 87.11% | 0.880 | 61.84 ms |
| Fused temporal + global | **96.00%** | **97.80%** | **88.51%** | **0.885** | 28.05 ms |
| Single-index control | 94.40% | 96.60% | 86.41% | 0.872 | 13.45 ms |

The run used `--segment-top-n 5`, `--adaptive-segment-max-n 12`, and the
`coverage-local` router. Latency is the benchmark's average per-query time for
each variant, excluding haystack indexing. The direct-index control's latency
is not comparable to standalone MemoryService latency above. Fused temporal +
global has the strongest recall and MRR in this table, with higher latency.
The intent baseline is fastest among the listed modes.

Full per-query report:
[`segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz).
The historical 133-query multi-session result remains at
[`segment-multisession-v0.2.0.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-multisession-v0.2.0.json).

### Reasoning-rules rerun (2026-10-12)

After moving reasoning cues into `data/reasoning_rules.json`, the full 500
questions were rerun with the same segmented harness and Tantivy 0.25.0. This
is a working-tree measurement. The fixed route searched the top five segments;
the adaptive route could expand to 12. Heuristic NER and the `coverage-local`
router were used.

| Mode | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Fixed top-5 segments | 94.2% | 94.2% | 86.84% | 0.887 | 4.85 ms |
| Fixed top-5, enriched + reranked | 94.2% | 94.2% | 86.84% | 0.882 | 6.30 ms |
| Adaptive 5→12, enriched + reranked | 94.8% | 96.4% | 87.06% | 0.888 | 9.66 ms |
| Intent | 94.8% | 96.4% | 86.40% | 0.869 | 1.31 ms |
| Fused temporal + global | **96.2%** | **98.4%** | **89.73%** | **0.895** | 41.59 ms |
| Single-index control | 95.8% | 98.0% | 88.94% | 0.885 | 13.49 ms |

Fused temporal + global had the strongest listed retrieval metrics and the
highest latency. These direct-index timings are not comparable to the
standalone MemoryService timing above.

Command:

```bash
cargo run --release --features experimental --bin segment_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --segment-compare \
  --segment-top-n 5 \
  --adaptive-segment-max-n 12 \
  --segment-router coverage-local \
  --ner-provider heuristic \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --out comparison/results/segment-longmemeval-500-2026-10-12-reasoning-rules.json.gz
```

Full report:
[`segment-longmemeval-500-2026-10-12-reasoning-rules.json.gz`](../comparison/results/segment-longmemeval-500-2026-10-12-reasoning-rules.json.gz).

### Default LocalDistinctiveness router comparison (2026-10-12)

The same comparison was rerun with the server's default
`LocalDistinctiveness` router. The fixed top-5 route was faster than the
`coverage-local` run, with lower recall and MRR. The global control queries all
segments, so its result does not depend on the segment router.

| Mode | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Fixed top-5 segments | 92.0% | 92.0% | 82.38% | 0.874 | 3.53 ms |
| Adaptive 5→12, enriched + reranked | 94.0% | 95.2% | 85.35% | 0.872 | 10.26 ms |
| Fused temporal + global | 95.0% | 98.2% | 87.25% | 0.887 | 42.50 ms |

Full report:
[`segment-longmemeval-500-2026-10-12-reasoning-rules-local-distinctiveness.json.gz`](../comparison/results/segment-longmemeval-500-2026-10-12-reasoning-rules-local-distinctiveness.json.gz).

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
