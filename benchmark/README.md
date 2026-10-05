# Benchmark Suite

This directory holds the retrieval benchmarks for `lint-ai`, including
haystack-style corpora, LongMemEval-S runs, and agent integration performance
scaffolds.

For the reproducible Lint-AI versus AgentMemory service comparison, including
latency scripts, seeders, recorded JSON results, and methodology, see
[`comparison/`](../comparison/).

## Strong evaluation (pillars 1–4)

The full evaluation protocol — retrieval quality, efficiency/cost, write-path
scaling, and LLM-judged answer quality — lives in
[`strong_eval/`](strong_eval/). Five independent runs per pillar, mean ± std
dev, with pinned dataset SHA, code commit, and hardware. Start there for any
published claim.

## Conversation Recency Benchmark

Run the controlled end-to-end freshness evaluation with:

```bash
cargo run --release --bin recency_benchmark
```

The benchmark creates identical fresh (0–30 days), warm (31–90 days), and
cold (91+ days) conversation memories for five topics. It reports whether the
fresh memory is ranked first or within the top three, its mean rank, the
recency score contribution, and the freshness distribution of top-1 results.
This is intended as a deterministic regression check for ranking behavior and
does not require external datasets or model downloads.

To evaluate the real persisted Lint-AI project memory corpus instead:

```bash
cargo run --bin project_memory_recency_benchmark -- \
  --records .lint-ai/memory/semantic/records.json
```

This reports the corpus freshness distribution and the freshness distribution
of the top-1 and top-5 results for representative project-memory queries. The
real corpus has no gold answer labels, so this measures freshness behavior and
does not claim semantic answer accuracy.

The pipeline also includes a regression case for Markdown-like project guidance
whose wording changes between revisions. It asks a natural-language ownership
question and verifies that newer guidance outranks older, still-relevant wording
while retaining the temporal score signal. This guards the current-state use case
independently of the LongMemEval conversation benchmark.

Benchmark subdirectories:

- `claude_code/`: Claude Code A/B scenarios and parser scaffold
- `codex_code/`: Codex A/B scenarios and parser scaffold

## Corpus-scale benchmark

Measure how build and retrieval behavior changes as the indexed corpus grows:

```bash
cargo run --release --bin corpus_scale_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --sizes 1000,5000,10000,25000 --queries 100
```

The output is CSV with unique session count, eligible labeled queries, index
build time, p50/p95 query latency, and top-k any-hit recall. Queries use the
dataset's candidate-session IDs as a filter against the global index, matching
the scoped retrieval contract while measuring a single large index. A query is eligible
only when all of its labeled answer sessions are present in the corpus slice;
this avoids penalizing smaller slices for missing gold records. This is a global
corpus stress test, not the question-scoped retrieval benchmark reported above,
and it does not claim production capacity. Rerun it on the target machine when
comparing hardware or revisions.

## Benchmark Results

The rust-bert POS/NER branch is retained as a separate experimental quality
run and should not be mixed into the homepage headline. The current
revalidation is recorded in
[`comparison/results/retrieval-longmemeval-current.json`](../comparison/results/retrieval-longmemeval-current.json);
the historical tables below are retained for comparison. Service-load
measurements (including concurrency) are documented in
[`comparison/`](../comparison/).

Evaluated on **LongMemEval-S** (500 questions), a public benchmark for long-context agent memory retrieval over multi-session conversation corpora. The scoped variant is used: each query searches only the sessions attached to that question, matching the realistic setting where a system knows which sessions are candidates for a given user. No embedding vectors are used anywhere in the pipeline.

The section below shows both the rust-bert POS/NER branch result and the default heuristic release result.

### Latest standalone rerun (2026-10-04)

A fresh release-mode run evaluated all 500 questions with the heuristic NER
backend, a single index, no embeddings, and cutoffs 1, 3, 5, 10, and 20. The
per-question report is saved at
`comparison/results/retrieval-longmemeval-current-2026-10-04.json.gz`.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| MRR | 87.0% |
| NDCG@10 | 85.11% |
| Search latency, mean / p50 / p95 | 2.68 / 2.15 / 4.47 ms |

Latency measures the `MemoryService::search` call and excludes per-question
haystack indexing. The older published 13.0 ms value has a different timing
scope and is not directly comparable.

### Rust-BERT POS/NER Branch

**Aggregate (n=500):**

| metric | value |
|---|---|
| Fractional recall@5 | 86.9% |
| Fractional recall@10 | 93.8% |
| Fractional recall@20 | 94.4% |
| recall_any@5 | 94.8% |
| recall_any@10 | 98.2% |
| MRR | 87.1% |
| NDCG@10 | 86.0% |
| avg query latency | 5.1 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

| question type | n | fractional recall@5 | fractional recall@10 | any-hit recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 99.1% | 99.3% |
| single-session-user | 70 | 97.1% | 98.6% | 97.1% | 86.8% | 89.8% |
| knowledge-update | 78 | 96.8% | 98.7% | 100.0% | 95.2% | 94.4% |
| single-session-preference | 30 | 80.0% | 96.7% | 80.0% | 64.7% | 72.3% |
| temporal-reasoning | 133 | 81.1% | 92.0% | 91.7% | 84.1% | 82.3% |
| multi-session | 133 | 77.7% | 87.1% | 94.7% | 85.3% | 80.2% |

Results file: `benchmark/data/lintai_longmemeval_scoped_results_0512.json`

### Heuristic Release Backend (current revalidation)

**Aggregate (n=500):**

| metric | value |
|---|---|
| Fractional recall@5 | 83.5% |
| Fractional recall@10 | 89.5% |
| Fractional recall@20 | 91.1% |
| recall_any@5 | 92.4% |
| recall_any@10 | 95.6% |
| recall_any@20 | 97.0% |
| MRR | 84.0% |
| NDCG@10 | 81.8% |
| avg query latency | 1.9 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

This breakdown is retained from the earlier heuristic run; reproduce the
current aggregate above with the command in the Reproduce section. The
published cross-system comparison uses the any-hit aggregate and the shared
scorer, not this historical per-type table.

| question type | n | fractional recall@5 | fractional recall@10 | any-hit recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 98.2% | 98.7% |
| single-session-user | 70 | 94.3% | 98.6% | 94.3% | 77.4% | 82.6% |
| knowledge-update | 78 | 94.2% | 96.8% | 98.7% | 94.6% | 92.4% |
| single-session-preference | 30 | 80.0% | 93.3% | 80.0% | 65.8% | 72.2% |
| temporal-reasoning | 133 | 77.0% | 85.7% | 86.5% | 81.8% | 78.2% |
| multi-session | 133 | 72.5% | 79.3% | 93.2% | 82.7% | 74.3% |

Results file: `benchmark/data/lintai_longmemeval_scoped_results.json`

## Dataset format

Provide a JSON file with this shape:

```json
{
  "documents": [
    {
      "doc_id": "doc-1",
      "source": "docs/alpha.md",
      "content": "# Title\nDocument text...",
      "concept": "optional-concept",
      "headings": ["Title"],
      "links": ["docs/beta.md"],
      "timestamp": "2026-04-25T00:00:00Z",
      "author_agent": "optional"
    }
  ],
  "queries": [
    {
      "id": "q-1",
      "query": "what does alpha say",
      "relevant_doc_ids": ["doc-1"],
      "relevant_chunk_ids": []
    }
  ]
}
```

Notes:
- `relevant_doc_ids` and `relevant_chunk_ids` are both optional arrays.
- If only `relevant_chunk_ids` are provided, the benchmark resolves them to `doc_id` via indexed chunk metadata.

## Haystack Run

```bash
cargo run --bin haystack_benchmark -- \
  --dataset benchmark/sample_dataset.json \
  --k 1 --k 3 --k 5 --k 10 \
  --out benchmark/results.json
```

To run the academic LongMemEval-S corpus directly and index each session as turn-level chunks, use the raw Hugging Face copy:

```bash
cargo run --bin haystack_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --limit 20 \
  --k 5 --k 10 --k 20 \
  --out benchmark/results.json
```

To refresh that local raw file from Hugging Face, run `python3 benchmark/download_longmemeval_raw.py`.

Note: the benchmark expects the raw LongMemEval-S source file at `benchmark/data/longmemeval_s_raw.json`. The helper script refreshes that file and verifies it against the published Hugging Face blob.
Source: https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s?download=true

If you want question-scoped haystack retrieval, where each query only searches the sessions attached to that question:

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 5 --k 10 --k 20 \
  --out benchmark/data/lintai_longmemeval_scoped_results.json
```

Scoped LongMemEval reporting includes both `recall@k` and `recall_any@k`. The any-hit metric matches the interpretation used by the current LongMemEval-S release notes and README.

To reproduce the published Home/benchmark/comparison headline, use the three
cutoffs above and score both per-question outputs with the shared evaluator:

```bash
python3 benchmark/score_retrieval.py \
  --dataset benchmark/data/longmemeval_s_raw.json \
  --lintai benchmark/data/lintai_longmemeval_scoped_results.json \
  --agentmemory /path/to/agentmemory/benchmark/data/longmemeval_results_bm25.json
```

The comparison repository records the exact Lint-AI and AgentMemory artifacts
and reports the resulting any-hit Recall@5/10/20, MRR, and NDCG@10. Do not
compare a run that only requested top-10 with the top-20 headline.

For a full 500-question segmented comparison, use the separate segmented
benchmark. It evaluates segmented variants alongside a same-harness global
single-index control:

```bash
cargo run --release --features experimental --bin segment_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --segment-compare \
  --segment-top-n 5 \
  --adaptive-segment-max-n 12 \
  --segment-router coverage-local \
  --ner-provider heuristic \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --out comparison/results/segment-longmemeval-500-2026-10-04.json.gz
```

The segmented benchmark reports experimental segment variants,
segment-specific enrichment diagnostics, and router-miss failure analysis.

The 500-question run produced:

| mode | candidates | recall_any@5 | recall_any@10 | fractional recall@5 | MRR | average latency |
|---|---|---:|---:|---:|---:|---:|
| segmented (fixed) | top-5 | 94.20% | 94.20% | **86.77%** | 0.880 | 4.08 ms |
| segmented (adaptive, enriched) | 5 → 12 | 94.20% | **96.20%** | 85.64% | 0.869 | 7.05 ms |
| intent baseline | intent | 94.20% | 96.20% | 85.54% | 0.863 | 0.77 ms |
| fused adaptive + global | fused | 95.40% | 97.20% | 87.11% | 0.880 | 29.47 ms |
| fused temporal + global | fused | **96.00%** | **97.80%** | **88.51%** | **0.885** | 26.98 ms |
| single-index control | global search | **94.40%** | 96.60% | 86.41% | **0.872** | 14.34 ms |

Mean latency is measured per query and excludes haystack indexing. These
direct-index timings use a different harness and scope from the standalone
MemoryService latency above. Fused temporal + global has the strongest recall
and MRR among the listed modes, with higher latency; the intent baseline is
fastest. The prior 133-question multi-session comparison
remains in `comparison/results/segment-multisession-v0.2.0.json`.

Full per-query output: `comparison/results/segment-longmemeval-500-2026-10-04.json.gz`.

### Tantivy 0.26.2 segmented rerun (2026-10-05)

With the experimental feature enabled, fixed top-5 scored 94.2% Any-hit@5,
94.2% Any-hit@10, 86.77% Fractional@5 and 0.858 MRR (1.84 ms). Fused temporal
+ global scored 95.2%, 97.6%, 87.07% and 0.865 MRR (18.78 ms). The single-index
control scored 92.6%, 96.2%, 82.89% and 0.837 MRR (13.28 ms). Compared with the
October 4 Tantivy 0.25.0 run, MRR is lower for these strategies. Full artifact:
[segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz](../comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz).

## Report Metrics

- `recall_at_k`: average recall at each configured K.
- `recall_any_at_k`: 1.0 when any gold session is in the top K, otherwise 0.0.
- `mrr`: mean reciprocal rank.
- `ndcg_at_10`: normalized DCG at 10.

The output JSON includes both aggregate metrics and per-query details.
