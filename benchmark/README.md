# Benchmark Suite

This directory holds the retrieval benchmarks for `lint-ai`, including haystack-style corpora and LongMemEval-S runs.

## Benchmark Results

Evaluated on **LongMemEval-S** (500 questions), a public benchmark for long-context agent memory retrieval over multi-session conversation corpora. The scoped variant is used: each query searches only the sessions attached to that question, matching the realistic setting where a system knows which sessions are candidates for a given user. No embedding vectors are used anywhere in the pipeline.

The section below shows both the rust-bert POS/NER branch result and the default heuristic release result.

### Rust-BERT POS/NER Branch

**Aggregate (n=500):**

| metric | value |
|---|---|
| recall@5 | 86.9% |
| recall@10 | 93.8% |
| recall@20 | 94.4% |
| recall_any@5 | 94.8% |
| recall_any@10 | 98.2% |
| MRR | 87.1% |
| NDCG@10 | 86.0% |
| avg query latency | 5.1 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

| question type | n | recall@5 | recall@10 | recall_any@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 99.1% | 99.3% |
| single-session-user | 70 | 97.1% | 98.6% | 97.1% | 86.8% | 89.8% |
| knowledge-update | 78 | 96.8% | 98.7% | 100.0% | 95.2% | 94.4% |
| single-session-preference | 30 | 80.0% | 96.7% | 80.0% | 64.7% | 72.3% |
| temporal-reasoning | 133 | 81.1% | 92.0% | 91.7% | 84.1% | 82.3% |
| multi-session | 133 | 77.7% | 87.1% | 94.7% | 85.3% | 80.2% |

Results file: `benchmark/data/lintai_longmemeval_scoped_results_0512.json`

### Heuristic Release Backend

**Aggregate (n=500):**

| metric | value |
|---|---|
| recall@5 | 83.7% |
| recall@10 | 89.6% |
| recall@20 | 91.1% |
| recall_any@5 | 92.4% |
| recall_any@10 | 95.6% |
| recall_any@20 | 97.0% |
| MRR | 84.3% |
| NDCG@10 | 81.9% |
| avg query latency | 6.1 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

| question type | n | recall@5 | recall@10 | recall_any@5 | MRR | NDCG@10 |
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

## Report Metrics

- `recall_at_k`: average recall at each configured K.
- `recall_any_at_k`: 1.0 when any gold session is in the top K, otherwise 0.0.
- `mrr`: mean reciprocal rank.
- `ndcg_at_10`: normalized DCG at 10.

The output JSON includes both aggregate metrics and per-query details.
