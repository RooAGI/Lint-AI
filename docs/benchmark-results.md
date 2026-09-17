# Lint-AI Retrieval and Corpus Benchmarks

This is Lint-AI's standalone benchmark track: it measures retrieval quality,
recency behavior, and corpus-scale performance without an agent client.

## Canonical LongMemEval-S comparison result

The public headline uses the fair 500-question comparison track: both systems
were evaluated with the same scorer and top-k protocol. These are **any-hit
recall** values, meaning a question scores as a hit when any relevant session
appears in the top *k*. The Lint-AI run uses the default heuristic backend and
no embedding vectors.

| Metric | Result |
|---|---:|
| Any-hit Recall@5 | 92.4% |
| Any-hit Recall@10 | 95.6% |
| Any-hit Recall@20 | 97.0% |
| MRR | 84.0% |
| NDCG@10 | 81.8% |

The canonical comparison artifact is
[`retrieval-longmemeval-500.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-500.json).

For a standalone diagnostic using the current code and fractional recall (a
different metric from the comparison headline), see
[`retrieval-longmemeval-current.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current.json).
The rust-bert POS/NER branch is reported separately in
[`benchmark/README.md`](https://github.com/RooAGI/Lint-AI/blob/main/benchmark/README.md).

## Segmented-index comparison

The latest 133-query multi-session run compares the supported index modes:

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | MRR | Average latency |
|---|---:|---:|---:|---:|
| Segmented (fixed top-5) | **95.49%** | 95.49% | **0.859** | **1.25 ms** |
| Segmented (adaptive 5→12) | **96.24%** | **96.24%** | 0.839 | 4.36 ms |
| Single index | 93.23% | 93.98% | 0.814 | 6.41 ms |

These results are scoped to the multi-session slice and are not interchangeable
with the 500-question aggregate headline above.

The checked-in summary is
[`segment-multisession-v0.2.0.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-multisession-v0.2.0.json).

## Reproduce

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --out benchmark/data/lintai_longmemeval_scoped_results.json

cargo run --release --bin corpus_scale_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --sizes 1000,5000,10000,25000 --queries 100
```

For the full recorded outputs and comparison methodology, see the repository
[`comparison/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison) folder.
