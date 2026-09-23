# Strong evaluation (pillars 1–4)

Reproducible end-to-end evaluation of Lint-AI retrieval, efficiency, and the
write path. Every pillar runs N independent repetitions (default 5) and
reports mean ± population standard deviation. Pin the dataset SHA, the code
commit, and the hardware with every published result, and keep the raw logs.

## Pillar 1 — retrieval quality (LongMemEval-S)

500 questions over 19,195 sessions, question-scoped retrieval. Metrics:
fractional and any-hit recall@1/3/5/10/20, MRR, NDCG@10.

```bash
cargo build --release --bin haystack_scoped_benchmark
DATA=/path/to/longmemeval_s_cleaned.json bash benchmark/strong_eval/run_pillar1.sh
python3 benchmark/strong_eval/analyze_pillars.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --out benchmark/results/strong_eval/pillar12_metrics.json \
  benchmark/results/strong_eval/haystack_run*.json
```

Run 1 is wrapped in `run_with_peak_rss.py`, which samples `/proc/PID/status`
VmHWM for peak RSS (Linux only). Feed the run JSONs to `analyze_pillars.py`
in any order; failed runs must be excluded or redone, never silently kept.

## Pillar 2 — efficiency / cost

Derived from the pillar-1 runs, no extra benchmark needed:

- per-query latency mean / p50 / p95 and queries/sec
- retrieved context tokens per query at top-5/10/20 (offline, chars ÷ 4 over
  the retrieved session texts)
- peak RSS from the run-1 wrapper (single sample — label it as such)
- estimated cost per 1K queries on commodity hardware; state the $/hr
  assumption explicitly

## Pillar 3 — write path scaling

Cold refresh vs. single-write incremental refresh at 1K / 2.5K / 5K / 10K
documents, 5 runs each. The 10K batch is skipped when available RAM is under
2GB (guard in the script).

```bash
cargo build --release --bin refresh_scaling_benchmark
DATA=/path/to/longmemeval_s_cleaned.json bash benchmark/strong_eval/run_pillar3.sh
```

## Pillar 4 — end-to-end answer quality (LLM judge)

Fixed-model, open-protocol judging. Build the judge inputs from one canonical
pillar-1 run (question + reference answer + top-k retrieved context):

```bash
python3 benchmark/strong_eval/build_pillar4_inputs.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --run benchmark/results/strong_eval/haystack_run1.json \
  --out judge_inputs.jsonl --k 20
```

Then, on a machine with an authenticated model CLI: batch 20 questions per
invocation. Pass 1 generates an answer from the retrieved context only and
must return `INSUFFICIENT CONTEXT` when unsupported. Pass 2 independently
judges the generated answer against the reference with strict factual 1/0
scoring. Publish the exact model string, prompts, k, and batch size.

## LoCoMo

`src/bin/locomo_benchmark.rs` ingests each LoCoMo conversation
(`locomo10.json` from snap-research/locomo) as turn-level documents, builds
one snapshot per conversation, and scores per-question retrieval against the
evidence pointers (`D{session}:{turn}`). Same recall/MRR/NDCG formulas as
pillar 1, reported overall, per category, and excluding the adversarial
(abstention) subset.

```bash
cargo run --release --bin locomo_benchmark -- \
  --locomo /path/to/locomo10.json \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --out benchmark/results/strong_eval/locomo_run1.json
```

## Reference results

2026-09-20, PR #78 (`70c0c58b`, tree `d5875a79`), dataset SHA-256
`d6f21ea9…`, 2-CPU / 7.9GB box, 5 independent runs:

- Pillar 1: any-hit recall @1 78.0% @3 89.2% @5 92.4% @10 95.8% @20 97.2%;
  MRR 84.1%; NDCG@10 81.9% (±0.0000 across runs)
- Pillar 2: 7.80ms mean (p50 6.08, p95 13.87) → 128.2 q/s; ~31K tokens/query
  at top-20; peak RSS ~626MB; ≈$0.00011 / 1K queries at $0.05/hr
- LoCoMo (1,986 QA): any-hit @5 77.9%, @10 87.8%, MRR 61.8%, 3.06ms/query;
  no-adversarial 1,540q: any-hit @5 77.1%
- Pillar 3 cold / single-write medians: 1K 18.9s/407ms, 2.5K 47.9s/1005ms,
  5K 96.1s/2007ms, 10K 200.3s/4422ms (roughly linear)
