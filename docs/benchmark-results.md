# Lint-AI Retrieval and Corpus Benchmarks

This is Lint-AI's standalone benchmark track: it measures retrieval quality,
recency behavior, and corpus-scale performance without an agent client.

## Canonical LongMemEval-S comparison result

The public headline uses the fair 500-question comparison track: both systems
were evaluated with the same scorer and top-k protocol. These are **any-hit
recall** values, meaning a question scores as a hit when any relevant session
appears in the top *k*. The Lint-AI run uses the single index (not segmented),
the default heuristic backend, and no embedding vectors. Measured 2026-09-28
on commit dbf6496a.

| Metric | Result |
|---|---:|
| Any-hit Recall@5 | 94.2% |
| Any-hit Recall@10 | 96.8% |
| Any-hit Recall@20 | 97.6% |
| MRR | 87.0% |
| NDCG@10 | 85.0% |

The canonical comparison artifact is
[`retrieval-longmemeval-500.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-500.json).

For a standalone diagnostic using the current code and fractional recall (a
different metric from the comparison headline), see
[`retrieval-longmemeval-current.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current.json).
The rust-bert POS/NER branch is reported separately in
[`benchmark/README.md`](https://github.com/RooAGI/Lint-AI/blob/main/benchmark/README.md).

### Latest standalone rerun (2026-10-04)

The 500-question release run used the heuristic NER backend, single-index
mode, no embeddings, and cutoffs 1, 3, 5, 10, and 20. Its detailed report is
`comparison/results/retrieval-longmemeval-current-2026-10-04.json.gz`.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| MRR | 87.0% |
| NDCG@10 | 85.11% |
| Search latency, mean / p50 / p95 | 2.68 / 2.15 / 4.47 ms |

The latency measures the `MemoryService::search` call and excludes indexing the
question's haystack. The older published 13.0 ms latency uses a different
timing scope, so it should not be compared directly.

## Segmented-index comparison (500 questions)

The 2026-10-04 segmented benchmark evaluated all 500 eligible LongMemEval-S
questions. These variants share the same dataset and harness, including a
single-index control:

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | Fractional Recall@5 | MRR | Average latency |
|---|---:|---:|---:|---:|---:|
| Segmented (fixed top-5) | 94.20% | 94.20% | **86.77%** | 0.880 | 4.08 ms |
| Segmented (adaptive 5→12, enriched) | 94.20% | 96.20% | 85.64% | 0.869 | 7.05 ms |
| Intent baseline | 94.20% | 96.20% | 85.54% | 0.863 | 0.77 ms |
| Fused adaptive + global | 95.40% | 97.20% | 87.11% | 0.880 | 29.47 ms |
| Fused temporal + global | **96.00%** | **97.80%** | **88.51%** | **0.885** | 26.98 ms |
| Single-index control | 94.40% | 96.60% | 86.41% | **0.872** | 14.34 ms |

The run used the heuristic backend, fixed top-5 routing, adaptive expansion to
12 segments, and the `coverage-local` router. Latency is per-query benchmark
time excluding haystack indexing. These direct-index timings are not comparable
to the standalone MemoryService timing in the 500-question headline above.
Fused temporal + global had the strongest recall and MRR among these listed
variants, with higher latency; the intent baseline was the fastest listed.

Full report: [`segment-longmemeval-500-2026-10-04.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-longmemeval-500-2026-10-04.json.gz).


### Tantivy 0.26.2 segmented rerun (2026-10-05)

The same 500-question segmented comparison was rerun with Tantivy 0.26.2 and
the experimental feature. Compared with the October 4 Tantivy 0.25.0 run,
fixed top-5 keeps Recall@5 and Fractional@5, while MRR falls from 0.880 to
0.858. Fused temporal + global remains strongest among these selected routes,
with lower recall and MRR than before the upgrade.

| Mode | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Routed fixed top-5 | 94.2% | 94.2% | 86.77% | 0.858 | 1.84 ms |
| Intent | 93.8% | 96.4% | 84.73% | 0.838 | 0.66 ms |
| Fused adaptive + global | 94.8% | 97.0% | 85.43% | 0.856 | 20.73 ms |
| Fused temporal + global | **95.2%** | **97.6%** | **87.07%** | **0.865** | 18.78 ms |
| Single-index control | 92.6% | 96.2% | 82.89% | 0.837 | 13.28 ms |

Full report: [segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz](../comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz).

The earlier 133-query multi-session comparison is retained as a historical
slice in [`segment-multisession-v0.2.0.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-multisession-v0.2.0.json).
## Segment router comparison (full 500, top_n=5)

Seven routing strategies were compared on the full 500-question LongMemEval-S
set with `segment_top_n=5`, release build, routed-only (no global fusion).
Adaptive and temporal paths reported separately:

| Router | Adaptive any@5 | Adaptive frac@5 | Adaptive MRR | Temporal any@5 | Temporal frac@5 | Temporal MRR |
|---|---:|---:|---:|---:|---:|---:|
| sparse | .8900 | .7915 | .8201 | .9020 | .7967 | .8221 |
| coverage-local | .9260 | .8442 | .8411 | .9380 | .8484 | .8436 |
| coverage-team | .9340 | .8225 | .8398 | .9440 | .8272 | .8417 |
| team-coverage-local | .9280 | .8390 | .8424 | .9340 | .8419 | .8444 |
| typed-evidence additive | .9200 | .8321 | .8366 | .9320 | .8346 | .8391 |
| **gated coverage-local** | .9280 | **.8464** | .8419 | .9400 | **.8499** | .8443 |
| **gated coverage-team** | **.9360** | .8294 | **.8442** | **.9460** | .8330 | **.8461** |

Gated coverage-team leads routed-only any-recall and MRR; gated coverage-local
leads fractional evidence recall at about half the latency (~9ms vs ~19ms).

### Gated router formulas

Both gated routers multiply a content score by a typed-evidence factor that
approaches but stays below 2x. The gate is genuine: a zero content score stays
zero, so typed evidence can amplify but never elect a content-free segment.

Gated coverage-local (`TypedEvidenceMultiplicative`):

```text
typed_factor = typed_score / (typed_score + 4.0)
score = coverage_local_score × (1 + typed_factor)
```

Gated coverage-team (`CoverageTeamTypedMultiplicative`): greedy team selection
maximizing marginal coverage, then gated the same way:

```text
combined = marginal_team_coverage + coverage_local_score × 1.6
score = combined × (1 + typed_factor)
```

## Production path: fused head-to-head

The production query path fuses the routed arm with a corpus-wide all-segments
arm via reciprocal rank fusion (RRF, k=60). Three routers were compared on the
same binary, full 500, top_n=5:

| Router | Fused adap any@5 | Fused adap MRR | Fused temp any@5 | Fused temp MRR |
|---|---:|---:|---:|---:|
| sparse | **.9420** | .8434 | .9380 | .8474 |
| gated coverage-local | .9340 | .8570 | .9360 | .8603 |
| gated coverage-team | .9380 | **.8605** | **.9420** | **.8642** |

Sparse still leads fused any-recall@5 on the adaptive path, but gated-team wins
MRR on both paths and temporal any@5. Latency on the fused path is dominated by
the all-segments arm (~70ms total); the routed-arm difference (9ms vs 19ms) is
noise against it.

### Fusion lift collapses with better routers

The marginal value of the global arm depends strongly on the router. Adaptive
any@5, routed-only → fused:

| Router | Routed-only | Fused | Lift (questions) |
|---|---:|---:|---:|
| sparse | .8900 | .9420 | **+26** |
| gated coverage-local | .9280 | .9340 | +3 |
| gated coverage-team | .9360 | .9380 | +1 |

With a strong router, the global arm rescues almost nothing while costing most
of the query latency. This motivated making fusion configurable (see below).

## Query modes: routed-only by default

`PipelineOptions.fuse_global_arm` (default `false`) controls whether the
corpus-wide arm runs. The server exposes it as `--fuse-global`:

- **Routed-only mode** (default): routed arm alone, using the gated
  coverage-local router. Measures .9280 any@5 at ~9ms per query on the full
  500-question set.
- **Fused mode** (`--fuse-global`): routed arm + all-segments arm via RRF.
  Best recall with weak routers; ~70ms per query. With gated coverage-local
  the lift is only +3 questions out of 500, so fusion is opt-in.

## LoCoMo agentic evaluation

A 150-question agentic-reader evaluation on the LoCoMo dataset, replicating
Letta's published methodology: the reader (`gpt-5.6-luna`) must start with
search, may run up to 3 search rounds with up to 3 keyword queries per round
(top-3 sessions per query), then answers. An independent Luna instance judges
binary answer correctness against the reference.

| Category | Accuracy |
|---|---:|
| Overall | 97/150 = 64.7% |
| Single-hop | 65/77 = 84.4% |
| Multi-hop | 15/37 = 40.5% |
| Temporal | 15/30 = 50.0% |
| Open-domain | 2/6 = 33.3% |

Caveats: the harness differs from Letta's direct agentic loop; reader and judge
both used `gpt-5.6-luna`; the sample was conversation-balanced rather than
category-stratified. Competitor numbers (Letta 74.0%, mem0 94.4%) use different
models, judges, and protocols and are not directly comparable.

## Temporal retrieval fixes

Three fixes landed for temporal queries, validated on the 133
`temporal-reasoning` questions:

1. **Anchor resolution**: relative phrases ("two weeks ago") are resolved
   against the reference date and wired into the production path (previously
   the benchmark-only fix never ran for real users). +2 net adaptive,
   +4 temporal-path, no regressions beyond one.
2. **Saturation fix**: the old temporal boost summed per-record proximity capped
   at 2.0, which saturated when many sessions fell in-window and drowned out
   content scores. Rewritten as a max-per-segment multiplicative factor in
   [0.0, 1.0] — temporal evidence amplifies content but can never elect a
   zero-content segment.
3. **Span-aware windows**: point anchors ("last Tuesday") use a ±7d window;
   range anchors ("past two months") use [anchor, reference]. Fixed a
   regression where "past two months" was treated as a 7-day point window.

## Negative results

- **Intent-based retrieval** (`reveal_question()` operand fan-out): clean
  negative. The parser works (86/500 questions reveal structured multi-operand
  ops), but the full-500 arm fired on only 24/500 with 0 rescues and 1
  regression — most misses are single-operand lookups where fan-out is
  structurally inapplicable. Kept as answer-composition infrastructure only.
- **Entity join**: rejected in favor of enhancement; no retrieval gain.

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

## Tantivy upgrade throughput verification

The paired 0.25.0 versus 0.26.2 check uses 23,366 records, five sessions, five
repetitions and 1,000 requests per cell. Single-index C=10 medians are
2,246.69 → 2,769.53 req/s; routed medians are 888.86 → 901.29 req/s.
See [Tantivy upgrade verification](releases/tantivy-0.26.2-verification.md)
for the full protocol, compatibility qualifications and raw artifact locations.
