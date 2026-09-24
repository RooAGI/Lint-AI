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
