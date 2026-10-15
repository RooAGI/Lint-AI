# Membrick Retrieval and Corpus Benchmarks

This is Membrick's standalone benchmark track: it measures retrieval quality,
recency behavior, and corpus-scale performance without an agent client.

## Latest standalone LongMemEval-S result

This latest 500-question single-index run is the current standalone headline.
The fair cross-system comparison and segmented route experiments below are
different benchmark tracks and retain their own reports.

<!-- BEGIN GENERATED LONGMEMEVAL-S 500-QUESTION HEADLINE -->
**LongMemEval-S, 500 questions · 2026-10-15 · single index · heuristic NER · structured-fact retrieval · no embeddings**
Working-tree run based on `b204b7e`; not a clean release measurement.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 89.78% / 95.20% / 98.34% |
| Any-hit Recall@5 / @10 / @20 | 96.6% / 98.8% / 100.0% |
| MRR | 88.85% |
| NDCG@10 | 87.91% |
| Mean search time | 3.39 ms |

Source report: [`retrieval-longmemeval-500-2026-10-15-v0.6.0-b204b7e-working-tree.json`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500-2026-10-15-v0.6.0-b204b7e-working-tree.json).
<!-- END GENERATED LONGMEMEVAL-S 500-QUESTION HEADLINE -->

## Canonical LongMemEval-S comparison result

The public headline uses the fair 500-question comparison track: both systems
were evaluated with the same scorer and top-k protocol. These are **any-hit
recall** values, meaning a question scores as a hit when any relevant session
appears in the top *k*. The Membrick run uses the single index (not segmented),
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
[`retrieval-longmemeval-500.json`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500.json).

## Miss-analysis run (2026-10-12)

This release-profile working-tree diagnostic evaluated all 500 questions with
heuristic NER, structured-fact retrieval enabled, and beKIND disabled. Since
LongMemEval scores unique sessions but `MemoryService` returns document
memories, the harness requested four times the maximum cutoff (80 documents
for @20), deduplicated by session ID, and scored the first 20 unique sessions.
This changes the benchmark retrieval depth, not production search behavior.
Structured-fact retrieval is now the benchmark default; pass
`--no-structured-fact-retrieval` for a lexical-only baseline.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 89.78% / 95.20% / 98.34% |
| Any-hit Recall@5 / @10 / @20 | 96.6% / 98.8% / 100.0% |
| MRR / NDCG@10 | 88.85% / 87.91% |
| Search latency, mean / p50 / p95 | 3.52 / 3.32 / 5.80 ms |

The three former top-20 misses ranked 19, 14, and 14. The hamster case is an
abstention answer without a supported hamster-name fact. The segmented path was
not rerun with this session-depth protocol. The [JSON report](https://github.com/RooAGI/Membrick/blob/main/scripts/longmemeval-500-misses-session-depth4-2026-10-12.json)
is retained under `scripts/`; the local console log is not part of the release artifacts.

Reproduce it with:

```bash
cargo run --release --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --ner-provider heuristic \
  --structured-fact-retrieval \
  --out scripts/longmemeval-500-misses-session-depth4-2026-10-12.json
```

For a standalone diagnostic using the current code and fractional recall (a
different metric from the comparison headline), see
[`retrieval-longmemeval-current.json`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-current.json).
The rust-bert POS/NER branch is reported separately in
[`benchmark/README.md`](https://github.com/RooAGI/Membrick/blob/main/benchmark/README.md).

## v0.5.0 release-candidate rerun (2026-10-14)

A release-profile run evaluated all 500 LongMemEval-S questions on the v0.5.0
candidate working tree. It used the single index, heuristic NER, structured-fact
retrieval enabled, beKIND disabled, no embeddings, and cutoffs 1, 3, 5, 10,
and 20.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 89.78% / 95.20% / 98.34% |
| Any-hit Recall@5 / @10 / @20 | 96.6% / 98.8% / 100.0% |
| MRR / NDCG@10 | 88.85% / 87.91% |
| Mean search time | 3.62 ms |

Compared with the Oct 13 release-candidate rerun, Recall, MRR, and Any-hit
results are unchanged at displayed precision. Mean search time changed from
5.59 ms to 3.62 ms. The top-20 lists differ for 301 questions, but all
per-question MRR values are unchanged, only three per-question NDCG@10 values
change, and no question has an empty result or loses an Any-hit@5. Differences
are among low-ranked results and do not affect the first relevant rank.

Reproduce with:

```bash
cargo run --release --locked --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --k 1 --k 3 --k 5 --k 10 --k 20 \
  --ner-provider heuristic --structured-fact-retrieval \
  --out comparison/results/retrieval-longmemeval-500-2026-10-14-v0.5.0.json
```

The complete per-question report is available as a
[compressed JSON artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500-2026-10-14-v0.5.0.json.gz).
The initial run was measured on an Apple Silicon macOS host from candidate
revision `52675e5` with local release-preparation changes in the working tree.
The same 500-question run was repeated on the pushed PR head `5d650067` with
the command above. Recall, MRR, and NDCG matched exactly. Four top-20 result
lists differed from the initial candidate report, with no MRR or NDCG change,
no empty result lists, and no Any-hit@5 regressions. Mean search time was
3.52 ms on this run. The complete exact-head report is available as a
[compressed JSON artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500-2026-10-14-v0.5.0-5d650067.json.gz).

### Matched v0.4.0 tag comparison

I also ran the `v0.4.0` tag (`b46291b2`) on the same Apple Silicon host, dataset,
and settings, explicitly passing `--structured-fact-retrieval` because that
tag's default is off. Retrieval quality matched the exact-head v0.5.0 run:

| Metric | v0.4.0 tag | v0.5.0 candidate |
|---|---:|---:|
| Fractional Recall@5 / @10 / @20 | 89.78% / 95.20% / 98.34% | 89.78% / 95.20% / 98.34% |
| Any-hit Recall@5 / @10 / @20 | 96.6% / 98.8% / 100.0% | 96.6% / 98.8% / 100.0% |
| MRR | 0.8885 | 0.8885 |
| NDCG@10 | 0.8791 | 0.8791 |
| Mean search time | 3.54 ms | 3.52 ms |

Three questions had a different top-20 list. Their MRR and NDCG stayed the
same; no result list was empty and no question lost Any-hit@5. Treat the small
latency difference as noise. The v0.4.0 tag has no committed `Cargo.lock`, so
Cargo resolved its latest compatible dependencies offline; this limits exact
dependency parity and makes latency a diagnostic. The quality comparison used
the same dataset, query set, top-k cutoffs, heuristic NER, single-index mode,
and structured-fact retrieval. The baseline's complete per-question report is
available as a [compressed JSON artifact](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500-2026-10-14-v0.4.0.json.gz).

### Corpus-scale retrieval

The candidate's corpus-scale run used the LongMemEval-S session corpus, no
embeddings, and 100 eligible questions per size where available. The corpus
contains 19,829 unique sessions, so the requested 25,000 size uses all 19,829.
The executable reports index build time, query p50/p95, and any-hit recall@10.

| Sessions | Queries | v0.4.0 build | v0.5.0 build | v0.4.0 query p50 / p95 | v0.5.0 query p50 / p95 | Any-hit recall@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 20 | 1.36 s | 1.40 s | 1.16 / 1.62 ms | 1.18 / 1.64 ms | 100.00% both |
| 5,000 | 100 | 7.00 s | 7.15 s | 2.52 / 3.96 ms | 2.61 / 4.04 ms | 99.00% both |
| 10,000 | 100 | 14.36 s | 15.39 s | 3.36 / 5.19 ms | 3.02 / 4.25 ms | 100.00% both |
| 19,829 | 100 | 29.66 s | 30.23 s | 4.22 / 6.77 ms | 4.26 / 6.47 ms | 99.00% both |

This in-process benchmark does not measure resident-service startup, memory
consumption, or transport overhead. The v0.4.0 tag baseline (`b46291b2`) was
built offline because it has no committed lockfile, so its dependency versions
may differ from v0.5.0. Treat small timing changes as diagnostic. Raw reports:
[v0.4.0 baseline](https://github.com/RooAGI/Membrick/blob/main/comparison/results/release-v0.4.0-corpus-scale-c8fef3d7.csv),
[v0.5.0 candidate](https://github.com/RooAGI/Membrick/blob/main/comparison/results/release-v0.5.0-corpus-scale-c8fef3d7.csv).

## Native event retrieval run (2026-10-12)

The latest full 500-question run uses the heuristic backend, structured claim
retrieval, and beKIND disabled. The native extractor recognizes additional
first-person event descriptions; structured search uses the question's
reference date and returns events in chronological order when requested. This
uses the normal result depth, so its scores are separate from the session-depth
overfetch diagnostic above.

| Configuration | Fractional Recall@5 / @10 / @20 | Any-hit Recall@5 / @10 / @20 | MRR | NDCG@10 | Mean search time |
|---|---|---|---:|---:|---:|
| Structured retrieval off | 89.00% / 94.40% / 95.69% | 96.0% / 98.0% / 98.6% | 88.76% | 87.53% | 2.80 ms |
| Structured native events | **89.38% / 94.70% / 95.93%** | **96.4% / 98.4% / 99.0%** | 88.52% | 87.50% | 3.50 ms |

The sports-events question now retrieves all three answer sessions in the top
three; before the native event changes, none appeared in the top 20. Across the
full set, Any-hit Recall@5 increased 0.4 percentage points. MRR changed from
88.76% to 88.52%, and mean search time increased from 2.80 to 3.50 ms. The
result is a working-tree diagnostic, not a tagged-release benchmark.

Reports: [structured native events](https://github.com/RooAGI/Membrick/blob/main/scripts/longmemeval-500-claim-contract-2026-10-12.json)
and [structured retrieval off](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-500-nobekind-2026-10-12.json).

### Previous standalone rerun (2026-10-04)

The 500-question release run used the heuristic NER backend, single-index
mode, no embeddings, and cutoffs 1, 3, 5, 10, and 20. It is retained here as
the previous measurement for comparison.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| MRR | 87.0% |
| NDCG@10 | 85.11% |
| Search latency, mean | 2.68 ms |

The latency measures the `MemoryService::search` call and excludes indexing the
question's haystack. The older published 13.0 ms latency uses a different
timing scope, so it should not be compared directly.

### Current Tantivy 0.25.0 rerun (2026-10-05)

This is the latest 500-question single-index run after the staged-write work.
It used a release profile, heuristic backend, and no embeddings. The source
tree contained local changes, so this is a working-tree result, not a tagged
release measurement. Retrieval quality matches the previous run at the
displayed precision; mean search latency is 2.58 ms.

| Metric | Result |
|---|---:|
| Fractional Recall@5 / @10 / @20 | 86.37% / 91.89% / 92.93% |
| Any-hit Recall@5 / @10 / @20 | 94.2% / 96.8% / 97.6% |
| MRR | 86.97% |
| NDCG@10 | 85.11% |
| Mean search latency | 2.58 ms |

Report: [latest 500-question run](https://github.com/RooAGI/Membrick/blob/main/comparison/results/retrieval-longmemeval-current-2026-10-05-write-publication-k20.json.gz).

## Segmented-index comparison (500 questions)

The latest segmented benchmark evaluated all 500 eligible LongMemEval-S
questions using Tantivy 0.25.0 and the experimental segmented feature. These
variants share the same dataset and harness, including a single-index control.
The source tree contained local changes, so this is a working-tree result.

| Mode | Any-hit Recall@5 | Any-hit Recall@10 | Fractional Recall@5 | MRR | Average latency |
|---|---:|---:|---:|---:|---:|
| Segmented (fixed top-5) | 94.20% | 94.20% | 86.77% | 0.880 | 2.49 ms |
| Segmented (enriched + reranked) | 94.20% | 94.20% | 86.77% | 0.881 | 3.85 ms |
| Adaptive (enriched + reranked) | 94.00% | 96.00% | 86.30% | 0.884 | 7.26 ms |
| Intent baseline | 94.20% | 96.20% | 85.54% | 0.863 | 0.71 ms |
| Fused adaptive + global | 95.40% | 97.20% | 87.11% | 0.880 | 61.84 ms |
| Fused temporal + global | **96.00%** | **97.80%** | **88.51%** | **0.885** | 28.05 ms |
| Single-index control | 94.40% | 96.60% | 86.41% | 0.872 | 13.45 ms |

The run used the heuristic backend, fixed top-5 routing, adaptive expansion to
12 segments, and the `coverage-local` router. Latency is per-query benchmark
time excluding haystack indexing. These direct-index timings are not comparable
to the standalone MemoryService timing in the 500-question headline above.
Fused temporal + global had the strongest recall and MRR among these listed
variants, with higher latency; the intent baseline was the fastest listed.

Full report: [`segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz).


### Historical Tantivy 0.26.2 segmented rerun (2026-10-05)

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

Full report: [segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz](https://github.com/RooAGI/Membrick/blob/main/comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.26.2.json.gz).

### Reasoning-rules rerun (2026-10-12)

The segmented benchmark was rerun after moving reasoning cues into the plain
text `data/reasoning_rules.json` configuration. It scored all 500 questions
with Tantivy 0.25.0, heuristic NER, fixed top-5 routing, adaptive expansion to
12 segments, and the `coverage-local` router. This is a working-tree result.

| Mode | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Fixed top-5 segments | 94.2% | 94.2% | 86.84% | 0.887 | 4.85 ms |
| Fixed top-5, enriched + reranked | 94.2% | 94.2% | 86.84% | 0.882 | 6.30 ms |
| Adaptive 5→12, enriched + reranked | 94.8% | 96.4% | 87.06% | 0.888 | 9.66 ms |
| Intent | 94.8% | 96.4% | 86.40% | 0.869 | 1.31 ms |
| Fused temporal + global | **96.2%** | **98.4%** | **89.73%** | **0.895** | 41.59 ms |
| Single-index control | 95.8% | 98.0% | 88.94% | 0.885 | 13.49 ms |

Fused temporal + global had the highest listed recall and MRR, with the highest
latency. The single-index control had slightly lower retrieval scores and lower
latency than that fused mode. The direct-index timings are not comparable to
the standalone MemoryService timings above.

Full report: [`segment-longmemeval-500-2026-10-12-reasoning-rules.json.gz`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/segment-longmemeval-500-2026-10-12-reasoning-rules.json.gz).

### Default LocalDistinctiveness router comparison (2026-10-12)

The same 500-question comparison was run with the server's default
`LocalDistinctiveness` router. Relative to `coverage-local`, the fixed top-5
route was faster but had lower recall and MRR. The global control is unchanged
because it queries all segments.

| Mode | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Fixed top-5 segments | 92.0% | 92.0% | 82.38% | 0.874 | 3.53 ms |
| Adaptive 5→12, enriched + reranked | 94.0% | 95.2% | 85.35% | 0.872 | 10.26 ms |
| Fused temporal + global | 95.0% | 98.2% | 87.25% | 0.887 | 42.50 ms |

Full report: [`segment-longmemeval-500-2026-10-12-reasoning-rules-local-distinctiveness.json.gz`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/segment-longmemeval-500-2026-10-12-reasoning-rules-local-distinctiveness.json.gz).

The earlier 133-query multi-session comparison is retained as a historical
slice in [`segment-multisession-v0.2.0.json`](https://github.com/RooAGI/Membrick/blob/main/comparison/results/segment-multisession-v0.2.0.json).
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
  --out benchmark/data/membrick_longmemeval_scoped_results.json

cargo run --release --bin corpus_scale_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --sizes 1000,5000,10000,25000 --queries 100
```

For the full recorded outputs and comparison methodology, see the repository
[`comparison/`](https://github.com/RooAGI/Membrick/tree/main/comparison) folder.

## Tantivy upgrade throughput verification

The paired 0.25.0 versus 0.26.2 check uses 23,366 records, five sessions, five
repetitions and 1,000 requests per cell. Single-index C=10 medians are
2,246.69 → 2,769.53 req/s; routed medians are 888.86 → 901.29 req/s.
See [Tantivy upgrade verification](releases/tantivy-0.26.2-verification.md)
for the full protocol, compatibility qualifications and raw artifact locations.
