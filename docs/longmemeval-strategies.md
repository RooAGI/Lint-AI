# LongMemEval segmented retrieval strategies

This guide explains the retrieval variants reported by the full 500-question
segmented LongMemEval-S benchmark. It separates **routing** (which segments to
search), **retrieval** (how to search selected segments), and **fusion** (how
to combine ranked lists). The benchmark is an experimental comparison; these
labels do not imply that every variant is a server default.

For commands and the complete score tables, see the
[benchmark overview](benchmark.md#segmented-index-comparison). The latest
per-query data is in
[`segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/segment-longmemeval-500-2026-10-05-tantivy-0.25.0-control.json.gz).

## Routing: choosing segments

Each segment represents a memory session. A router scores segments against the
query, then a retrieval mode decides how many of the ranked segments to search.
The 500-question run used `coverage-local` routing with top-5 selection.

| Router | How it ranks segments |
|---|---|
| `sparse` | Weighted overlap with query terms, entity terms, and topic terms. |
| `kl` | Similarity between the query term distribution and a segment's term distribution, using smoothed KL divergence. |
| `local` | Query-term weights multiplied by their distinctiveness across the corpus. |
| `coverage-local` | Local distinctiveness plus a reward for covering rare and multiple query terms. This was used in the 500-question comparison. |
| Typed-evidence and team variants | Add evidence categories such as people, time, subjects, actions, and objects, or select segments to improve coverage across terms. |

The router can fall back to expanded query terms when literal query terms do
not produce a useful route. This helps with related wording, while focus-term
gating reduces routes driven only by generic question words.

## Retrieval: fixed, adaptive, and enriched

**Single-index control** searches one index over the question's entire
question-scoped haystack. It is the reference for comparing retrieval quality
inside this harness.

**Fixed top-N** searches only the first *N* routed segments. In the reported
fixed mode, *N* is 5. This bounds search work, but relevant evidence outside
those five segments cannot be returned.

**Adaptive top-N** starts with five segments and can expand to twelve when the
selected segments do not cover enough query terms. It chooses additional
segments that contribute missing query terms or have scores close to the
current cutoff. This trades extra search work for broader evidence coverage.

**Segment enrichment** adds query-relevant terms inferred from segment-local
evidence before querying the selected segments. The benchmark also measures a
route-aware reranked form, which adjusts candidates using evidence from their
source segment. Those are separate variants; the table labels which one is
reported.

**Temporal-path enrichment** starts with routed segments, then can add sessions
connected by temporal relations when the query has an active temporal context.
This is intended for questions where nearby or ordered events matter.

## Intent and multi-operand retrieval

The intent path tries to identify an operation such as comparing two dates,
ordering alternatives, aggregating values, filtering events around a date, or
finding a shared relation. It validates extracted operands against entities
present in the corpus, retrieves for each operand, and combines the results.
If the question does not express a supported multi-operand operation or has
fewer than two usable operands, it falls back to adaptive retrieval.

This makes intent retrieval a conditional strategy rather than a universal
rewrite. Its aggregate LongMemEval score therefore includes ordinary fallback
queries as well as any questions handled by the intent path.

## Fusion: combining routed and exhaustive rankings

The fused variants combine a routed ranking with an exhaustive all-segments
ranking using reciprocal rank fusion (RRF). RRF combines positions rather than
raw search scores, so differently scaled scores from separate paths do not
dominate the merge.

| Variant | Ranked lists combined |
|---|---|
| Fused adaptive + global | Adaptive enriched ranking + exhaustive all-segments ranking. |
| Fused temporal + global | Temporal-path enriched ranking + exhaustive all-segments ranking. |

Here, **global** means the all-segments coverage arm in the segmented
comparison. It is distinct from the single-index control row. The exhaustive
arm helps recover evidence that the router did not select, but querying all
segments costs more time.

## What the 500-question results show

The latest run used LongMemEval-S, Tantivy 0.25.0, the heuristic NER backend,
no embeddings, `coverage-local` routing, five initial routed segments, and an
adaptive maximum of twelve. It used the experimental segmented feature and a
working tree with local changes. Latency is the benchmark's per-query
measurement and excludes question haystack indexing.

| Strategy | Any-hit @5 | Any-hit @10 | Fractional @5 | MRR | Avg. latency |
|---|---:|---:|---:|---:|---:|
| Fixed top-5 | 94.2% | 94.2% | 86.77% | 0.880 | 2.49 ms |
| Enriched + reranked | 94.2% | 94.2% | 86.77% | 0.881 | 3.85 ms |
| Adaptive + enrichment + reranking | 94.0% | 96.0% | 86.30% | 0.884 | 7.26 ms |
| Intent with adaptive fallback | 94.2% | 96.2% | 85.54% | 0.863 | 0.71 ms |
| Fused adaptive + global | 95.4% | 97.2% | 87.11% | 0.880 | 61.84 ms |
| Fused temporal + global | **96.0%** | **97.8%** | **88.51%** | **0.885** | 28.05 ms |
| Single-index control | 94.4% | 96.6% | 86.41% | 0.872 | 13.45 ms |

In this run, temporal fusion had the strongest listed recall and MRR. The
intent path was fastest, but its score includes adaptive fallback behavior and
should not be read as the isolated speed of a successful intent decomposition.
The direct-index benchmark timings should not be compared with the separate
MemoryService latency headline, which measures a different query path.
