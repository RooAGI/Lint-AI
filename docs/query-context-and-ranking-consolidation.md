# Query Context and Ranking Consolidation

Status: design proposal

This document describes the proposed consolidation of query-context handling and
session/group ranking across the global and segmented retrieval paths. It does
not describe an implemented change.

## Motivation

The repository currently has two relevant retrieval paths:

```text
global path:
query → MemoryIndex → document candidates → aggregate_group_score

segmented path:
query → segment routing → selected segment indexes →
document results → aggregate_segment_results_by_session
```

Both paths can return results belonging to the same session or group, but they
currently use different aggregation formulas. The global path has count-query
behavior, rank-decayed sibling support, and a score threshold. The segmented
path uses flat sibling support and adds its own term, entity, document-count,
and graph bonuses.

The goal is not to make segment routing and document retrieval identical. The
goal is to make the final session/group aggregation policy explicit and shared.

## Query-context behavior

`TemporalQueryContext` already carries temporal filters, document scoping, and
an optional `QueryRoutingIntent`. The intent values are:

```rust
QueryRoutingIntent::Count
QueryRoutingIntent::Sum
QueryRoutingIntent::Sequence
```

The existing high-level engine path analyzes the query and populates this
field. Low-level convenience APIs currently default the context, so direct
global and segmented calls do not automatically infer intent.

This field is not currently the source of the global aggregator's
`count_query` decision. In `src/index.rs`, `count_query` is independently
derived from lowercase query-string markers such as `how many`, `count`,
`number of`, and `times did i`. Consequently, a direct call such as
`MemoryIndex::query("How many projects...", ...)` still enters the count
aggregation branch even though its `TemporalQueryContext.query_routing_intent`
is `None`.

`QueryRoutingIntent` does affect other global behavior, including the expanded
candidate limit, disabling some follow-up graph boosts for count/sum queries,
and intent-specific evidence processing. The context gap is therefore real,
but it is not what currently controls `aggregate_group_score`'s count branch.
`QueryRoutingIntent::Sequence` currently has no defined role in group
aggregation; the consolidated policy must either give it one explicitly or
state that it has no aggregation effect, rather than leaving it unspecified.

The proposed behavior is:

| Entry point | Context behavior |
| --- | --- |
| `MemoryIndex::query` | Build an automatic default context from the raw query (intent only, no augmentation). |
| `MemoryIndex::query_timed` | Build the same automatic default context. |
| Segmented convenience APIs (`query_top_segment`, `query_top_segments`, `query_all_segments`; not the internal `_with_strategy`/`_with_corpus_stats` variants) | Build the same automatic default context. |
| `query_with_temporal_context` APIs | Preserve caller-provided context and values. |
| High-level engine (`engine.rs`'s `--query`/`--llm-context` handling) | Continue passing its already analyzed, richer context (intent **and** query augmentation) without re-analysis. |
| `MemoryService::search` (memory server `/search`) | Should be treated like the high-level engine, not like a low-level convenience API — see "Application-level entry points" below. |

The automatic helper should initially infer only `query_routing_intent` and
preserve the existing raw query text. Query augmentation and explicit temporal
behavior remain the responsibility of the high-level engine or explicit context
callers. The shared ranking module should then use the context intent directly;
the eventual consolidation should remove the independent global string
heuristic rather than preserve two count-query classifiers.

Conceptually:

```rust
fn automatic_query_context(query: &str) -> TemporalQueryContext<'static> {
    let analysis = analyze_query(query);
    TemporalQueryContext {
        query_routing_intent: analysis.query_routing_intent,
        ..TemporalQueryContext::default()
    }
}
```

Explicit context APIs must not overwrite a caller's intent. This preserves
advanced callers that have already performed analysis or intentionally selected
a context.

## Application-level entry points

Query-context alignment needs to be traced past the library API and into the
two concrete places that actually build a query today: the memory server and
the CLI. They currently diverge, and one of them (the memory server) is the
newest and most externally-exposed entry point in the codebase.

### Production memory server

```text
MemoryService::search
    → IndexStore::query_filtered
    → MemoryIndex::query_with_filters_and_lexical
```

See `src/memory_api.rs:120` and `src/pipeline.rs:957`.

This is the actual `/search` production path (the memory server introduced in
commit `1f2d8d7`, "Add AML-compatible memory server", 2026-08-13). It does not
go through `engine.rs::query_temporal_context` or `analyze_query` at all today
— it has no query-intent awareness and no query augmentation. The underlying
`query_filtered`/`query_with_filters_and_lexical` plumbing itself is older
(introduced 2026-06-13, commit `7240333`, part of the last tagged release,
`v0.1.8`); the gap is that the August server work exposed it over HTTP without
also wiring it into the context/intent story.

### CLI / high-level engine

```text
engine.rs (args.query / args.llm_context handling)
    → analyze_query(query)
    → search_query = analysis.augmented_query
    → query_temporal_context(&analysis)
    → MemoryIndex::query_with_temporal_context(&search_query, top_k, temporal_context)
```

See `src/engine.rs:2409-2447`. This path does more than the "automatic
default context" proposed above for low-level convenience APIs: it also
augments the query text (`analysis.augmented_query`), not just infers intent.

### Proposed alignment

`MemoryService::search` should be aligned with the CLI/engine path, not with
the lighter default meant for low-level convenience callers:

1. `analysis = analyze_query(&request.query)`
2. `search_query = analysis.augmented_query`
3. `temporal_context = query_temporal_context(&analysis)`
4. Build `allowed_doc_ids` from the `memory_user_id` filter (the same shape
   `query_with_filters_and_lexical` already builds internally from its
   `filters` map — `MemoryService` only ever filters on that one key)
5. Set `temporal_context.allowed_doc_ids = Some(&allowed)`
6. Call `index.query_with_temporal_context(&search_query, top_k, temporal_context)`

This resolves the intent gap, the query-augmentation gap, and the
filters-vs-`allowed_doc_ids` duplication for this caller in one change.

### Consequence for `query_with_filters_and_lexical`

`TemporalQueryContext` already has `allowed_doc_ids: Option<&'a HashSet<String>>`
(`src/index.rs:463`), which `query_with_temporal_context` already threads
through to the same underlying scoring call `query_with_filters_and_lexical`
uses. The filter-derived `allowed` set the latter builds today
(`src/index.rs:1548-1557`) maps onto that field directly. Once
`MemoryService::search` is aligned as above, `query_with_filters_and_lexical`
and `query_with_filters_multi` as distinct code paths are not technically
necessary — `filters` → `allowed_doc_ids` and intent are both already covered
by `query_with_temporal_context`.

The one thing this does not fold in for free: `query_with_filters_and_lexical`
accepts a caller-supplied, pre-computed `lexical_hits` map, while
`query_with_temporal_context` → `query_timed_with_context` always computes
lexical hits itself against `MemoryIndex`'s own internal `self.lexical:
Option<LexicalIndex>` (`src/index.rs:329`). The pre-computed-hits parameter
exists to let `IndexStore` (`src/pipeline.rs`) supply hits from its own,
separate tantivy index — `self.lexical: LexicalState` (`src/pipeline.rs:357`)
— which is otherwise unused: its only four call sites are
`query_with_lexical_hits` (for `query_latest`/`query_fresh`) and
`query_with_filters_and_lexical`/`query_with_filters_multi`
(`src/pipeline.rs:895,908,972,996`). Because all four of those callers call
`self.refresh()` first, which rebuilds `MemoryIndex` — and therefore its own
internal lexical index — from scratch whenever the store is dirty
(`src/pipeline.rs:801-833`), `MemoryIndex`'s internal index is already exactly
as fresh as `IndexStore`'s separate one by the time any of them run.

Consequently, folding these callers onto `query_with_temporal_context` makes
`IndexStore`'s separate `LexicalState` — its schema, writer, reader, and the
`upsert_record`/`remove_doc`/`commit_reload` maintenance that runs on every
mutation — removable, not just the four call sites that use it. This is a
larger change than the aggregation-formula consolidation and should be scoped
and reviewed as its own step, not folded silently into it.

## Internal ranking module

Add a small internal module:

```text
src/ranking.rs
```

Register it as a private or crate-visible module from `src/lib.rs`, depending on
which APIs need to call it. The module should contain the shared final
session/group aggregation policy, not segment-routing logic and not Tantivy
retrieval logic.

The module should accept normalized result evidence rather than either
`CandidateState` or the full segment implementation. A conceptual input model
is:

```rust
struct GroupRankingItem {
    group_id: String,
    doc_id: String,
    base_score: f32,
    matched_terms: Vec<String>,
    matched_entities: Vec<String>,
    graph_support: f32,
}
```

The exact representation can use borrowed data or an internal owned adapter;
the important property is that both retrieval paths provide the same semantic
signals.

The shared entry point would be conceptually:

```rust
fn aggregate_groups(
    items: Vec<GroupRankingItem>,
    intent: Option<QueryRoutingIntent>,
) -> Vec<GroupRankingResult>
```

The shared implementation owns:

- grouping by session/group ID;
- sorting items within a group;
- best-result contribution;
- rank-decayed sibling support;
- the sibling support threshold;
- coverage counting and caps;
- count-query adjustments;
- unique-term and unique-entity caps;
- deterministic group ordering;
- assigning the aggregate group score to returned items.

The current formulas should not be copied blindly. Their useful signals should
be reconciled into one documented policy, with constants defined in one place.

## Integration points

### Global retrieval

Replace the direct use of `aggregate_group_score` in `src/index.rs` with an
adapter:

```text
CandidateState + document metadata
    → GroupRankingItem
    → ranking::aggregate_groups(..., query_context.query_routing_intent)
    → existing global evidence adjustments, where applicable
```

Tantivy scoring, candidate generation, graph retrieval, and temporal filtering
remain in the global path.

### Segmented retrieval

Replace `aggregate_segment_results_by_session` in `src/segments.rs` with an
adapter:

```text
segment-local SearchResult values
    → GroupRankingItem
    → ranking::aggregate_groups(..., temporal.query_routing_intent)
    → explicitly retained segment-specific adjustments, where applicable
```

Segment routing remains independent:

```text
segment routing chooses the search scope
shared ranking aggregates results inside that scope
```

The ranking module must not decide which segments are searched.

## Query flow after consolidation

```text
raw query
    ↓
automatic context helper, unless explicit context was supplied
    ↓
query intent + temporal/scoping context
    ↓
global retrieval or segment routing
    ↓
document candidates and evidence
    ↓
shared ranking::aggregate_groups
    ↓
path-specific final presentation adjustments
```

The high-level engine remains compatible with this flow because it already
performs query analysis. It should pass its existing context through rather than
analyze the query a second time. `MemoryService::search` should join the engine
on this same footing (see "Application-level entry points" above) rather than
entering through the automatic-default path meant for low-level convenience
callers.

## Tests required

The implementation should add tests for:

1. `MemoryIndex::query` and `query_timed` automatically infer count, sum, and
   sequence intent where applicable.
2. Segmented convenience APIs infer the same intent.
3. Explicit `query_with_temporal_context` values are preserved.
4. High-level engine queries do not double-apply query analysis or augmentation.
5. Identical synthetic group items produce identical group ordering through the
   global and segmented adapters.
6. Count-query aggregation changes support weighting consistently in both paths.
7. Unique evidence and sibling-support caps are applied consistently.
8. Segment routing still limits the document scope before shared aggregation.
9. Results are mapped back to the correct session/group after aggregation.
10. `MemoryService::search` produces the same `query_routing_intent` and
    augmented query text as the CLI/engine path for the same input query.
11. Removing `IndexStore`'s separate `LexicalState` does not change
    `query_filtered`/`query_filtered_multi`/`query_latest`/`query_fresh`
    results, since `MemoryIndex`'s own internal lexical index is already
    rebuilt fresh on every `refresh()` these callers perform.

The parity tests should compare ordering and selected group IDs, not raw scores
from Tantivy and segment routing. Those scores come from different retrieval
stages and are not expected to be numerically identical.

**Benchmark regression gate.** Structural/ordering parity tests are not
sufficient on their own: a plausible-looking consolidation can silently shift
retrieval quality in either direction depending on which path is unified
toward which. (A tokenization unification tried during this design's review
cost ~5pp of recall@5 on segment routing one way, and ~0.2pp on the global
path the other way, despite passing all structural tests.) Before landing the
ranking-module consolidation, run the existing `haystack_scoped_benchmark`
(global path, LongMemEval-S, 500 scoped queries) and `segment_scoped_benchmark`
(segmented path, multi-session slice) before and after, and gate on no
regression beyond the ~0.3pp noise floor already established for other
dependency-level changes in this codebase.

## Non-goals

This proposal does not:

- replace Tantivy's BM25 implementation;
- make segment IDF numerically match document IDF;
- remove `SegmentRoutingStrategy`;
- force global retrieval and segment routing to use the same candidate-generation
  algorithm;
- apply query augmentation automatically in every low-level API;
- change the public explicit-context APIs' caller-controlled behavior.
