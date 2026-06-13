# Filtered Search Design

This document describes the filtered search feature added in v0.1.8: how
`filters` work, where they live in the data model, the query API, and the
design rationale.

## Problem

A single `IndexStore` can hold documents from many logical scopes — different
runs, workspaces, graphs, or artifact types. Without a way to restrict a query
to a subset of documents, callers must post-filter results or maintain separate
`IndexStore` instances per scope.

Separate stores work at small scale but become expensive at large scale: each
store carries its own Tantivy index, in-memory `MemoryIndex`, and rebuild cost.
Shared stores with post-filtering waste BM25 candidate budget on documents the
caller will discard anyway.

Filtered search solves this by pushing scope constraints into the query layer.

## Data Model

### `SourceDocument.filters`

`filters` is an opaque `BTreeMap<String, String>` on `SourceDocument`:

```rust
pub struct SourceDocument {
    pub doc_id: String,
    pub source: String,
    pub content: String,
    // ... existing fields ...
    pub filters: BTreeMap<String, String>,
}
```

Callers populate it with whatever key-value scoping they need:

```rust
SourceDocument {
    doc_id: "post::abc::item::0".into(),
    content: "...",
    filters: BTreeMap::from([
        ("run_id".into(),       "3b0217db-f83c-...".into()),
        ("workspace_id".into(), "ws-456".into()),
        ("artifact_id".into(),  "cb629d14-...".into()),
    ]),
    ..
}
```

Keys and values are arbitrary strings. The library treats them as opaque; it
does not interpret or validate them.

`filters` defaults to an empty map (`#[serde(default)]`) so existing
serialized records without the field remain valid.

### Propagation

`filters` flows through every layer without transformation:

```
SourceDocument.filters
  → assemble_doc_record    → DocRecord.filters
  → PersistedDocRecord     → serialized to disk
  → From<PersistedDocRecord> for DocRecord   → restored on load
  → source_document_from_record → SourceDocument.filters (round-trip)
```

Nothing in the pipeline modifies, normalizes, or indexes `filters` as a
searchable field. It is metadata used only at query time.

## Query Semantics

All filter predicates use **exact-match AND** semantics:

- every key-value pair in the supplied filter map must match the document's
  `filters` map
- a document that is missing a required key does not match
- an empty filter map means no restriction — all documents are candidates

This is intentionally simple. Range queries, OR conditions, and prefix matches
are not supported.

## Query API

### Single-term: `IndexStore::query_filtered`

```rust
pub fn query_filtered(
    &mut self,
    query: &str,
    top_k: usize,
    filters: &BTreeMap<String, String>,
) -> Result<Vec<SearchResult>>
```

1. Calls `self.refresh()` to ensure the snapshot is current.
2. Runs `self.lexical.search(query, candidate_k)` — BM25 via Tantivy — to get
   a set of lexical hit scores keyed by `doc_id`.
3. Passes lexical hits and `filters` to `MemoryIndex::query_with_filters_and_lexical`.
4. `MemoryIndex` builds the allowed set, then scores candidates that are in
   both the lexical hits and the allowed set.

### Multi-term: `IndexStore::query_filtered_multi`

```rust
pub fn query_filtered_multi(
    &mut self,
    queries: &[&str],
    top_k: usize,
    filters: &BTreeMap<String, String>,
) -> Result<Vec<Vec<SearchResult>>>
```

Returns one `Vec<SearchResult>` per input query, in the same order.

Key difference from calling `query_filtered` N times: the allowed-doc set is
built **once** from `filters` and reused for every query term. BM25 is still
called once per term (Tantivy does not batch lexical searches), but the O(n_docs)
filter scan is paid exactly once.

At small corpus sizes the difference is negligible. At thousands of documents
with many query terms per call, `query_filtered_multi` avoids O(n_docs × N)
rescanning.

### `MemoryIndex` layer

Two methods on `MemoryIndex` back the public API:

```rust
// single-term, no pre-computed lexical hits
pub fn query_with_filters(
    &self,
    query: &str,
    top_k: usize,
    filters: &BTreeMap<String, String>,
) -> Vec<SearchResult>

// single-term, with pre-computed BM25 hits
pub fn query_with_filters_and_lexical(
    &self,
    query: &str,
    top_k: usize,
    filters: &BTreeMap<String, String>,
    lexical_hits: Option<&HashMap<String, f32>>,
) -> Vec<SearchResult>

// multi-term, allowed set built once
pub fn query_with_filters_multi(
    &self,
    queries: &[&str],
    top_k: usize,
    filters: &BTreeMap<String, String>,
    lexical_hits_per_query: &[HashMap<String, f32>],
) -> Vec<Vec<SearchResult>>
```

`IndexStore` is the recommended entry point. The `MemoryIndex` methods are
public for callers that manage their own BM25 pipeline.

## Implementation Notes

### Why BM25 before filter

BM25 runs first and produces a candidate set. The filter then restricts that
set. This order avoids scoring documents that would be excluded anyway — Tantivy
already narrows the field to lexically relevant documents before the filter scan.

An alternative — filter first, then BM25 — would require a Tantivy reader that
restricts to a doc-id allowlist. Tantivy supports this via `BitSetDocSet` but it
requires building a Tantivy `DocId` bitset, which needs a mapping from our
string `doc_id` to internal Tantivy row ids. That mapping is not currently
maintained. The current approach (BM25 → filter on string `doc_id`) is simpler
and correct for corpora up to tens of thousands of documents.

### Why `BTreeMap` not `HashMap`

`BTreeMap` gives deterministic iteration order, which matters for consistent
behaviour in tests and for stable serialization. Filter maps are small (typically
2–5 keys), so the O(log n) lookup cost versus `HashMap` is irrelevant.

### No filter indexing

Filters are not stored in Tantivy and are not indexed as facets in `MemoryIndex`.
The allowed-doc set is built by a linear scan over `MemoryIndex.docs` at query
time. This is intentional: the scan is O(n_docs) regardless of how selective the
filter is, but it is cache-friendly and requires no additional index structure.
If filter selectivity becomes a bottleneck, a secondary facet index keyed on
filter fields would be the natural next step.

## Usage Example

```rust
use lint_ai::{IndexStore, PipelineOptions, SourceDocument};
use std::collections::BTreeMap;

let mut store = IndexStore::new(PipelineOptions::default());

// Index documents tagged with run and artifact scope
for (i, text) in texts.iter().enumerate() {
    store.upsert(SourceDocument {
        doc_id: format!("doc-{i}"),
        content: text.clone(),
        filters: BTreeMap::from([
            ("run_id".into(), "run-abc".into()),
            ("artifact_id".into(), "artifact-xyz".into()),
        ]),
        ..Default::default()
    });
}

// Single-term filtered query
let scope = BTreeMap::from([("run_id".into(), "run-abc".into())]);
let results = store.query_filtered("NVDA earnings", 5, &scope)?;

// Multi-term filtered query — filter scan happens once
let terms = &["NVDA", "TSLA", "AMD"];
let per_term = store.query_filtered_multi(terms, 5, &scope)?;
// per_term[0] = results for "NVDA"
// per_term[1] = results for "TSLA"
// per_term[2] = results for "AMD"
```

## Design Constraints

- **No deep merge in `filters`**: values are plain strings, not nested objects.
  Compound keys (e.g. `"workspace_id:graph_id"`) are the caller's responsibility.
- **No negation**: there is no way to express "exclude documents where key=value".
- **No partial match**: filter values are compared with `==`, not `starts_with`
  or regex.
- **Filters are not returned in `SearchResult`**: callers already know what
  filters they applied. Echoing them in results adds noise.

These constraints keep the implementation simple and the performance predictable.
If richer filter semantics are needed in the future, the right direction is a
dedicated facet index rather than extending the current linear-scan model.
