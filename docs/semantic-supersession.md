# Semantic supersession

Lint-AI treats supersession as a core semantic relationship. Source adapters
and provider integrations only normalize content into `SourceDocument` records;
the core decides how records relate over time.

```text
source adapters and agent integrations
                 │
                 ▼
          SourceDocument
                 │
                 ▼
           IndexStore::refresh
          ├─ claims and provenance
          ├─ temporal facts
          └─ semantic relations
                 │
                 ▼
              query
          ├─ relevance ranking
          ├─ temporal ranking
          └─ current-state policy
```

## What the core records

Each extracted claim retains its source document, evidence text, scope,
confidence, and effective timestamp when one is available. Relationships are
directional and may be:

- `supersedes`: newer evidence replaces an older claim for normal queries;
- `conflicts_with`: two claims disagree but the core cannot safely choose one;
- `confirms`: later evidence repeats the same claim and value.

Original documents and claims are never deleted. A historical query can still
retrieve superseded evidence, together with its replacement and relationship
evidence.

## Detection policy

The default detector is local and conservative. It compares claims within a
project semantic scope using canonical subjects and predicates.

- Explicit replacement metadata is strongest evidence.
- A direct correction cue can establish replacement across source types.
- A newer changed value with the same canonical claim and source kind can be
  superseded automatically.
- Automatic replacement suppresses a document only when that document contains
  only the superseded claim. If the document also carries other guidance, it
  remains searchable and is marked as conflicted until claim- or chunk-level
  filtering can isolate the replaced evidence.
- A timestamp-only disagreement between a document and an agent memory is a
  conflict, not an automatic replacement.
- Claims without enough evidence remain visible; they are never hidden merely
  because they are older.

Normal searches suppress only high-confidence superseded evidence. Conflicts
remain visible and carry status, confidence, and evidence in search results.
The core searches project-wide, while `semantic_scope` and `memory_user_id`
prevent unrelated users or scenarios from affecting one another.

For normal queries, the core computes the eligible current document IDs before
ranking and intersects them with user, project, temporal, and segment filters.
Superseded documents therefore never consume ranked candidate slots. Search
returns up to `top_k` matching current documents; historical queries rank the
full eligible corpus and annotate superseded results as historical.

## Integration contract

An integration should provide content and provenance such as `source`,
`timestamp`, `group_id`, `author_agent`, and optional relationship hints. It
should not implement its own supersession filtering. Markdown frontmatter and
the memory API's optional `supersedes_id` remain supported as explicit evidence
for the same core relation engine.

The public Rust API exposes `SemanticClaim`, `SemanticRelation`,
`SemanticRelationStore`, and `SupersessionOptions`. `IndexStore` rebuilds this
state alongside its lexical, semantic, chunk-lifecycle, and temporal state on
refresh.

Enabled supersession configurations are validated during `IndexStore`
construction and refresh. Fallible callers can use
`SemanticRelationStore::try_from_documents` directly and receive the same
validation error. Setting `enabled: false` intentionally returns an empty
relation store without validating thresholds that will not be used.
