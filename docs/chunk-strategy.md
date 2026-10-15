# Chunk Strategy Design

This document defines how Membrick chunks documents for Tier 1 indexing and query retrieval.

## Goals
- Keep chunks small enough for LLM context windows.
- Keep chunk IDs stable enough for incremental updates.
- Preserve document structure where possible.

## Built-in behavior

### Heading
Splits content by Markdown headings (`#` to `######`).

Rules:
- Text before first heading becomes a `(document)` chunk.
- Each heading section becomes one chunk.
- If no headings exist, the whole file is one chunk.

Best for:
- Well-structured Markdown docs.
- Section-aware retrieval and explanation.

Membrick uses its built-in chunking defaults during indexing and retrieval. Chunk
size and scan limits are implementation details rather than routine CLI tuning
options. For normal use, index a project with the standard command:

```bash
membrick index /path/to/docs
```

## Notes on Caching
Query cache validity includes the chunking behavior, so cached results are
reused only when the indexing inputs remain compatible.

## Internal Chunk Schema
Each chunk stores:
- `chunk_id`
- `heading`
- `content`
- `start_line`
- `end_line`
- `key_entities`
- `important_terms`

## Chunk ID Generation (Current)

Chunk IDs are deterministic, content-derived, and stable across rebuilds when a
chunk's semantic identity is unchanged.

Current generation logic is implemented via:

- `stable_chunk_id(doc_id, heading, content, start_line, end_line)`
- `src/ids.rs`

Behavior:

1. Normalize heading/content components:
   - lowercase
   - trim
   - collapse whitespace
2. Build hash input parts:
   - `doc_id`
   - normalized `heading`
   - normalized `content`
   - `start_line`
   - `end_line`
3. Compute FNV-1a 64-bit hash
4. Emit `chunk:<hex>`

Pseudo-shape:

```text
chunk_id = "chunk:" + fnv1a64_hex(
  doc_id,
  normalize(heading),
  normalize(content),
  start_line,
  end_line
)
```

Implications:

- Same document content + same chunk span/heading -> same `chunk_id`
- If chunk content changes materially, `chunk_id` changes
- IDs are independent of transient chunk ordering (`doc_id::index` is no longer used)

## Chunk Lifecycle Interaction

Chunk identity and chunk lifecycle are intentionally separate:

- `SectionChunk` stores raw content/span/indexing features
- lifecycle metadata tracks:
  - `version`
  - `is_latest`
  - `supersedes_chunk_id`
  - lineage key and timestamps

Snapshot queries use the current `DocRecord.section_chunks` set (latest chunks),
while lifecycle metadata preserves evolution history.
