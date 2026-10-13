# Chunk Strategy Design

This document defines how Membrick chunks documents for Tier 1 indexing and query retrieval.

## Goals
- Keep chunks small enough for LLM context windows.
- Keep chunk IDs stable enough for incremental updates.
- Preserve document structure where possible.

## Available Strategies

### 1. `heading` (default)
Splits content by Markdown headings (`#` to `######`).

Rules:
- Text before first heading becomes a `(document)` chunk.
- Each heading section becomes one chunk.
- If no headings exist, the whole file is one chunk.

Best for:
- Well-structured Markdown docs.
- Section-aware retrieval and explanation.

### 2. `line`
Splits by fixed line windows.

Rules:
- Uses `--chunk-lines` as window size.
- Uses `--chunk-overlap` as overlap between windows.
- Produces line-ranged headings like `lines 1-40`.

Best for:
- Stable diffs and incremental updates.
- Documents with weak heading quality.

### 3. `hybrid`
Line-stable chunking + token-aware splitting.

Rules:
- First chunk by line windows (`--chunk-lines`, `--chunk-overlap`).
- Estimate token size for each line chunk.
- If chunk exceeds token budget, split further into smaller parts.
- Controlled by:
  - `--chunk-target-tokens` (preferred size)
  - `--chunk-max-tokens` (hard cap)

Best for:
- LLM context-fit retrieval with stable change tracking.
- Large documents where fixed line windows can be too large.

## CLI Flags
- `--chunk-strategy heading|line|hybrid`
- `--chunk-lines <N>` (default: `40`)
- `--chunk-overlap <N>` (default: `10`)
- `--chunk-target-tokens <N>` (default: `450`)
- `--chunk-max-tokens <N>` (default: `800`)

## Recommended Defaults
For general use:
- `--chunk-strategy hybrid`
- `--chunk-lines 40`
- `--chunk-overlap 10`
- `--chunk-target-tokens 450`
- `--chunk-max-tokens 800`

## Example Commands
Build index with hybrid chunking:

```bash
./target/debug/membrick --index /path/to/docs --chunk-strategy hybrid --chunk-lines 40 --chunk-overlap 10 --chunk-target-tokens 450 --chunk-max-tokens 800
```

Query with same chunk strategy:

```bash
./target/debug/membrick --query "linux install docker compose" /path/to/docs --chunk-strategy hybrid
```

## Notes on Caching
Query cache validity includes chunk settings. Changing strategy or chunk parameters triggers rebuild.

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
