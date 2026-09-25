# Tier 1 Indexing and Search Design

Tier 1 should act as a retrieval metadata layer on top of raw documents.

## Tier 1 Record for Indexing
Store one Tier 1 record per `doc_id` with:
- `probable_topic`
- `key_entities` (text, label, score)
- `important_terms` (term, score, ranker source)
- `doc_type_guess`
- `embedding` (reserved for future vector workflows; not used by the live query path)
- `top_claims`

## Indexing Strategy
Build multiple indexes from Tier 1 outputs:
- lexical inverted index on `important_terms` and headings
- entity index (`entity -> doc_ids`)
- topic and document-type facets

Keep provenance fields (`source`, timestamps, ranker version) to support reindexing and reproducibility.

## Background key-phrase enrichment

New documents are enriched with grammar-accepted entity key phrases
("Harry Potter conference") after they are written, so the segment
entity channel sees the same entity mentions the benchmark extractor
produces — without slowing down the write path.

How it works:

- `add` writes the document immediately with empty `key_phrases` and
  queues it for enrichment. The write never waits for extraction.
- A single worker thread drains the queue in batches of up to 32 (after a
  500ms linger so rapid writes coalesce into one extractor run) and runs
  the extractor's `key_phrases_only` mode off-thread, bounded by the same
  120-second timeout as the relations path.
- Finished phrases are posted to an inbox; the next write or `refresh`
  backfills them into the documents and marks them dirty. Because the
  record content hash covers `key_phrases`, the refresh rebuilds those
  records and the segment profiles pick the phrases up in the cap-exempt,
  literally-indexed entity channel.
- A document replaced while its batch was in flight is recognized as
  stale (content hash mismatch) and its phrases are dropped; the
  replacement re-queued itself when it was written. Deleting a document
  drops its queued and finished enrichment work.

Fail-open: a missing or slow extractor simply leaves documents with
empty key phrases — today's behavior — and writes and searches are
never blocked or failed by enrichment. Disable it with
`PipelineOptions.key_phrase_enrichment = false`. `update` clears a
document's phrases and re-queues it, since they described the old text.

## Search and Retrieval Strategy
Use hybrid retrieval:
- BM25 or keyword retrieval on content and important terms
- entity-match boost from key-entity overlap
- no vector similarity search in the current live pipeline

Re-rank candidates with Tier 1 signals:
- entity score overlap
- term salience overlap
- same `doc_type_guess`
- same or related `probable_topic`

Use claim hints (when available) to select comparison candidates for contradiction and alignment checks.

## Query Flow
1. Parse query into entities and terms.
2. Retrieve top N from lexical and entity indexes.
3. Re-rank using Tier 1 features.
4. Return documents with transparent match reasons (matched entities and terms).

## Expected Outcomes
- improved clustering quality from entity, term, topic, and claim signals
- faster comparison-candidate generation
- terminology drift detection over time
- prioritization using salience, confidence, recency, and conflict likelihood
- vector retrieval can be added later without changing the Tier 1 record shape
