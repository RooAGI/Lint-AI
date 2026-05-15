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
