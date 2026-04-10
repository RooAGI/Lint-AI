# Tier 1 Hybrid Indexing and Retrieval Design

This document defines how to use Tier 1 outputs for in-memory indexing and search.

## 1. Canonical Tier 1 Record
Store one Tier 1 record per `doc_id`.

Required now:
- `probable_topic`
- `key_entities` (text, label, score)
- `important_terms` (term, score, ranker source)

Planned fields:
- `doc_type_guess`
- `embedding`
- `top_claims`

Provenance fields (required):
- `source`
- `timestamp`
- `ner_provider`
- `term_ranker`
- `index_version`

## 2. In-Memory Indexes
Build multiple in-memory indexes from Tier 1 records.

### 2.1 Lexical Inverted Index
- key: term
- value: postings list of docs with term-level weights
- source fields:
  - `important_terms`
  - headings (optional boost)

### 2.2 Entity Index
- key: normalized entity string
- value: postings list (`doc_id`, entity score, label)
- source field:
  - `key_entities`

### 2.3 Facet Index
- topic facet: `probable_topic -> doc_ids`
- doc-type facet: `doc_type_guess -> doc_ids` (when available)

### 2.4 Vector Index
- in-memory embedding table: `doc_id -> embedding`
- used for vector similarity retrieval when embeddings are available

## 3. Search / Retrieval Strategy
Use hybrid retrieval, then re-rank.

### 3.1 Candidate Retrieval
- lexical retrieval (BM25/keyword style on content + important terms)
- entity retrieval (entity overlap boost)
- vector retrieval (when embeddings exist)

### 3.2 Candidate Merge
- union candidate sets by `doc_id`
- keep per-source scores for explainability

### 3.3 Re-Ranking Features
Re-rank with Tier 1 signals:
- entity score overlap
- term salience overlap
- same `doc_type_guess` (when available)
- same/related `probable_topic`
- optional recency boost via `timestamp`

### 3.4 Output
Return ranked docs with transparent reasons:
- matched entities
- matched terms
- facet matches
- per-component score contributions

## 4. Query Flow
1. Parse query into entities and terms.
2. Retrieve top N from lexical and entity indexes.
3. Merge with vector top N (if available).
4. Re-rank using Tier 1 features.
5. Return results with `why_matched` evidence.

## 5. Claim-Hint Extension (Later)
When `top_claims` is available, use claim overlap/conflict hints to:
- prioritize contradiction checks
- generate comparison candidate sets
- improve alignment diagnostics

## 6. Implementation Notes
- Keep indexing deterministic and reproducible via provenance/version fields.
- Rebuild index whenever ranker/provider/version changes.
- Keep index structures plain memory objects for algorithm-first iteration.
