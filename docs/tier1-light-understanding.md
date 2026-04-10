# Tier 1: Light Understanding

## Goal
For every document, extract a cheap fingerprint.

## Outputs Per Document
- probable topic
- key entities/concepts
- important terms
- document type guess
- summary embedding
- a few top claims

## What This Enables
- cluster similar docs
- find candidates for comparison
- detect outdated terminology
- prioritize review and alignment work

## Working Definition: Key Entity
A key entity is an entity that is both:
1. salient in this document (high-signal position/frequency), and
2. structurally central to claims (acts as subject/object in top claims, not just mentioned once)

Current implementation (`heuristic` provider) uses a scored ranker with:
- mention frequency
- heading presence
- section coverage
- early-position bonus
- candidate types from concepts, title-case spans, and acronyms

### Entity
A named thing in the document, such as:
- person
- organization
- system
- product
- dataset
- API
- component
- project
- document
- metric
- date/versioned artifact

## Cheap + Deterministic Extraction Spec

### 1. Probable Topic
Use title + headings + first paragraph + highest-TF terms.
Output one short phrase.

### 2. Key Entities
Candidate generation:
- title-case spans
- acronym spans
- code identifiers
- link anchors
- frontmatter fields

Scoring:
- add score if in title/headings
- add score if repeated across sections
- add score if near predicates like "is", "uses", "depends", "requires"
- add score if linked/referenced by other docs (corpus signal)
- subtract score if generic/noise term (for example: "system", "doc", "process")

Keep top N (for example 5-12), with score and evidence snippets.

### 3. Important Terms
Use n-grams with high within-doc salience and low corpus ubiquity.
Keep separate from entities.

### 4. Document Type Guess
Rule-based first pass:
- spec
- runbook
- decision
- incident
- status
- report
- tutorial
- reference
- changelog
- task-log
- other

Use heading patterns and lexical cues.

### 5. Summary
2-4 sentence cheap summary (extractive first, abstractive optional).

### 6. Embedding
One document embedding over cleaned text (or title + headings + lead paragraph for cost cap).

### 7. Top Claims
Extract 3-8 claim tuples:
- subject
- predicate
- object
- confidence
- source_span

Use lightweight pattern extraction first; LLM fallback optional.

## Open Policy Decisions
1. Include abstract entities (for example "alignment layer") or only named entities?
2. Merge aliases at Tier 1, or postpone to Tier 2 resolution?
3. Fixed cap per document (`top 8`) or confidence-threshold output?
4. Precision-first (fewer, cleaner entities) or recall-first (broader candidate set)?

## Recommended Initial Defaults
- include both named and abstract entities, but tag `entity_kind`
- postpone aggressive alias merge to Tier 2; keep local alias hints only
- fixed cap + minimum confidence (hybrid)
- precision-first for Tier 1

## Local spaCy Integration (Optional)
Tier 1 key entities can be extracted with a local spaCy subprocess (no service required).

Implementation design:
- `KeyEntityRanker` trait in `src/tier1.rs`
- `HeuristicKeyEntityRanker` and `SpacyKeyEntityRanker` implementations
- engine selects ranker at runtime, enabling plug-and-play ranker swaps

CLI examples:

```bash
./target/debug/lint-ai /path/to/repo --show-tier1-entities
```

```bash
./target/debug/lint-ai /path/to/repo --show-tier1-entities --tier1-ner-provider spacy --spacy-model en_core_web_sm
```

Behavior:
- default provider is `heuristic`
- `spacy` provider calls `scripts/spacy_ner.py` via `python3`
- if spaCy is unavailable, the pipeline falls back to heuristic entities

## Important Terms Rankers
Tier 1 important terms are now pluggable through the `ImportantTermRanker` trait.

CLI examples:

```bash
./target/debug/lint-ai /path/to/repo --show-tier1-terms --tier1-term-ranker yake
```

```bash
./target/debug/lint-ai /path/to/repo --show-tier1-terms --tier1-term-ranker rake
```

Available rankers:
- `yake`: YAKE-style scoring (position, frequency, dispersion)
- `rake`: RAKE-style phrase extraction by stopword boundaries
- `cvalue`: C-value-style multi-word term scoring with nested-phrase penalty
- `textrank`: TextRank-style graph ranking over co-occurring terms
