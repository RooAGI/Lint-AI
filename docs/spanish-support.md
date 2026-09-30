# Spanish Language Support

Lint-AI supports Spanish memories end-to-end: detection, tokenization,
indexing, NER, question understanding, temporal expressions, stopwords,
numbers, and aggregation. This document describes the implementation,
the design decisions, and the known limitations.

All language-selection logic lives in Rust. The Python spaCy scripts
(`scripts/spacy_ner.py`, `scripts/spacy_relations.py`) only receive a
model name via their existing `ALLOWED_MODELS` allowlists; the model is
chosen in Rust (`src/lang.rs::default_spacy_model_for_lang`).

## Language detection (`src/lang.rs`)

`Lang::Es` is detected from:

- **Distinctive characters** (strong signal): `ñ Ñ á é í ó ú ü Á É Í Ó Ú Ü ¿ ¡`.
  Two or more distinctive characters, or one combined with function-word
  hits, classifies as Spanish.
- **Function-word hit rate** (for unaccented informal text): a list of
  common Spanish function words (`el`, `la`, `de`, `que`, `en`, `y`, `por`,
  `para`, `con`, …). Two or more hits classifies as Spanish.
- Ambiguous text defaults to English. A single Spanish-looking word
  (`hola`, `el`) is not enough — this avoids misclassifying English text
  containing loanwords.

`--lang es` forces Spanish; `--lang auto` (default) detects per text.

## Tokenization and indexing

**Script agreement (no lossy deunicode).** Like Hangul, which is never
romanized, accented Latin is indexed in its original script. The tantivy
`"default"` tokenizer (`src/index/cjk_tokenizer.rs`) lowercases but
never folds: `niño` indexes as `niño`, not `nino`. Queries are tokenized
the same way, so index and query meet in the same script.

This is deliberately lossy-free: ASCII-folding would conflate distinct
words (`sí` = yes vs `si` = if) irreversibly.

The boosted `entities` and `important_terms` fields use
`normalize_for_index` (which deunicodes + stems) — a *separate* field with
its own consistent normalization. Query terms for boosted fields go
through the same normalization, so index and query agree *within* each
field: content agrees on the original script, boosted fields agree on the
normalized form.

**Limitation:** accent-insensitive search is not supported. Query `nino`
will not match document `niño` (and vice versa). Users must match accents.

The unstemmed Rust tokenizer (`src/tokenizer.rs`) also preserves accents
(`niño`, `dónde`, `está` stay intact pre-index).

## spaCy NER (`es_core_news_sm`)

- `default_spacy_model_for_lang(Lang::Es)` returns `"es_core_news_sm"`.
- Tier-1 document batches are grouped by `PipelineOptions::spacy_model_for_text`,
  so Spanish documents use the Spanish model.
- `spacy_model_for_request()` (`src/memory_api.rs`) picks the model: an
  explicit `--lang` pins it, while `auto`/omitted follows the document
  turns' detected script (`extractor_model_for_turns`) — Spanish turns get
  `es_core_news_sm`.
- Install with: `python3 -m spacy download es_core_news_sm`
  (use `PIP_BREAK_SYSTEM_PACKAGES=1` on Debian-based systems if needed).

## Interrogatives and question focus

Spanish interrogatives are detected on the **raw (accented) token** —
never the stemmed form, where `qué` (what) is indistinguishable from the
relative pronoun `que` (that). Covered forms:

| Interrogative | QueryKind |
|---|---|
| qué | What |
| quién / quiénes | Who |
| dónde | Where |
| cuándo | When |
| cuánto / cuánta | HowMuch |
| cuántos / cuántas | HowMany |
| cuál / cuáles | Which |
| por qué | Why |
| cómo | (focus word; no `How` QueryKind variant exists) |

Leading `¿`/`¡` and punctuation are stripped before classification.
Unaccented `como` (like/as) is deliberately excluded from wh-detection;
only accented `cómo` counts. Unaccented `que` (relative pronoun) is never
a question word.

## Temporal expressions (`src/temporal.rs`)

A Rust pre-layer resolves Spanish temporal expressions before the English
path, returning the same `TemporalTarget` shape:

- Relative days: `hoy`, `ayer`, `anteayer` / `antes de ayer`, `mañana`,
  `pasado mañana`. `por la mañana` (in the morning) is *not* tomorrow.
- Weeks/months/years: `semana pasada`, `esta semana`, `próxima semana`,
  `mes pasado` / `próximo`, `año pasado` / `próximo`.
- Offsets: `hace N días/semanas/meses/años`, `en N días/semanas/meses/años`
  — with digits *and* number words (`hace dos semanas` works via Spanish
  text2num normalization).
- Weekdays: `lunes` … `domingo`, plus `próximo/pasado <weekday>`.
- Absolute: `29 de septiembre de 2026`, `29/09/2026`, `29-09-2026`.
  ISO `2026-09-29` (year-first) is excluded from the day-first pattern.

Spanish weekday and month names are also recognized at index time.

## Stopwords

Four lists, all with Spanish coverage:

1. English unstemmed (`src/tokenizer.rs::unstemmed_stopwords`) — unchanged.
2. English stemmed (`src/tokenizer.rs::stemmed_stopwords`) — unchanged.
3. Spanish (`src/tokenizer.rs::spanish_stopwords`) — raw accented forms
   (`está`, `dónde`, `qué`) **plus** deunicoded + Porter-stemmed forms
   (`esta`, `dond`, `que`). The set is verified as a fixed point under
   `normalize_for_index` by test
   (`spanish_stopwords_cover_normalized_forms`): every stopword, once
   normalized, lands back in the set. Note the Porter stemmer is aggressive
   on Spanish (`dónde` → `dond`, `puede` → `pued` → `pu`) — the normalized
   forms are empirically derived, not hand-written.
4. Tier-1 ranker stopwords (`src/tier1.rs::default_stopwords_for_lang`) —
   Spanish raw accented forms for YAKE/RAKE/TextRank, resolved from
   document content language.

`is_stopword_for_lang` gates per language: English text never consults
the Spanish list, so colliding surface forms (`no`, `son`, `era`, `tan`,
`la`, `el`) stay live in English.

## Numbers and aggregation (`src/aggregation.rs`)

- `text2num::Language::spanish()` is selected when Spanish is detected
  (the `text2num` crate natively supports Spanish — no table needed).
- Count triggers: `cuántos`, `cuántas`, `número de`.
- Sum triggers: `cuánto`, `cuánta`, `total`, `en total`, `suma`.
- Guard: `en cuanto a` (regarding) does not trigger aggregation.
- Spanish number words and unit forms are recognized in routing.

## Query expansion — limitation

**Query expansion is a no-op for Spanish.** The expansion store is
English WordNet/ConceptNet only. Applying it to Spanish terms would cause
harmful cross-language matches (Spanish `pan` = bread vs English `pan` =
cooking vessel). Spanish lexical resources (synonyms, hypernyms) are out
of scope. `expand_query_terms` returns the original terms unexpanded for
`Lang::Es`. Spanish question words, pronouns, and generic verbs are still
excluded from focus-term expansion via `is_expandable_concept`.

## Testing

- Unit tests: `cargo test --lib spanish_` (16 tests: detection, accents,
  stopword fixed-point, focus, temporal, aggregation, expansion no-op).
- Tokenizer tests: `cargo test --lib cjk_tokenizer` (accent preservation,
  ASCII unchanged, long-token drop, CJK segmentation).
- Smoke test: `cargo test --test spanish_smoke` — add Spanish memories,
  search in Spanish, verify retrieval (explicit `--lang es` and
  `--lang auto`).
- English regression: `cargo test --test english_smoke` — add/search in
  English, verify no behavior change.

## Merge notes (zh/ko parallel tracks)

- `src/lang.rs`, `src/index/cjk_tokenizer.rs`, `--lang` plumbing, and spaCy
  allowlists are owned by the zh (foundation) track. The `Es` variant,
  Spanish detection signals, and `es_core_news_sm` mapping in this branch
  are additive and merge into the canonical shared files at integration.
- The unified `src/index/cjk_tokenizer.rs` overrides tantivy's `"default"`
  tokenizer (Han → character bigrams, Hangul → eojeol + stem, Latin →
  lowercased with accents preserved). The old `latin_tokenizer` module was
  folded into it at rebase time per its MERGE NOTE.
