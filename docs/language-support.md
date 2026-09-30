# Language support

Lint-AI retrieves memories lexically (Tantivy BM25 plus heuristic
re-ranking) — there is no embedding model, so every language needs its own
tokenization, normalization, and linguistic rules. This page documents what
each supported language gets and, just as importantly, what it does not.

Supported languages: English (`en`), Chinese (`zh`), Korean (`ko`),
Spanish (`es`).

## Selecting a language

`--lang auto|en|zh|ko|es` (default `auto`) is available on the CLI, in
`PipelineOptions`, and on the HTTP search API (`SearchRequest.lang`).
`auto` detects per text from script statistics: a Han majority selects
Chinese, a Hangul majority Korean, otherwise English. Ambiguous or
script-free text falls back to English.

The language drives per-language defaults — most visibly the spaCy model.
Unless `--spacy-model` is passed explicitly, the model is chosen in Rust
(`src/lang.rs`):

| Language | Default spaCy model |
|----------|---------------------|
| `zh`     | `zh_core_web_sm`    |
| `ko`     | `ko_core_news_sm`   |
| `es`     | `es_core_news_sm`   |
| `en` / default | `en_core_web_sm` |

The Python side (`scripts/spacy_ner.py`, `scripts/spacy_relations.py`)
only allowlists these model names; all language *selection* logic lives in
Rust. Install the model you need before use, e.g.
`python -m spacy download zh_core_web_sm` (the Chinese model also needs
`spacy-pkuseg`).

The language is part of the query-cache key, so `zh` and `en` queries never
share cached results.

## Chinese (`zh`)

Chinese has no word boundaries in running text, so the pipeline treats it
at the character level throughout.

- **Tokenization** (`src/tokenizer.rs`): Han runs emit overlapping
  character bigrams plus lone single characters (`清华大学` →
  `清华 / 华大 / 大学`). Mixed Han/Latin text is segmented by script so
  Latin keeps its exact historical behavior.
- **Index/query parity** (`src/index/cjk_tokenizer.rs`): the same bigram
  segmentation is registered as Tantivy’s `default` tokenizer on every
  index builder, so indexed postings and query terms agree.
- **No transliteration**: Chinese text stays in Han script end to end.
  `normalize_for_index` preserves Han (and Hangul) instead of converting
  to Pinyin/romanization, and no English stemmer is applied to CJK tokens.
- **Question focus** (`src/question_focus.rs`): Chinese interrogatives
  (`谁 / 什么 / 哪个 / 哪里 / 何时 / 为什么 / 怎么 / 如何 / 多少 / 几个`
  …) are detected by substring scan and excluded from focus terms, so a
  question like `我毕业于哪所大学？` yields a question word (`哪`) and
  content focus instead of an empty result.
- **Temporal expressions** (`src/temporal.rs`): relative words (`昨天 /
  明天 / 上周 / 下周 / 去年 / 今年 …`), explicit dates (`2026年9月29日`,
  including Chinese-numeral forms like `二〇二六年九月二十九日`), numeric
  offsets (`三天前`, `两周后`), and weekday mentions (`上星期一`,
  `下周三`) resolve against the anchor date before the English patterns
  run.
- **Stopwords**: Chinese function words are filtered in the shared
  tokenizer, the tier-1 term rankers, focus classification, and query
  expansion gating (`的 / 了 / 在 / 是 / 我们 / 因为 / 可以` …).
- **Numbers and aggregation** (`src/aggregation.rs`,
  `src/lang.rs`): Chinese numerals (`二十五`, `三千五百万`) normalize to
  digits — single-character numerals only when followed by a measure word,
  so idioms like `一起` are untouched. Count/sum intent covers `多少 /
  几个 / 几次 / 总共 / 一共 / 合计`, and advice-seeking (`应该买多少`)
  is excluded from aggregation. Numeric evidence extraction treats Han
  characters as token boundaries (`买了3本书` → `3`).
- **Chunking and sentences**: token estimates count CJK characters
  instead of whitespace words, and sentence splitting recognizes
  `。！？` alongside `. ! ?`.

## Explicit limitations

- **Query expansion is English-only.** The embedded lexical store is built
  from English WordNet/ConceptNet subsets, so Chinese (and Korean)
  concepts are never expanded. This is a documented gap, not a silent
  no-op: `expand_query_terms` returns every unexpandable non-English term
  in `ExpandedQuery::unexpanded_non_english_terms`.
- **Query-time behood analysis is English-only.** Index-time spaCy
  (`spacy_ner.py`, `spacy_relations.py`) uses the Rust-selected
  per-language model, but `scripts/behood_query.py` — the query-time
  entity analysis — hardcodes `en_core_web_sm` and its protocol has no
  model parameter. Changing that would require Python-side logic, which
  is out of scope for the Rust-only language-support rule, so Chinese
  queries get no query-time entity evidence (fail-open) until the
  protocol is extended.
- **Chinese relation extraction fails open ([E894]).** `spacy_ner.py`
  works with `zh_core_web_sm` (verified: `清华大学` → `ORG`), but
  `spacy_relations.py` uses `doc.noun_chunks`, which spaCy does not
  implement for Chinese — every Chinese relations call errors and the
  Rust side returns empty evidence. A script-side fallback needs
  Python logic, so this awaits a scope decision; it is recorded in
  `tests/verify_spacy_health.rs` (`spacy_chinese_relations_fail_open_on_e894`)
  rather than left as a silent gap.
- **No cross-lingual retrieval.** A Chinese query matches Chinese
  memories; there is no translation or language-bridging layer.
- **Index rebuild after tokenizer changes.** Registering the CJK
  tokenizer does not retokenize already-persisted Tantivy postings.
  Indexes built before this support need a rebuild to search Chinese
  correctly.

## Korean (`ko`) and Spanish (`es`)

Both languages share the same foundation (`src/lang.rs`,
`src/index/cjk_tokenizer.rs`, `--lang` plumbing). Korean additionally
gets Hangul eojeol tokenization with particle-stripped stems and Korean
stopwords; Spanish currently uses the Latin pipeline with its spaCy model
(`es_core_news_sm`) and language selection. Per-language interrogative,
temporal, and number handling for Korean and Spanish is tracked
separately.
