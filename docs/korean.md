# Korean Language Support

Lint-AI supports Korean (한국어) content across the full retrieval pipeline: indexing, search, NER, question understanding, temporal parsing, and aggregation. All language logic is implemented in Rust; the Python spaCy scripts only load the model name they're given.

## Quick start

No configuration is needed. Language is auto-detected per text from script statistics:

- `--lang auto` (default): detect per text — Korean text gets Korean handling, English text gets English handling, mixed content gets both.
- `--lang ko`: force Korean for all content.
- `--lang en`: force English.

```bash
lint-ai serve --lang ko
```

The search API also accepts a per-request override:

```json
{ "query": "학교", "user_id": "u1", "lang": "ko" }
```

## Tokenization

Korean is agglutinative: particles and endings attach to stems (`학교에` = 학교 + 에). The tokenizer is script-aware:

- **Hangul runs** (one eojeol each — spaces break runs) emit the eojeol **plus** a particle-stripped stem: `학교에` → `학교에`, `학교`. Up to two suffixes are stripped (`학교에서는` → `학교`), never to an empty stem, and the original eojeol is always kept for exact matching.
- **Han runs** emit sliding character bigrams (same convention as Chinese support).
- **Latin runs** keep the historical behavior byte-for-byte.

This means a document containing `학교에` is retrieved by the bare query `학교`, and vice versa.

## Search index (tantivy)

All TEXT fields use a CJK-aware tokenizer: Han runs index as character bigrams, Hangul runs as eojeol + stem. The same tokenizer runs at query time, so index and query segmentation agree. Pure-English text indexes byte-identically to before.

Hangul is never romanized: `normalize_for_index` and the BM25 query sanitizer keep Han/Hangul runs in the original script (no Pinyin/romanization, no English Porter stemming on CJK).

## Named entity recognition (spaCy)

Korean documents use `ko_core_news_sm` (selected per-text in Rust; an explicit `--spacy-model` still overrides). Install it with:

```bash
python3 -m pip install --break-system-packages spacy==3.8.16
python3 -m spacy download ko_core_news_sm
```

No MeCab installation is required — the model loads and tokenizes out of the box.

The model name is allowlisted in `scripts/spacy_ner.py` and `scripts/spacy_relations.py` (allowlist only; no Python logic changes).

## Question understanding

Korean interrogatives map to question kinds:

| Korean | Kind |
|--------|------|
| 누구 / 누가 | Who |
| 무엇 / 뭐 / 무슨 | What |
| 어디 | Where |
| 언제 | When |
| 얼마나 / 얼마 | How much |
| 왜 | Why |
| 어떻게 | How |
| 어느 / 어떤 | Which |

Korean is head-final, so the question word is matched at the start or as a token anywhere in the query.

## Temporal expressions

A Korean pre-layer recognizes native relative dates before the English patterns run:

- Days: 오늘, 어제, 내일, 그저께/그제, 모레
- Weeks: 지난주, 이번주, 다음주
- Months: 지난달, 이번달, 다음달
- Years: 작년, 올해, 내년
- Explicit: `2026년 9월 29일` (spaces optional)

Window sizes mirror the English arms (2 days for day expressions, 7 for weeks, 14 for months, 30 for years).

## Numbers and aggregation

- **Triggers**: 몇 → count; 총 / 합계 / 얼마나 → sum.
- **Numerals**: native words (`하나`→1 … `아흔`→90) and Sino-Korean compounds with units (`삼천원`→`3000원`, `오만개`→`50000개`) are normalized to digits.

Out of scope (documented): compound native numbers (`스물다섯`), single-character Sino-Korean (`일/이/삼` — ambiguous with non-numeric uses), and time-counter forms (`이월` — February vs "2 months").

## Stopwords

Korean particles and function words are filtered in all four stopword lists: the tokenizer (both modes), the tier-1 term ranker, and heuristic noun-phrase extraction.

## Known limitations

- **Query expansion** (WordNet/ConceptNet synonyms) is English-only and silently no-ops for Korean. Korean queries do not get synonym expansion.
- **Structured relations** (`spacy_relations.py`): `ko_core_news_sm` does not implement spaCy's `noun_chunks` iterator ([E894]), so the relations extractor fail-opens for Korean (returns no relations; lexical retrieval is unaffected). NER via `spacy_ner.py` works fully.
- **behood query-time analysis** (`scripts/behood_query.py`) uses the English spaCy model only; relation-judgment evidence degrades for Korean input (fail-open).
- **bekind** linguistic knowledge (weekdays, wh-patterns) is English-only.
