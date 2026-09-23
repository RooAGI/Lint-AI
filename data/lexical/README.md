# Lexical expansion data

`wordnet_subset.json.gz` and `conceptnet_subset.json.gz` feed query-side synonym
expansion in `src/query_expansion.rs` (loaded via `include_bytes!` + gzip, so the
embedded vocabulary does not blow the crates.io package size limit).

## Provenance

- **WordNet**: all single-word lemmas from Princeton WordNet 3.0, with
  same-synset lemmas as `Synonym` edges (confidence 0.95). ~57k entries.
- **ConceptNet**: ConceptNet 5.7 assertions dump
  (`conceptnet-assertions-5.7.0.csv.gz`), `Synonym`/`SimilarTo`/`RelatedTo`
  edges touching the WordNet lemma vocabulary. `RelatedTo` edges below
  confidence 0.82 are dropped at generation time because the Rust loader
  (`CONCEPTNET_MIN_CONFIDENCE`) would discard them anyway. `RelatedTo` is
  treated as symmetric (the dump usually asserts one direction).

## Regenerating

```bash
# WordNet dict: unpack nltk_data/packages/corpora/wordnet.zip to a dir
# containing data.noun / data.verb / data.adj / data.adv
python3 scripts/build_lexical_subsets.py \
  --seed-terms <(python3 scripts/wordnet_lemma_seeds.py) \
  --wordnet-dict ~/nltk_data/corpora/wordnet \
  --compact
python3 scripts/build_lexical_subsets.py \
  --seed-terms ~/workspace/lexical/wn_seeds.txt \
  --conceptnet-assertions ~/workspace/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
  --compact
```

Tunable knobs: `--min-conceptnet-weight` (default 1.0),
`--min-relatedto-confidence` (default 0.82, must match the loader),
`--max-related-per-term` (default 12).
