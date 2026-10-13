# Lexical Expansion Data

`membrick` uses compact JSON lexical subsets at:

- `data/lexical/wordnet_subset.json`
- `data/lexical/conceptnet_subset.json`

These files are intentionally small enough to check into the repo. They should be regenerated from upstream lexical resources instead of expanded through hardcoded Rust aliases.

## Upstream Sources

- ConceptNet assertions: `https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz`
- Princeton WordNet: download the WordNet database package from `https://wordnet.princeton.edu/`

ConceptNet is licensed under CC BY-SA 4.0. WordNet has its own Princeton license and citation requirements. Keep those requirements in mind before redistributing generated data.

## Generate Subsets

Create or edit seed terms in:

```bash
data/lexical/seed_terms.txt
```

Generate from a local WordNet `dict` directory:

```bash
python3 scripts/build_lexical_subsets.py \
  --wordnet-dict /path/to/WordNet-3.0/dict
```

Generate from a local ConceptNet assertions dump:

```bash
python3 scripts/build_lexical_subsets.py \
  --conceptnet-assertions /path/to/conceptnet-assertions-5.7.0.csv.gz
```

Generate both in one command:

```bash
python3 scripts/build_lexical_subsets.py \
  --wordnet-dict /path/to/WordNet-3.0/dict \
  --conceptnet-assertions /path/to/conceptnet-assertions-5.7.0.csv.gz
```

The generated JSON uses the existing schema:

```json
[
  {
    "term": "query",
    "related": [
      {"term": "search", "relation": "Synonym", "confidence": 0.95}
    ]
  }
]
```

## Policy

- Keep benchmark-specific synonyms out of Rust code.
- Add domain vocabulary to `seed_terms.txt`.
- Regenerate the JSON from upstream data.
- Prefer `Synonym` and `SimilarTo` for bidirectional expansion.
- Treat `RelatedTo` as directional unless the source explicitly supports symmetry.
