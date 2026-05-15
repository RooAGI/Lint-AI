# Benchmark Suite

This directory holds the retrieval benchmarks for `lint-ai`, including haystack-style corpora, LongMemEval-S runs, and the EverMemBench-Static setup.

## Dataset format

Provide a JSON file with this shape:

```json
{
  "documents": [
    {
      "doc_id": "doc-1",
      "source": "docs/alpha.md",
      "content": "# Title\nDocument text...",
      "concept": "optional-concept",
      "headings": ["Title"],
      "links": ["docs/beta.md"],
      "timestamp": "2026-04-25T00:00:00Z",
      "author_agent": "optional"
    }
  ],
  "queries": [
    {
      "id": "q-1",
      "query": "what does alpha say",
      "relevant_doc_ids": ["doc-1"],
      "relevant_chunk_ids": []
    }
  ]
}
```

Notes:
- `relevant_doc_ids` and `relevant_chunk_ids` are both optional arrays.
- If only `relevant_chunk_ids` are provided, the benchmark resolves them to `doc_id` via indexed chunk metadata.

## Haystack Run

```bash
cargo run --bin haystack_benchmark -- \
  --dataset benchmark/sample_dataset.json \
  --k 1 --k 3 --k 5 --k 10 \
  --out benchmark/results.json
```

To run the academic LongMemEval-S corpus directly and index each session as turn-level chunks:

```bash
cargo run --bin haystack_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_cleaned.json \
  --limit 20 \
  --k 5 --k 10 --k 20 \
  --out benchmark/results.json
```

In LongMemEval mode, turn docs are grouped back to their parent session for scoring so you measure session retrieval quality while using finer-grained chunks for indexing.

If you want the raw Hugging Face copy instead of the cleaned one, use:

```bash
cargo run --bin haystack_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --limit 20 \
  --k 5 --k 10 --k 20 \
  --out benchmark/results.json
```

If you want question-scoped haystack retrieval, where each query only searches the sessions attached to that question:

```bash
cargo run --bin haystack_scoped_benchmark -- \
  --longmemeval benchmark/data/longmemeval_s_raw.json \
  --limit 20 \
  --k 5 --k 10 --k 20 \
    --out benchmark/results.json
```

Scoped LongMemEval reporting includes both `recall@k` and `recall_any@k`. The any-hit metric matches the interpretation used by the current LongMemEval-S release notes and README.

To prepare a reusable turn-based dataset file first, run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

src = Path('benchmark/data/longmemeval_s_cleaned.json')
out = Path('benchmark/data/longmemeval_turn_full_for_lintai.json')
raw = json.loads(src.read_text())

abst = {
    'single-session-user_abs',
    'multi-session_abs',
    'knowledge-update_abs',
    'temporal-reasoning_abs',
}

entries = [e for e in raw if e.get('question_type') not in abst]
documents = []
queries = []
seen_docs = set()

for entry in entries:
    session_turns = {
        sid: turns
        for sid, turns in zip(
            entry.get('haystack_session_ids', []),
            entry.get('haystack_sessions', []),
        )
    }
    for session_id, turns in session_turns.items():
        for turn_idx, turn in enumerate(turns):
            doc_id = f'{session_id}::turn{turn_idx}'
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            documents.append({
                'doc_id': doc_id,
                'source': f'longmemeval/session/{session_id}/turn/{turn_idx}',
                'content': f"{turn.get('role', 'unknown')}: {turn.get('content', '')}",
                'concept': 'longmemeval-turn',
                'headings': [f'session:{session_id}'],
                'links': [],
                'timestamp': entry.get('question_date'),
                'author_agent': None,
            })

    queries.append({
        'id': entry['question_id'],
        'query': entry['question'],
        'relevant_doc_ids': entry.get('answer_session_ids', []),
        'relevant_chunk_ids': [],
    })

with out.open('w') as f:
    json.dump({'documents': documents, 'queries': queries}, f)

print(f'wrote {out}')
print(f'documents={len(documents)} queries={len(queries)}')
PY
```

## Report Metrics

- `recall_at_k`: average recall at each configured K.
- `recall_any_at_k`: 1.0 when any gold session is in the top K, otherwise 0.0.
- `mrr`: mean reciprocal rank.
- `ndcg_at_10`: normalized DCG at 10.

The output JSON includes both aggregate metrics and per-query details.
