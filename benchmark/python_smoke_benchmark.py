import argparse
import json
import string
import sys
import time
from pathlib import Path

import lint_ai


ABSTENTION_TYPES = {
    "single-session-user_abs",
    "multi-session_abs",
    "knowledge-update_abs",
    "temporal-reasoning_abs",
}
QUERY_SYNTAX_CHARS = str.maketrans({char: " " for char in string.punctuation})


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def recall_any_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved, relevant):
    for idx, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (idx + 1)
    return 0.0


def query_text_for_index_store(query):
    return " ".join(query.translate(QUERY_SYNTAX_CHARS).split())


def run_entry(entry, ks):
    store = lint_ai.IndexStore()
    for session_id, turns in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        for turn_idx, turn in enumerate(turns):
            doc_id = f"{session_id}::turn{turn_idx}"
            content = f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
            source = f"longmemeval/session/{session_id}/turn/{turn_idx}"
            store.upsert(doc_id, content, source, entry.get("question_date"), session_id)

    started = time.perf_counter()
    query_text = query_text_for_index_store(entry["question"])
    results = store.query(query_text, max(ks))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    retrieved = dedupe_preserve_order(
        result.get("group_id") or result["doc_id"].split("::turn", 1)[0]
        for result in results
    )
    relevant = set(entry.get("answer_session_ids", []))

    return {
        "id": entry["question_id"],
        "query": entry["question"],
        "index_query": query_text,
        "elapsed_ms": elapsed_ms,
        "retrieved_session_ids": retrieved,
        "recall_at_k": {str(k): recall_at_k(retrieved, relevant, k) for k in ks},
        "recall_any_at_k": {str(k): recall_any_at_k(retrieved, relevant, k) for k in ks},
        "mrr": reciprocal_rank(retrieved, relevant),
    }


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run the Python IndexStore smoke benchmark.")
    parser.add_argument(
        "--longmemeval",
        default="benchmark/smoke_longmemeval.json",
        help="Path to a LongMemEval-shaped JSON file.",
    )
    parser.add_argument("--k", dest="ks", type=int, action="append")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    requested_ks = args.ks if args.ks is not None else [1, 3]
    ks = sorted(set(k for k in requested_ks if k > 0))
    entries = [
        entry
        for entry in json.loads(Path(args.longmemeval).read_text())
        if entry.get("question_type") not in ABSTENTION_TYPES
    ]
    if args.limit is not None:
        entries = entries[: args.limit]

    per_query = []
    for idx, entry in enumerate(entries, 1):
        per_query.append(run_entry(entry, ks))
        if idx % 25 == 0 or idx == len(entries):
            print(f"[{idx}/{len(entries)}] completed", file=sys.stderr)

    report = {
        "aggregate": {
            "query_count": len(per_query),
            "avg_elapsed_ms": average(item["elapsed_ms"] for item in per_query),
            "recall_at_k": {
                str(k): average(item["recall_at_k"][str(k)] for item in per_query)
                for k in ks
            },
            "recall_any_at_k": {
                str(k): average(item["recall_any_at_k"][str(k)] for item in per_query)
                for k in ks
            },
            "mrr": average(item["mrr"] for item in per_query),
        },
        "per_query": per_query,
    }
    payload = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"wrote Python benchmark report to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
