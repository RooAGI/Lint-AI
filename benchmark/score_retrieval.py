#!/usr/bin/env python3
"""Score ranked session IDs with one shared LongMemEval metric implementation.

System evaluators should only produce per-question ranked_session_ids. This
script owns the labels, cutoff handling, and metric formulas for comparisons.
"""
import argparse
import json
import math
from pathlib import Path


def dcg(rels):
    return sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(rels))


def score(ranked, gold):
    gold = set(gold)
    out = {}
    for k in (5, 10):
        out[f"recall_any_at_{k}"] = float(bool(set(ranked[:k]) & gold))
    out["recall_any_at_20"] = float(bool(set(ranked[:20]) & gold))
    first = next((i + 1 for i, sid in enumerate(ranked) if sid in gold), None)
    out["mrr"] = 1.0 / first if first else 0.0
    rels = [sid in gold for sid in ranked[:10]]
    ideal = [True] * min(10, len(gold))
    out["ndcg_at_10"] = dcg(rels) / dcg(ideal) if ideal else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lintai", required=True)
    ap.add_argument("--agentmemory", required=True)
    args = ap.parse_args()
    dataset = {x["question_id"]: x for x in json.loads(Path(args.dataset).read_text())}
    reports = {
        "Lint-AI": json.loads(Path(args.lintai).read_text())["per_query"],
        "AgentMemory": json.loads(Path(args.agentmemory).read_text())["per_question"],
    }
    for name, rows in reports.items():
        values = []
        for row in rows:
            qid = row.get("id", row.get("question_id"))
            entry = dataset[qid]
            ranked = row.get("retrieved_session_ids", [])
            values.append(score(ranked, entry["answer_session_ids"]))
        n = len(values)
        print(name, json.dumps({
            "questions": n,
            "recall_any_at_5": sum(x["recall_any_at_5"] for x in values) / n,
            "recall_any_at_10": sum(x["recall_any_at_10"] for x in values) / n,
            "recall_any_at_20": sum(x["recall_any_at_20"] for x in values) / n,
            "mrr": sum(x["mrr"] for x in values) / n,
            "ndcg_at_10": sum(x["ndcg_at_10"] for x in values) / n,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
