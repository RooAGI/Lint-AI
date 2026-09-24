#!/usr/bin/env python3
"""Build pillar-4 LLM-judge inputs: per-question top-k retrieved sessions,
their full texts, and the reference answers, from one canonical haystack run.

Usage:
    build_pillar4_inputs.py --dataset <longmemeval.json> --run <haystack_run.json> \
        --out <judge_inputs.jsonl> [--k 20]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="canonical haystack run JSON")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    texts = {}
    for entry in dataset:
        for sid, turns in zip(entry.get("haystack_session_ids", []),
                              entry.get("haystack_sessions", [])):
            if sid not in texts:
                texts[sid] = "\n".join(
                    f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)
    qmap = {e["question"]: e for e in dataset}

    run = json.load(open(args.run))
    n_out, n_missing_q, n_missing_t = 0, 0, 0
    with open(args.out, "w") as out:
        for idx, q in enumerate(run["per_query"]):
            entry = qmap.get(q["query"])
            if entry is None:
                n_missing_q += 1
                continue
            ctx = []
            for sid in q["retrieved_session_ids"][:args.k]:
                t = texts.get(sid)
                if t is None:
                    n_missing_t += 1
                    continue
                ctx.append({"session_id": sid, "text": t})
            out.write(json.dumps({
                "question_id": idx,
                "question": entry["question"],
                "question_type": entry.get("question_type"),
                "reference_answer": entry.get("answer"),
                "top_k": args.k,
                "retrieved_context": ctx,
            }) + "\n")
            n_out += 1
    print(f"wrote {n_out} records to {args.out}; "
          f"missing questions: {n_missing_q}; missing texts: {n_missing_t}")


if __name__ == "__main__":
    sys.exit(main())
