#!/usr/bin/env python3
"""Pillars 1+2 analysis: aggregate N haystack runs, compute tokens/query offline.

Reads per-run JSON files from run_pillar1.sh and the LongMemEval-S dataset.
Writes a metrics JSON and prints a summary (mean ± population std dev).

Usage:
    analyze_pillars.py --dataset <longmemeval.json> --out <metrics.json> run1.json [run2.json ...]
"""
import argparse
import json
import statistics
import sys

KS = [1, 3, 5, 10, 20]


def build_session_texts(data_path):
    """Map session_id -> full text (turns joined) from the dataset."""
    data = json.load(open(data_path))
    texts = {}
    for entry in data:
        for sid, turns in zip(entry.get("haystack_session_ids", []),
                              entry.get("haystack_sessions", [])):
            if sid not in texts:
                texts[sid] = "\n".join(
                    f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)
    return texts


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = round(p / 100.0 * (len(sorted_vals) - 1))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LongMemEval-S cleaned JSON")
    ap.add_argument("--out", required=True, help="output metrics JSON path")
    ap.add_argument("runs", nargs="+", help="per-run haystack result JSONs")
    args = ap.parse_args()

    run_files = args.runs
    runs = [json.load(open(p)) for p in run_files]
    for path, r in zip(run_files, runs):
        n = r["aggregate"]["query_count"]
        assert n == 500, f"{path}: expected 500 questions, got {n}"

    metrics = {}
    for k in KS:
        frac = [r["aggregate"]["recall_at_k"][str(k)] for r in runs]
        anyh = [r["aggregate"]["recall_any_at_k"][str(k)] for r in runs]
        metrics[f"frac_recall@{k}"] = (statistics.mean(frac), statistics.pstdev(frac))
        metrics[f"anyhit_recall@{k}"] = (statistics.mean(anyh), statistics.pstdev(anyh))
    for m in ["mrr", "ndcg_at_10"]:
        vals = [r["aggregate"][m] for r in runs]
        metrics[m] = (statistics.mean(vals), statistics.pstdev(vals))

    # Pillar 2: latency from per-query timings.
    lat_means, lat_p50s, lat_p95s = [], [], []
    for r in runs:
        t = sorted(q["timings"]["total_ms"] for q in r["per_query"])
        lat_means.append(sum(t) / len(t))
        lat_p50s.append(percentile(t, 50))
        lat_p95s.append(percentile(t, 95))
    metrics["latency_mean_ms"] = (statistics.mean(lat_means), statistics.pstdev(lat_means))
    metrics["latency_p50_ms"] = (statistics.mean(lat_p50s), statistics.pstdev(lat_p50s))
    metrics["latency_p95_ms"] = (statistics.mean(lat_p95s), statistics.pstdev(lat_p95s))

    # Pillar 2: tokens/query offline (chars/4 over retrieved session texts).
    texts = build_session_texts(args.dataset)
    print(f"sessions in text map: {len(texts)}")
    tok = {cut: [] for cut in (5, 10, 20, None)}
    missing = 0
    for r in runs:
        for q in r["per_query"]:
            full = []
            for sid in q["retrieved_session_ids"]:
                t = texts.get(sid)
                if t is None:
                    missing += 1
                    continue
                full.append(len(t) / 4.0)
            for cut in (5, 10, 20):
                tok[cut].append(sum(full[:cut]))
            tok[None].append(sum(full))
    print(f"missing session texts: {missing}")
    for cut in (5, 10, 20, None):
        name = f"tokens_per_query_top{cut}" if cut else "tokens_per_query_full_list"
        v = tok[cut]
        metrics[name] = (statistics.mean(v), statistics.pstdev(v))

    print(f"\n=== PILLARS 1+2 ({len(runs)}-run mean ± std) ===")
    for k, (m, s) in metrics.items():
        print(f"{k}: {m:.4f} ± {s:.4f}")

    mean_ms = metrics["latency_mean_ms"][0]
    print(f"\nqueries_per_sec: {1000.0 / mean_ms:.1f}")

    json.dump(
        {k: {"mean": m, "std": s} for k, (m, s) in metrics.items()},
        open(args.out, "w"),
        indent=1,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
