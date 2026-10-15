#!/usr/bin/env python3
"""Refresh the current 500-question LongMemEval headline in README and MkDocs.

Usage:
  python3 scripts/update_benchmark_results.py comparison/results/report.json

The report is explicit input. Only generated sections are replaced; dated
historical runs and other benchmark tracks remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED LONGMEMEVAL-S 500-QUESTION HEADLINE -->"
END = "<!-- END GENERATED LONGMEMEVAL-S 500-QUESTION HEADLINE -->"
TARGETS = (
    "README.md",
    "docs/index.md",
    "docs/benchmark.md",
    "docs/benchmark-results.md",
)


def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def replace_generated_block(text: str, body: str, path: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"{path} must contain exactly one generated-results marker pair")
    start = text.index(START) + len(START)
    end = text.index(END)
    if start > end:
        raise ValueError(f"generated-results markers are reversed in {path}")
    return text[:start] + "\n" + body.rstrip() + "\n" + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="LongMemEval-S aggregate JSON report")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path = report_path.resolve()
    try:
        report_path.relative_to(ROOT)
    except ValueError:
        parser.error("report must be stored inside this repository")

    data = json.loads(report_path.read_text())
    aggregate = data.get("aggregate", {})
    if aggregate.get("query_count") != 500:
        parser.error("report must cover exactly 500 questions")
    if data.get("index_mode") != "single" or data.get("structured_fact_retrieval") is not True:
        parser.error("headline updater requires single-index mode with structured-fact retrieval enabled")
    if str(data.get("ner_provider", "")).lower() != "heuristic":
        parser.error("headline updater requires the heuristic NER backend")

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", report_path.name)
    run_date = date_match.group(0) if date_match else "date not encoded in report filename"
    working_tree = "working-tree" in report_path.stem
    revision_match = re.search(r"-([0-9a-f]{7,40})-working-tree$", report_path.stem)
    fractional = aggregate["recall_at_k"]
    any_hit = aggregate["recall_any_at_k"]
    mrr = aggregate["mrr"]
    ndcg = aggregate["ndcg_at_10"]
    mean_ms = aggregate["timings"]["total_ms"]
    relative_report = report_path.relative_to(ROOT).as_posix()

    raw_report = f"https://github.com/RooAGI/Membrick/blob/main/{relative_report}"
    table = "\n".join((
        f"**LongMemEval-S, 500 questions · {run_date} · single index · heuristic NER · structured-fact retrieval · no embeddings**",
        *( [f"Working-tree run based on `{revision_match.group(1)}`; not a clean release measurement."] if working_tree and revision_match else ["Working-tree run; not a clean release measurement."] if working_tree else [] ),
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Fractional Recall@5 / @10 / @20 | {percent(fractional['5'])} / {percent(fractional['10'])} / {percent(fractional['20'])} |",
        f"| Any-hit Recall@5 / @10 / @20 | {percent(any_hit['5'], 1)} / {percent(any_hit['10'], 1)} / {percent(any_hit['20'], 1)} |",
        f"| MRR | {percent(mrr)} |",
        f"| NDCG@10 | {percent(ndcg)} |",
        f"| Mean search time | {mean_ms:.2f} ms |",
        "",
        f"Source report: [`{report_path.name}`]({raw_report}).",
    ))

    # The website home page uses compact cards; its generated block links to
    # the same report as the README and detailed benchmark pages.
    homepage = "\n".join((
        f"<p><strong>Latest LongMemEval-S result · 500 questions · {run_date}</strong></p>",
        *( [f"<p class=\"benchmark-note\">Working-tree run based on <code>{revision_match.group(1)}</code>; not a clean release measurement.</p>"] if working_tree and revision_match else ["<p class=\"benchmark-note\">Working-tree run; not a clean release measurement.</p>"] if working_tree else [] ),
        "<section class=\"proof-grid proof-grid--five\" aria-label=\"Latest LongMemEval-S benchmark results\">",
        f"  <div><strong>{percent(fractional['5'])}</strong><span>fractional Recall@5</span></div>",
        f"  <div><strong>{percent(any_hit['10'], 1)}</strong><span>any-hit Recall@10</span></div>",
        f"  <div><strong>{percent(mrr)}</strong><span>MRR</span></div>",
        f"  <div><strong>{percent(ndcg)}</strong><span>NDCG@10</span></div>",
        f"  <div><strong>{mean_ms:.2f} ms</strong><span>mean search time</span></div>",
        "</section>",
        f"<p class=\"benchmark-note\">Single index · heuristic NER · structured-fact retrieval · no embeddings. <a href=\"benchmark-results/\">Methods and report</a> · <a href=\"{raw_report}\">Raw JSON</a>.</p>",
    ))

    bodies = {
        "README.md": table,
        "docs/index.md": homepage,
        "docs/benchmark.md": table,
        "docs/benchmark-results.md": table,
    }
    updated_files: dict[Path, str] = {}
    for relative_path in TARGETS:
        path = ROOT / relative_path
        updated_files[path] = replace_generated_block(path.read_text(), bodies[relative_path], relative_path)

    for path, content in updated_files.items():
        path.write_text(content)
    print(f"Updated the current 500-question headline in {len(updated_files)} files from {relative_report}:")
    for relative_path in TARGETS:
        print(f"  {relative_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark results update failed: {error}", file=sys.stderr)
        raise SystemExit(1)
