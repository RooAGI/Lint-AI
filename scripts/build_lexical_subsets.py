#!/usr/bin/env python3
"""Build lint-ai lexical JSON subsets from upstream WordNet and ConceptNet files.

The output schema intentionally matches data/lexical/*_subset.json:
[
  {
    "term": "query",
    "related": [
      {"term": "search", "relation": "Synonym", "confidence": 0.95}
    ]
  }
]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict


WORDNET_FILES = ("data.noun", "data.verb", "data.adj", "data.adv")
CONCEPTNET_RELATIONS = {
    "/r/Synonym": "Synonym",
    "/r/SimilarTo": "SimilarTo",
    "/r/RelatedTo": "RelatedTo",
}
MAX_RELATED_PER_TERM = 12
MIN_CONCEPTNET_WEIGHT = 1.0


def normalize(term: str) -> str:
    term = term.replace("_", " ").replace("-", " ").lower()
    term = re.sub(r"[^a-z0-9 ]+", " ", term)
    return re.sub(r"\s+", " ", term).strip()


def read_seed_terms(path: Path) -> set[str]:
    terms = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = normalize(line)
        if key:
            terms.add(key)
    return terms


def read_seed_terms_many(paths: list[Path]) -> set[str]:
    terms = set()
    for path in paths:
        terms.update(read_seed_terms(path))
    return terms


def add_edge(
    out: DefaultDict[str, dict[str, tuple[str, float]]],
    source: str,
    target: str,
    relation: str,
    confidence: float,
) -> None:
    source = normalize(source)
    target = normalize(target)
    if not source or not target or source == target:
        return
    current = out[source].get(target)
    if current is None or confidence > current[1]:
        out[source][target] = (relation, confidence)


def parse_wordnet_data_file(
    path: Path,
    seeds: set[str],
    out: DefaultDict[str, dict[str, tuple[str, float]]],
) -> None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("  "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            word_count = int(parts[3], 16)
        except ValueError:
            continue
        word_start = 4
        word_end = word_start + word_count * 2
        if len(parts) < word_end:
            continue
        words = [normalize(parts[idx]) for idx in range(word_start, word_end, 2)]
        words = [word for word in words if word]
        if not any(word in seeds for word in words):
            continue
        for source in words:
            if source not in seeds:
                continue
            for target in words:
                add_edge(out, source, target, "Synonym", 0.95)


def build_wordnet_subset(wordnet_dict: Path, seeds: set[str]) -> list[dict]:
    out: DefaultDict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    wordnet_dict = resolve_wordnet_dict(wordnet_dict)
    for filename in WORDNET_FILES:
        path = wordnet_dict / filename
        if path.exists():
            parse_wordnet_data_file(path, seeds, out)
    return format_entries(out)


def resolve_wordnet_dict(path: Path) -> Path:
    if any((path / filename).exists() for filename in WORDNET_FILES):
        return path
    nested = path / "dict"
    if any((nested / filename).exists() for filename in WORDNET_FILES):
        return nested
    return path


def conceptnet_node_to_term(node: str) -> str | None:
    parts = node.strip().split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return normalize(parts[3])


def parse_conceptnet_assertions(
    assertions_path: Path,
    seeds: set[str],
    out: DefaultDict[str, dict[str, tuple[str, float]]],
) -> None:
    opener = gzip.open if assertions_path.suffix == ".gz" else open
    with opener(assertions_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                continue
            _, raw_relation, raw_start, raw_end, raw_meta = fields
            relation = CONCEPTNET_RELATIONS.get(raw_relation)
            if relation is None:
                continue
            start = conceptnet_node_to_term(raw_start)
            end = conceptnet_node_to_term(raw_end)
            if start is None or end is None:
                continue
            if start not in seeds and end not in seeds:
                continue
            try:
                weight = float(json.loads(raw_meta).get("weight", 1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                weight = 1.0
            if weight < MIN_CONCEPTNET_WEIGHT:
                continue
            confidence = min(0.99, max(0.5, weight / 4.0))
            if start in seeds:
                add_edge(out, start, end, relation, confidence)
            if end in seeds and relation in {"Synonym", "SimilarTo"}:
                add_edge(out, end, start, relation, confidence)


def build_conceptnet_subset(assertions_path: Path, seeds: set[str]) -> list[dict]:
    out: DefaultDict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    parse_conceptnet_assertions(assertions_path, seeds, out)
    return format_entries(out)


def format_entries(edges: DefaultDict[str, dict[str, tuple[str, float]]]) -> list[dict]:
    entries = []
    for term in sorted(edges):
        related = sorted(
            edges[term].items(),
            key=lambda item: (-item[1][1], item[0]),
        )[:MAX_RELATED_PER_TERM]
        if not related:
            continue
        entries.append(
            {
                "term": term,
                "related": [
                    {
                        "term": target,
                        "relation": relation,
                        "confidence": round(confidence, 3),
                    }
                    for target, (relation, confidence) in related
                ],
            }
        )
    return entries


def write_json(path: Path, value: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-terms",
        type=Path,
        action="append",
        default=[Path("data/lexical/seed_terms.txt")],
        help="Seed terms file. Repeat the flag to combine multiple files.",
    )
    parser.add_argument("--wordnet-dict", type=Path)
    parser.add_argument("--conceptnet-assertions", type=Path)
    parser.add_argument("--wordnet-out", type=Path, default=Path("data/lexical/wordnet_subset.json"))
    parser.add_argument(
        "--conceptnet-out",
        type=Path,
        default=Path("data/lexical/conceptnet_subset.json"),
    )
    args = parser.parse_args()

    seeds = read_seed_terms_many(args.seed_terms)
    resolved_wordnet = resolve_wordnet_dict(args.wordnet_dict) if args.wordnet_dict else None
    if args.wordnet_dict:
        write_json(args.wordnet_out, build_wordnet_subset(resolved_wordnet, seeds))
    if args.conceptnet_assertions:
        write_json(
            args.conceptnet_out,
            build_conceptnet_subset(args.conceptnet_assertions, seeds),
        )
    if not args.wordnet_dict and not args.conceptnet_assertions:
        parser.error("pass --wordnet-dict and/or --conceptnet-assertions")


if __name__ == "__main__":
    main()
