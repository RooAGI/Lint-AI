#!/usr/bin/env python3
"""Build lint-ai lexical JSON subsets from upstream WordNet and ConceptNet files.

The output schema intentionally matches data/lexical/*_subset.json(.gz):
[
  {
    "term": "query",
    "related": [
      {"term": "search", "relation": "Synonym", "confidence": 0.95}
    ]
  }
]

Outputs are gzip-compressed (.json.gz): the Rust loader reads them via
include_bytes! + libflate, which keeps the crates.io package under the
size limit while embedding the full vocabulary in the binary.
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
# Parse-time candidate cap per term (must stay >= MAX_RELATED_PER_TERM; the
# final format keeps the top MAX_RELATED_PER_TERM, so pruning to this bound
# while streaming is lossless and keeps memory flat on huge dumps).
PARSE_PRUNE_CAP = 48
# Matches CONCEPTNET_MIN_CONFIDENCE in src/query_expansion.rs: RelatedTo edges
# below this never survive loading, so there is no point emitting them.
MIN_RELATEDTO_CONFIDENCE = 0.82


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
    # Bound memory while streaming huge dumps: the final format keeps only the
    # top max_related_per_term entries per term, so pruning the parse-time
    # candidate set to a slightly larger bound is lossless.
    if len(out[source]) > PARSE_PRUNE_CAP:
        best = sorted(out[source].items(), key=lambda item: (-item[1][1], item[0]))[
            :PARSE_PRUNE_CAP
        ]
        out[source] = dict(best)


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
    min_weight: float,
    min_relatedto_confidence: float,
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
            if weight < min_weight:
                continue
            confidence = min(0.99, max(0.5, weight / 4.0))
            if relation == "RelatedTo" and confidence < min_relatedto_confidence:
                continue
            if start in seeds:
                add_edge(out, start, end, relation, confidence)
            # RelatedTo is conceptually symmetric ("A related to B" implies the
            # reverse for query expansion); the dump usually asserts one
            # direction, so add the reverse too. Synonym/SimilarTo were already
            # reversed here and get a second reversal in the Rust loader.
            if end in seeds and relation in {"Synonym", "SimilarTo", "RelatedTo"}:
                add_edge(out, end, start, relation, confidence)


def build_conceptnet_subset(
    assertions_path: Path,
    seeds: set[str],
    min_weight: float,
    min_relatedto_confidence: float,
    max_related_per_term: int,
) -> list[dict]:
    out: DefaultDict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    parse_conceptnet_assertions(assertions_path, seeds, out, min_weight, min_relatedto_confidence)
    return format_entries(out, max_related_per_term)


def format_entries(
    edges: DefaultDict[str, dict[str, tuple[str, float]]],
    max_related_per_term: int = MAX_RELATED_PER_TERM,
) -> list[dict]:
    entries = []
    for term in sorted(edges):
        related = sorted(
            edges[term].items(),
            key=lambda item: (-item[1][1], item[0]),
        )[:max_related_per_term]
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


def write_json(path: Path, value: list[dict], compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    data = text.encode("utf-8")
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=9)
    path.write_bytes(data + (b"" if path.suffix == ".gz" else b"\n"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-terms",
        type=Path,
        action="append",
        default=[],
        help="Seed terms file. Repeat the flag to combine multiple files.",
    )
    parser.add_argument("--wordnet-dict", type=Path)
    parser.add_argument("--conceptnet-assertions", type=Path)
    parser.add_argument(
        "--wordnet-out",
        type=Path,
        default=Path("data/lexical/wordnet_subset.json.gz"),
    )
    parser.add_argument(
        "--conceptnet-out",
        type=Path,
        default=Path("data/lexical/conceptnet_subset.json.gz"),
    )
    parser.add_argument(
        "--min-conceptnet-weight",
        type=float,
        default=MIN_CONCEPTNET_WEIGHT,
        help="Drop ConceptNet edges below this weight (default %(default)s).",
    )
    parser.add_argument(
        "--min-relatedto-confidence",
        type=float,
        default=MIN_RELATEDTO_CONFIDENCE,
        help="Drop ConceptNet RelatedTo edges below this confidence; matches the "
        "loader gate in src/query_expansion.rs (default %(default)s).",
    )
    parser.add_argument(
        "--max-related-per-term",
        type=int,
        default=MAX_RELATED_PER_TERM,
        help="Cap related terms per entry (default %(default)s).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON (no indentation) to keep large files small.",
    )
    args = parser.parse_args()

    seeds = read_seed_terms_many(args.seed_terms)
    resolved_wordnet = resolve_wordnet_dict(args.wordnet_dict) if args.wordnet_dict else None
    if args.wordnet_dict:
        write_json(
            args.wordnet_out,
            build_wordnet_subset(resolved_wordnet, seeds),
            compact=args.compact,
        )
    if args.conceptnet_assertions:
        write_json(
            args.conceptnet_out,
            build_conceptnet_subset(
                args.conceptnet_assertions,
                seeds,
                args.min_conceptnet_weight,
                args.min_relatedto_confidence,
                args.max_related_per_term,
            ),
            compact=args.compact,
        )
    if not args.wordnet_dict and not args.conceptnet_assertions:
        parser.error("pass --wordnet-dict and/or --conceptnet-assertions")


if __name__ == "__main__":
    main()
