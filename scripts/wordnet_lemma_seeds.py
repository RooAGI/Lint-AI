#!/usr/bin/env python3
"""Print all single-word WordNet lemmas (one per line) for use as
--seed-terms input to build_lexical_subsets.py.

Reads the Princeton WordNet dict files (data.noun / data.verb / data.adj /
data.adv) directly, no NLTK required.

Usage:
    python3 scripts/wordnet_lemma_seeds.py ~/nltk_data/corpora/wordnet
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WORDNET_FILES = ("data.noun", "data.verb", "data.adj", "data.adv")


def iter_lemmas(wordnet_dict: Path):
    for filename in WORDNET_FILES:
        path = wordnet_dict / filename
        if not path.exists():
            continue
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
            for idx in range(word_count):
                word = parts[4 + idx * 2].replace("_", " ").lower()
                word = re.sub(r"[^a-z0-9 ]+", " ", word)
                word = re.sub(r"\s+", " ", word).strip()
                if re.fullmatch(r"[a-z][a-z0-9]*", word):
                    yield word


def main() -> None:
    wordnet_dict = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("wordnet")
    seeds = sorted(set(iter_lemmas(wordnet_dict)))
    sys.stdout.write("\n".join(seeds) + "\n")


if __name__ == "__main__":
    main()
