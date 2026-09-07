#!/usr/bin/env python3
"""Download the cleaned LongMemEval-S JSON from Hugging Face.

The benchmark pins the official cleaned replacement dataset at a specific
revision and verifies both SHA-256 and record count for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/"
    "longmemeval_s_cleaned.json?download=true"
)
EXPECTED_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
EXPECTED_RECORDS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned cleaned LongMemEval-S dataset used by the benchmarks."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Dataset download URL. Defaults to the pinned cleaned LongMemEval-S file.",
    )
    parser.add_argument(
        "--out",
        default=Path("benchmark/data/longmemeval_s_cleaned.json"),
        type=Path,
        help="Output file path.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA-256 and record-count verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url) as response:
        data = response.read()

    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)")

    if args.skip_verify:
        return 0

    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != EXPECTED_SHA256:
        print(
            f"sha256 mismatch: expected {EXPECTED_SHA256}, got {sha256}",
            file=sys.stderr,
        )
        return 1

    import json

    obj = json.loads(data)
    if not isinstance(obj, list) or len(obj) != EXPECTED_RECORDS:
        print(
            f"record count mismatch: expected {EXPECTED_RECORDS}, got "
            f"{len(obj) if isinstance(obj, list) else type(obj).__name__}",
            file=sys.stderr,
        )
        return 1

    print(f"verified sha256={sha256} records={len(obj)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
