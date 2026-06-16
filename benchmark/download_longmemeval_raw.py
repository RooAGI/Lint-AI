#!/usr/bin/env python3
"""Download the raw LongMemEval-S JSON from Hugging Face.

This fetches the exact source file used by the benchmark:
https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s?download=true

The script writes the file to benchmark/data/longmemeval_s_raw.json by default
and verifies the SHA-256 hash so the local copy stays byte-for-byte identical.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/"
    "resolve/main/longmemeval_s?download=true"
)
EXPECTED_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
EXPECTED_RECORDS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the raw LongMemEval-S dataset used by the benchmarks."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Dataset download URL. Defaults to the official Hugging Face raw file.",
    )
    parser.add_argument(
        "--out",
        default=Path("benchmark/data/longmemeval_s_raw.json"),
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
