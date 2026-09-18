#!/usr/bin/env python3
"""Run the complete cold-start layout throughput harness."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/server")
parser.add_argument("--features", default="agent-integrations")
parser.add_argument("--skip-build", action="store_true")
parser.add_argument("--records", type=int, default=23366)
parser.add_argument("--requests", type=int, default=100)
parser.add_argument("--warmup-requests", type=int, default=0)
parser.add_argument("--repetitions", type=int, default=5)
parser.add_argument("--base-port", type=int, default=18300)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

if args.repetitions < 1:
    parser.error("--repetitions must be positive")

if not args.skip_build:
    build = ["cargo", "build", "--release", "--bin", "server"]
    if args.features:
        build.extend(["--features", args.features])
    subprocess.run(build, check=True, cwd=ROOT)

results = {"single": [], "global": [], "segment": []}
run_number = 0
for mode in results:
    for repetition in range(1, args.repetitions + 1):
        run_number += 1
        command = [
            sys.executable,
            str(ROOT / "comparison/throughput.py"),
            "--server-bin",
            str(args.server_bin),
            "--mode",
            mode,
            "--records",
            str(args.records),
            "--requests",
            str(args.requests),
            "--warmup-requests",
            str(args.warmup_requests),
            "--port",
            str(args.base_port + run_number),
            "--no-cache",
        ]
        completed = subprocess.run(
            command,
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout[completed.stdout.index("{"):])
        results[mode].append(
            {
                "run": repetition,
                "measurements": payload["measurements"],
            }
        )

summary = {}
for mode, runs in results.items():
    summary[mode] = {}
    for index, concurrency in enumerate((1, 10)):
        summary[mode][f"c{concurrency}"] = {
            "throughput_median_req_s": round(
                statistics.median(
                    run["measurements"][index]["throughput_per_s"] for run in runs
                ),
                2,
            ),
            "p50_median_ms": round(
                statistics.median(
                    run["measurements"][index]["p50_ms"] for run in runs
                ),
                3,
            ),
        }

report = {
    "benchmark": "layout-throughput-harness",
    "records": args.records,
    "requests_per_cell": args.requests,
    "warmup_requests_per_cell": args.warmup_requests,
    "repetitions": args.repetitions,
    "cache": "disabled",
    "server": str(args.server_bin),
    "summary": summary,
    "runs": results,
}
serialized = json.dumps(report, indent=2) + "\n"
print(serialized, end="")
if args.output:
    args.output.write_text(serialized)
