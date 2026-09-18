#!/usr/bin/env python3
"""Run a reproducible HTTP throughput comparison for index layouts.

Examples:
  python3 comparison/throughput.py --mode single
  python3 comparison/throughput.py --mode global
  python3 comparison/throughput.py --mode segment --no-cache
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("single", "global", "segment"), required=True)
parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/server")
parser.add_argument("--records", type=int, default=23366)
parser.add_argument("--requests", type=int, default=100)
parser.add_argument("--warmup-requests", type=int, default=0)
parser.add_argument("--port", type=int, default=18080)
parser.add_argument("--no-cache", action="store_true")
parser.add_argument("--keep-index", action="store_true")
args = parser.parse_args()

index_dir = Path(tempfile.mkdtemp(prefix=f"lint-ai-throughput-{args.mode}."))
bind = f"127.0.0.1:{args.port}"
command = [str(args.server_bin), "--bind", bind, "--index", str(index_dir)]
if args.mode == "single":
    command.append("--single-index")
elif args.mode == "global":
    command.append("--global-index")
environment = os.environ.copy()
if args.no_cache:
    environment["LINT_AI_DISABLE_QUERY_CACHE"] = "1"

server = subprocess.Popen(command, cwd=ROOT, env=environment)
try:
    health = f"http://{bind}/health"
    for _ in range(100):
        try:
            with urllib.request.urlopen(health, timeout=1):
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("server did not become healthy")

    subprocess.run(
        [
            "python3",
            str(ROOT / "comparison/seed_lint_ai.py"),
            "--url",
            f"http://{bind}/add",
            "--count",
            str(args.records),
            "--batch-size",
            "1024",
            "--bulk",
        ],
        check=True,
        cwd=ROOT,
    )
    payload = json.dumps(
        {
            "query": "deployment configuration system decision",
            "user_id": "bench-user",
            "top_k": 20,
        }
    )
    measured = subprocess.run(
        [
            "python3",
            str(ROOT / "comparison/http_latency.py"),
            "--url",
            f"http://{bind}/search",
            "--payload",
            payload,
            "--requests",
            str(args.requests),
            "--warmup-requests",
            str(args.warmup_requests),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    result = {
        "mode": args.mode,
        "records": args.records,
        "requests_per_cell": args.requests,
        "warmup_requests_per_cell": args.warmup_requests,
        "cache": "disabled" if args.no_cache else "enabled",
        "server": str(args.server_bin),
        "measurements": [json.loads(line) for line in measured.stdout.splitlines()],
    }
    print(json.dumps(result, indent=2))
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
    if args.keep_index:
        print(f"index_dir={index_dir}")
