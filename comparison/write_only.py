#!/usr/bin/env python3
"""Measure sequential HTTP write throughput on a pre-seeded Lint-AI index."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent


def post(url, payload, timeout=300):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def percentile(values, p):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)], 3)


def add_requests(batch_size, sequence, session_count):
    requests = []
    for offset in range(batch_size):
        item = sequence + offset
        requests.append({
            "request_id": f"write-only-{item}",
            "user_id": "write-bench-user",
            "session_id": f"bench-session-{item % session_count}",
            "messages": [{
                "role": "user",
                "content": f"Writeonlyunique{item} benchmark record: persistent memory write throughput.",
            }],
        })
    return requests


def run_cell(args, repetition, batch_size):
    temp_root = ROOT / ".benchmark-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"lint-ai-write-only-{repetition}-{batch_size}-", dir=temp_root
    ) as index_tmp:
        port = args.port + (repetition - 1) * len(args.batch_sizes) + args.batch_sizes.index(batch_size)
        bind = f"127.0.0.1:{port}"
        server = subprocess.Popen(
            [str(args.server_bin), "--bind", bind, "--index", index_tmp], cwd=ROOT
        )
        try:
            health = f"http://{bind}/health"
            for _ in range(300):
                try:
                    with urllib.request.urlopen(health, timeout=1):
                        break
                except Exception:
                    if server.poll() is not None:
                        raise RuntimeError("server exited before becoming healthy")
                    time.sleep(0.1)
            else:
                raise RuntimeError("server did not become healthy")

            subprocess.run([
                "python3", str(ROOT / "comparison/seed_lint_ai.py"),
                "--url", f"http://{bind}/add", "--count", str(args.records),
                "--batch-size", "1024", "--sessions", str(args.sessions), "--bulk",
            ], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

            write_url = f"http://{bind}/add/batch" + ("?wait_for_visibility=true" if args.wait_for_visibility else "")
            sequence = (repetition - 1) * 10_000_000 + args.batch_sizes.index(batch_size) * 1_000_000
            for _ in range(args.warmup_requests):
                post(write_url, add_requests(batch_size, sequence, args.sessions))
                sequence += batch_size

            post(f"http://{bind}/v1/memories/refresh", {})
            samples = []
            statuses = []
            last_successful_sequence = None
            measurement_started = time.monotonic()
            deadline = measurement_started + args.seconds
            while time.monotonic() < deadline:
                payload = add_requests(batch_size, sequence, args.sessions)
                started = time.monotonic()
                try:
                    status, _ = post(write_url, payload)
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    exc.read()
                except Exception:
                    status = 0
                samples.append((time.monotonic() - started) * 1000)
                statuses.append(status)
                if 200 <= status < 300:
                    last_successful_sequence = sequence
                sequence += batch_size
            elapsed = time.monotonic() - measurement_started
            successes = sum(200 <= status < 300 for status in statuses)
            rejected = sum(status == 429 for status in statuses)
            other_errors = len(statuses) - successes - rejected

            flush_started = time.monotonic()
            flush_status, _ = post(f"http://{bind}/v1/memories/refresh", {})
            flush_seconds = time.monotonic() - flush_started
            visible_query = f"Writeonlyunique{last_successful_sequence}"
            visible = False
            try:
                status, body = post(
                    f"http://{bind}/search",
                    {"query": visible_query, "user_id": "write-bench-user", "top_k": 20},
                )
                result = json.loads(body)
                visible = status == 200 and any(
                    visible_query in hit.get("content", "") for hit in result.get("data", [])
                )
            except Exception:
                pass

            return {
                "repetition": repetition,
                "batch_size_add_requests": batch_size,
                "seed_records": args.records,
                "measured_seconds": round(elapsed, 3),
                "requests": len(statuses),
                "successful_requests": successes,
                "rejected_429": rejected,
                "other_errors": other_errors,
                "accepted_records": successes * batch_size,
                "request_throughput_per_s": round(successes / elapsed, 2) if elapsed else 0,
                "record_throughput_per_s": round(successes * batch_size / elapsed, 2) if elapsed else 0,
                "latency_ms": {
                    "p50": percentile(samples, 0.50),
                    "p95": percentile(samples, 0.95),
                    "p99": percentile(samples, 0.99),
                    "mean": round(statistics.mean(samples), 3),
                },
                "flush_seconds": round(flush_seconds, 3),
                "flush_status": flush_status,
                "published_record_throughput_per_s": round(successes * batch_size / (elapsed + flush_seconds), 2),
                "last_written_record_searchable": visible,
            }
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/server")
    parser.add_argument("--records", type=int, default=23366)
    parser.add_argument("--sessions", type=int, default=23)
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-for-visibility", action="store_true")
    args = parser.parse_args()
    if min(args.records, args.sessions, args.seconds, args.repetitions) < 1:
        parser.error("records, sessions, seconds, and repetitions must be positive")
    if args.warmup_requests < 0 or any(size < 1 or size > 128 for size in args.batch_sizes):
        parser.error("warm-up must be nonnegative and batch sizes must be in 1..=128")
    if not args.server_bin.is_file():
        parser.error(f"server binary not found: {args.server_bin}")

    runs = []
    for repetition in range(1, args.repetitions + 1):
        order = list(args.batch_sizes)
        random.Random(repetition).shuffle(order)
        for batch_size in order:
            result = run_cell(args, repetition, batch_size)
            runs.append(result)
            print(
                f"rep={repetition} batch={batch_size} accepted={result['accepted_records']} "
                f"records/{result['measured_seconds']}s "
                f"{result['record_throughput_per_s']} records/s "
                f"p50={result['latency_ms']['p50']}ms"
            , flush=True)

    summary = {}
    for batch_size in args.batch_sizes:
        cells = [r for r in runs if r["batch_size_add_requests"] == batch_size]
        summary[str(batch_size)] = {
            "request_throughput_median_per_s": round(statistics.median(r["request_throughput_per_s"] for r in cells), 2),
            "published_record_throughput_median_per_s": round(statistics.median(r["published_record_throughput_per_s"] for r in cells), 2),
            "record_throughput_median_per_s": round(statistics.median(r["record_throughput_per_s"] for r in cells), 2),
            "p50_latency_median_ms": round(statistics.median(r["latency_ms"]["p50"] for r in cells), 3),
            "p95_latency_median_ms": round(statistics.median(r["latency_ms"]["p95"] for r in cells), 3),
            "p99_latency_median_ms": round(statistics.median(r["latency_ms"]["p99"] for r in cells), 3),
        }
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = None
    try:
        tree = subprocess.check_output(["cargo", "tree", "-i", "tantivy"], cwd=ROOT, text=True)
        tantivy = next((line.strip().split()[1] for line in tree.splitlines() if line.strip().startswith("tantivy v")), None)
    except Exception:
        tantivy = None
    report = {
        "benchmark": "write-only-http-load",
        "write_acknowledgement": "published" if args.wait_for_visibility else "durable-staged",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_binary": str(args.server_bin.resolve()),
        "server_binary_sha256": hashlib.sha256(args.server_bin.read_bytes()).hexdigest(),
        "git_revision": revision,
        "tantivy_version": tantivy,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "records_before_each_cell": args.records,
        "sessions": args.sessions,
        "writer_workers": 1,
        "readers": 0,
        "measured_seconds_per_cell": args.seconds,
        "warmup_requests_per_cell": args.warmup_requests,
        "batch_sizes_add_requests": args.batch_sizes,
        "repetitions": args.repetitions,
        "summary": summary,
        "runs": runs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
