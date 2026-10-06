#!/usr/bin/env python3
"""Benchmark the current bekind --serve JSONL request path.

Measures warm request throughput and latency at caller concurrency levels,
with one or more independent bekind daemon processes. One process mirrors
Lint-AI's current process-wide JsonLinesDaemon; larger pools show how a
process pool would scale. Each daemon is driven through the same one-line-in,
one-line-out protocol and each connection is serialized, as in Lint-AI.

Example:
    python3 scripts/bench_bekind.py --bekind-bin /path/to/bekind \
        --requests 1000 --concurrency 1,2,4,8 --pool-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_TEXTS = (
    "Which city have both Jean and John visited?",
    "What is Alice's usual weekend exercise routine?",
    "Priya enjoys cooking with cilantro and basil on weekdays.",
    "When does Morgan usually go hiking with friends?",
)


class BekindClient:
    """One bekind --serve child with serialized JSONL request/response I/O."""

    def __init__(self, binary: Path, timeout: float) -> None:
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [str(binary), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise RuntimeError("bekind did not expose stdin/stdout pipes")
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_responses, daemon=True)
        self.reader.start()

    def _read_responses(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def request(self, request_id: int, text: str) -> float:
        payload = json.dumps(
            {"texts": [{"id": str(request_id), "text": text, "with_scope": True}]},
            separators=(",", ":"),
        )
        started = time.perf_counter()
        deadline = started + self.timeout
        if not self.lock.acquire(timeout=self.timeout):
            raise TimeoutError("timed out waiting for this daemon's request slot")
        try:
            if self.process.poll() is not None:
                raise RuntimeError(f"bekind exited with status {self.process.returncode}")
            assert self.process.stdin is not None
            self.process.stdin.write(payload + "\n")
            self.process.stdin.flush()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.close()
                raise TimeoutError("bekind request exceeded the per-call timeout")
            try:
                response_line = self.responses.get(timeout=remaining)
            except queue.Empty as exc:
                # Match Lint-AI: discard a timed-out child so a late reply
                # cannot be mistaken for the next request's response.
                self.close()
                raise TimeoutError("bekind did not reply before the per-call timeout") from exc
            if response_line is None:
                raise RuntimeError("bekind closed stdout before replying")
        finally:
            self.lock.release()
        elapsed = time.perf_counter() - started
        response: dict[str, Any] = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"bekind returned an error: {response['error']}")
        results = response.get("text_results")
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError(f"unexpected bekind response: {response_line.strip()}")
        if results[0].get("id") != str(request_id):
            raise RuntimeError(f"bekind response ID mismatch: {response_line.strip()}")
        return elapsed

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def percentile(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


def run_cell(
    clients: list[BekindClient], concurrency: int, requests: int, texts: tuple[str, ...],
) -> dict[str, Any]:
    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()

    def one(index: int) -> tuple[float | None, str | None]:
        client = clients[index % len(clients)]
        try:
            return client.request(index, texts[index % len(texts)]), None
        except Exception as exc:  # report failed calls instead of silently dropping them
            return None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency, error in pool.map(one, range(requests)):
            if latency is not None:
                latencies.append(latency)
            if error is not None:
                errors.append(error)
    elapsed = time.perf_counter() - started
    if not latencies:
        raise RuntimeError(f"all bekind requests failed: {errors[:3]}")
    return {
        "callers": concurrency,
        "requests": requests,
        "completed": len(latencies),
        "errors": len(errors),
        "elapsed_s": round(elapsed, 4),
        "throughput_req_s": round(len(latencies) / elapsed, 2),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50) * 1000, 3),
            "p95": round(percentile(latencies, 0.95) * 1000, 3),
            "p99": round(percentile(latencies, 0.99) * 1000, 3),
            "mean": round(statistics.mean(latencies) * 1000, 3),
        },
        "sample_errors": errors[:3],
    }


def csv_positive_ints(value: str, option: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError(f"{option} values must be positive")
    return sorted(set(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bekind-bin",
        type=Path,
        help="bekind executable; defaults to the bekind found on PATH",
    )
    parser.add_argument("--requests", type=int, default=1000, help="measured requests per cell")
    parser.add_argument("--warmup", type=int, default=20, help="warmup requests per daemon")
    parser.add_argument("--concurrency", default="1,2,4,8", help="caller counts, comma-separated")
    parser.add_argument("--pool-sizes", default="1,2,4", help="daemon counts, comma-separated")
    parser.add_argument(
        "--text", action="append", dest="texts",
        help="query text; repeat to provide a representative workload",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="per-call timeout in seconds")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("--requests and --timeout must be positive; --warmup cannot be negative")
    try:
        callers = csv_positive_ints(args.concurrency, "--concurrency")
        pool_sizes = csv_positive_ints(args.pool_sizes, "--pool-sizes")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    binary = args.bekind_bin
    if binary is None:
        found = shutil.which("bekind")
        if found is None:
            parser.error("bekind was not found on PATH; pass --bekind-bin /path/to/bekind")
        binary = Path(found)
    binary = binary.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"bekind executable does not exist: {binary}")

    texts = tuple(args.texts or DEFAULT_TEXTS)
    report: dict[str, Any] = {
        "benchmark": "bekind-jsonl-daemon-throughput",
        "binary": str(binary),
        "request_shape": {"texts_per_request": 1, "with_scope": True},
        "requests_per_cell": args.requests,
        "warmup_requests_per_daemon": args.warmup,
        "results": [],
        "note": (
            "One daemon mirrors Lint-AI's serialized JsonLinesDaemon. Larger pools are a scaling "
            "comparison, not the current Lint-AI architecture."
        ),
    }

    for pool_size in pool_sizes:
        clients: list[BekindClient] = []
        try:
            clients = [BekindClient(binary, args.timeout) for _ in range(pool_size)]
            request_id = -1
            for client in clients:
                for warm_index in range(args.warmup):
                    client.request(request_id, texts[warm_index % len(texts)])
                    request_id -= 1
            for concurrency in callers:
                result = run_cell(clients, concurrency, args.requests, texts)
                result["daemon_pool_size"] = pool_size
                report["results"].append(result)
        finally:
            for client in clients:
                client.close()

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
