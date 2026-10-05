#!/usr/bin/env python3
"""Measure HTTP reads during sustained MemoryService writes.

The run seeds a corpus, measures a read-only control, then measures searches
while writers continuously submit /add/batch requests. Results include request
latencies, successful operations, HTTP errors (including writer-gate 429s),
and write records accepted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)], 3)


def post(url, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.status, response.read()


def summarize(name, started, ended, samples, statuses, records_per_success=0):
    elapsed = ended - started
    successes = sum(1 for status in statuses if 200 <= status < 300)
    rejected = sum(1 for status in statuses if status == 429)
    errors = len(statuses) - successes - rejected
    return {
        "operation": name,
        "elapsed_seconds": round(elapsed, 3),
        "requests": len(statuses),
        "successful_requests": successes,
        "rejected_429": rejected,
        "other_errors": errors,
        "requests_per_second": round(successes / elapsed, 2) if elapsed else 0,
        "records_accepted": successes * records_per_success,
        "latency_ms": {
            "p50": percentile(samples, 0.50),
            "p95": percentile(samples, 0.95),
            "p99": percentile(samples, 0.99),
            "mean": round(statistics.mean(samples), 3) if samples else None,
        },
    }


def run_phase(name, seconds, readers, writers, write_batch, bind, user_id):
    deadline = time.monotonic() + seconds
    gate = threading.Barrier(readers + writers + 1)
    lock = threading.Lock()
    samples = {"search": [], "write": []}
    statuses = {"search": [], "write": []}
    sequence = 0
    visible_candidates = []
    empty_searches = 0

    def record(kind, latency, status):
        with lock:
            samples[kind].append(latency)
            statuses[kind].append(status)

    def reader():
        nonlocal empty_searches
        gate.wait()
        request_id = 0
        while time.monotonic() < deadline:
            payload = {
                # Preserve the useful terms while avoiding repeated-query
                # preparation-cache hits in either phase.
                "query": f"deployment configuration system decision mixednonce{request_id}",
                "user_id": "bench-user",
                "top_k": 20,
            }
            start = time.monotonic()
            try:
                status, body = post(f"http://{bind}/search", payload)
                if not json.loads(body).get("data"):
                    with lock:
                        empty_searches += 1
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.read()
            except Exception:
                status = 0
            record("search", (time.monotonic() - start) * 1000, status)
            request_id += 1

    def writer(worker):
        nonlocal sequence
        gate.wait()
        while time.monotonic() < deadline:
            with lock:
                start_id = sequence
                sequence += write_batch
            requests = []
            for offset in range(write_batch):
                item = start_id + offset
                requests.append({
                    "request_id": f"mixed-{name}-{worker}-{item}",
                    "user_id": user_id,
                    "session_id": f"mixed-session-{worker}-{item}",
                    "messages": [{
                        "role": "user",
                        "content": f"Mixedloadunique{item} record: deployment configuration system decision",
                    }],
                })
            start = time.monotonic()
            try:
                status, _ = post(f"http://{bind}/add/batch", requests)
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.read()
            except Exception:
                status = 0
            record("write", (time.monotonic() - start) * 1000, status)
            if 200 <= status < 300:
                with lock:
                    visible_candidates.append(requests[-1]["messages"][0]["content"])

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=readers + writers) as pool:
        futures = [pool.submit(reader) for _ in range(readers)]
        futures.extend(pool.submit(writer, worker) for worker in range(writers))
        gate.wait()
        for future in futures:
            future.result()
    ended = time.monotonic()

    flush_started = time.monotonic()
    flush_status, _ = post(f"http://{bind}/v1/memories/refresh", {})
    flush_seconds = time.monotonic() - flush_started
    visibility = None
    if visible_candidates:
        content = visible_candidates[-1]
        marker_query = content.split()[0]
        try:
            status, body = post(
                f"http://{bind}/search",
                {"query": marker_query, "user_id": user_id, "top_k": 20},
            )
            result = json.loads(body)
            hits = result.get("data", [])
            visibility = {
                "checked": True,
                "http_status": status,
                "written_record_visible": any(
                    content in hit.get("content", "") for hit in hits
                ),
                "query": marker_query,
            }
        except Exception as exc:
            visibility = {"checked": True, "error": str(exc)}

    return {
        "phase": name,
        "reader_workers": readers,
        "empty_search_responses": empty_searches,
        "reader_user_id": "bench-user",
        "writer_user_id": user_id,
        "writer_workers": writers,
        "write_batch_records": write_batch,
        "search": summarize("search", started, ended, samples["search"], statuses["search"]),
        "write": summarize(
            "write", started, ended, samples["write"], statuses["write"], write_batch
        ),
        "flush_seconds": round(flush_seconds, 3),
        "flush_status": flush_status,
        "published_write_records_per_second": round(sum(200 <= s < 300 for s in statuses["write"]) * write_batch / (ended - started + flush_seconds), 2),
        "post_write_visibility": visibility,
    }


def run_once(args, repetition):
    temp_root = ROOT / ".benchmark-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"lint-ai-mixed-load-{repetition}-", dir=temp_root
    ) as index_tmp:
        port = args.port + repetition - 1
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
            ], check=True, cwd=ROOT)

            control = run_phase(
                "read-only-control", args.seconds, args.readers, 0,
                args.write_batch, bind, "mixed-bench-user",
            )
            time.sleep(1)
            mixed = run_phase(
                "concurrent-read-write", args.seconds, args.readers, args.writers,
                args.write_batch, bind, "mixed-bench-user",
            )
            return {
                "repetition": repetition,
                "records_before_load": args.records,
                "seed_sessions": min(args.sessions, (args.records + 1023) // 1024),
                "readers": args.readers,
                "writers": args.writers,
                "write_batch_records": args.write_batch,
                "configured_phase_seconds": args.seconds,
                "phases": [control, mixed],
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
    parser.add_argument("--seconds", type=int, default=15, help="duration of each phase")
    parser.add_argument("--readers", type=int, default=10)
    parser.add_argument("--writers", type=int, default=1)
    parser.add_argument("--write-batch", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.records, args.sessions, args.seconds, args.readers,
           args.write_batch, args.repetitions) < 1:
        parser.error("records, sessions, seconds, readers, write-batch, and repetitions must be positive")
    if args.writers < 0:
        parser.error("writers cannot be negative")
    if args.write_batch > 128:
        parser.error("write-batch cannot exceed /add/batch's 128-request limit")

    runs = [run_once(args, repetition) for repetition in range(1, args.repetitions + 1)]
    result = {
        "benchmark": "http-mixed-read-write-load",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_tree": "working tree",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "server": str(args.server_bin),
        "records_before_load": args.records,
        "readers": args.readers,
        "writers": args.writers,
        "write_batch_records": args.write_batch,
        "configured_phase_seconds": args.seconds,
        "repetitions": args.repetitions,
        "runs": runs,
        "notes": [
            "Each repetition uses a fresh server process and temporary index.",
            "Read-only control and mixed phases are paired on one server and corpus per repetition.",
            "Search queries vary by nonce to prevent repeated-query cache reuse in either phase.",
            "Write requests are /add/batch calls with distinct request IDs and session IDs.",
            "HTTP 429 responses are counted as rejected writes, not retried.",
            "Each mixed phase searches for a record from a successfully accepted write after load ends.",
        ],
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
