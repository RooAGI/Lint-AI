#!/usr/bin/env python3
"""Compare the resident MCP route over HTTP and local IPC.

The script uses only the Python standard library. It starts an isolated server
for every repetition, seeds the same Markdown corpus, and measures four search
clients plus one memory writer. Connection setup is included in every request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import platform
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


ROUTE = "/provider-memory/mcp/tools/call"
QUERY = "relay valve amber quartz sequence"
DOC_TEXT = (
    "The relay valve amber quartz sequence controls the isolated benchmark "
    "workspace. The operator documents the startup order, recovery steps, "
    "and verified state before changing the relay."
)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float = 5.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def socket_path_for(root: Path) -> Path:
    identity = os.environ.get("USER", "")
    digest = hashlib.sha256(identity.encode() + b"\0" + os.fsencode(root)).hexdigest()
    home = Path.home()
    return home / ".membrick" / "runtime" / f"mb-{digest[:24]}.sock"


def request(
    transport: str,
    endpoint: str | Path,
    root: Path,
    tool: str,
    arguments: dict[str, Any],
    session_id: str | None = None,
) -> tuple[int, dict[str, Any], float]:
    body_obj: dict[str, Any] = {
        "workspace": str(root),
        "provider": "codex",
        "user_id": "mcp",
        "tool": tool,
        "arguments": arguments,
    }
    if session_id is not None:
        body_obj["session_id"] = session_id
    body = json.dumps(body_obj, separators=(",", ":")).encode()
    start = time.perf_counter()
    if transport == "ipc":
        connection: http.client.HTTPConnection = UnixHTTPConnection(Path(endpoint))
    else:
        host, port = str(endpoint).rsplit(":", 1)
        connection = http.client.HTTPConnection(host, int(port), timeout=5.0)
    try:
        connection.request(
            "POST",
            ROUTE,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        raw = response.read()
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw[:500].decode(errors="replace")}
        return response.status, payload, elapsed_ms
    finally:
        connection.close()


def wait_ready(
    transport: str,
    endpoint: str | Path,
    root: Path,
    timeout_s: float = 60.0,
) -> float:
    start = time.perf_counter()
    deadline = start + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            status, _, _ = request(transport, endpoint, root, "info", {})
            if status == 200:
                return (time.perf_counter() - start) * 1000
        except Exception as error:  # startup race; report the final cause on timeout
            last_error = error
        time.sleep(0.05)
    raise TimeoutError(f"{transport} server did not become ready: {last_error}")


def start_server(
    binary: Path,
    transport: str,
    root: Path,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], str | Path, float]:
    command = [
        str(binary),
        "--project-root",
        str(root),
        "--allow-unauthenticated",
        "--refresh-interval-ms",
        "250",
        "--checkpoint-interval-seconds",
        "30",
    ]
    if transport == "ipc":
        command.append("--resident-server")
        endpoint: str | Path = socket_path_for(root)
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command.extend(["--bind", f"127.0.0.1:{port}"])
        endpoint = f"127.0.0.1:{port}"
    log = log_path.open("wb")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        startup_ms = wait_ready(transport, endpoint, root)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        log.close()
        raise
    log.close()
    return process, endpoint, startup_ms


def summarize(samples: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "mean_ms": statistics.fmean(samples) if samples else None,
    }


def run_repetition(
    binary: Path,
    transport: str,
    run_number: int,
    seconds: float,
    readers: int,
    root_base: Path,
) -> dict[str, Any]:
    root = (root_base / f"{transport}-{run_number}").resolve()
    docs = root / "benchmark-docs"
    docs.mkdir(parents=True)
    for index in range(100):
        (docs / f"note-{index:03}.md").write_text(
            f"# Benchmark note {index}\n\n{DOC_TEXT} Record {index}.\n",
            encoding="utf-8",
        )

    process, endpoint, startup_ms = start_server(
        binary,
        transport,
        root,
        root_base / f"{transport}-{run_number}.log",
    )
    stop = threading.Event()
    result_lock = threading.Lock()
    search_latencies: list[float] = []
    write_latencies: list[float] = []
    errors: list[str] = []
    last_write: str | None = None

    def search_worker() -> None:
        while not stop.is_set():
            try:
                status, payload, elapsed = request(
                    transport,
                    endpoint,
                    root,
                    "search",
                    {"query": QUERY, "top_k": 10},
                )
                if status != 200 or not isinstance(payload.get("results"), list):
                    raise RuntimeError(f"HTTP {status}: {payload}")
                with result_lock:
                    search_latencies.append(elapsed)
            except Exception as error:
                with result_lock:
                    errors.append(f"search: {error}")

    def writer_worker() -> None:
        nonlocal last_write
        index = 0
        while not stop.is_set():
            index += 1
            content = f"Mixed load writer added memory amber relay marker write-{index}."
            session = f"ipc-http-bench-{run_number}"
            try:
                status, payload, elapsed = request(
                    transport,
                    endpoint,
                    root,
                    "add_memory",
                    {
                        "content": content,
                        "request_id": f"{transport}-{run_number}-{index}",
                        "session_id": session,
                        "role": "user",
                    },
                    session_id=session,
                )
                if status != 200 or "error" in payload:
                    raise RuntimeError(f"HTTP {status}: {payload}")
                with result_lock:
                    write_latencies.append(elapsed)
                    last_write = f"write-{index}"
            except Exception as error:
                with result_lock:
                    errors.append(f"write: {error}")
            stop.wait(0.5)

    workers = [threading.Thread(target=search_worker) for _ in range(readers)]
    workers.append(threading.Thread(target=writer_worker))
    try:
        for worker in workers:
            worker.start()
        time.sleep(seconds)
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10)

    visible = False
    if last_write is not None:
        status, payload, _ = request(
            transport,
            endpoint,
            root,
            "search",
            {"query": last_write, "top_k": 10},
        )
        visible = status == 200 and any(
            last_write in str(item) for item in payload.get("results", [])
        )

    with result_lock:
        searches = list(search_latencies)
        writes = list(write_latencies)
        failures = list(errors)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    return {
        "transport": transport,
        "repetition": run_number,
        "duration_s": seconds,
        "startup_to_ready_ms": startup_ms,
        "readers": readers,
        "search_requests": len(searches),
        "search_requests_per_s": len(searches) / seconds,
        "search_latency": summarize(searches),
        "write_requests": len(writes),
        "write_requests_per_s": len(writes) / seconds,
        "write_latency": summarize(writes),
        "errors": failures,
        "last_acknowledged_write": last_write,
        "last_write_visible_after_ack": visible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=Path("target/release/server"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.server_bin.resolve()
    if not binary.is_file():
        raise SystemExit(f"server binary not found: {binary}")
    if args.runs < 1 or args.seconds <= 0 or args.readers < 1:
        raise SystemExit("runs, seconds, and readers must be positive")

    with tempfile.TemporaryDirectory(prefix="membrick-ipc-http-bench-") as temp:
        root = Path(temp)
        results = []
        for repetition in range(1, args.runs + 1):
            # Alternate order to reduce ordering effects from thermals or load.
            order = ("http", "ipc") if repetition % 2 else ("ipc", "http")
            for transport in order:
                results.append(
                    run_repetition(
                        binary,
                        transport,
                        repetition,
                        args.seconds,
                        args.readers,
                        root,
                    )
                )

    report = {
        "benchmark": "matched-mcp-http-vs-ipc-read-write",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "configuration": {
            "iterations_per_transport": args.runs,
            "seconds_per_run": args.seconds,
            "search_readers": args.readers,
            "writer_interval_s": 0.5,
            "connections": "fresh connection per request",
            "search_query": QUERY,
            "seeded_markdown_documents": 100,
        },
        "runs": results,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
