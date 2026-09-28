#!/usr/bin/env python3
"""Benchmark the behood socket singleton: cold spawn vs warm cross-process.

Simulates what a short-lived MCP/hook process pays per query:
  1. cold: no daemon -> spawn (double-fork) -> handshake -> query
  2. warm: fresh process -> connect to live daemon -> handshake -> query
  3. one-shot baseline: python3 behood_query.py "<question>" (cold interpreter)

NOTE: spaCy/bekind are not installed in this environment, so the daemon
answers [] via fail-open. These numbers measure IPC + spawn overhead, NOT
model inference. The real warm-vs-cold delta (25ms vs 3.4-5.0s) was measured
with a real bekind release in the superseded 7dba8ce prototype.
"""
import json
import os
import socket
import subprocess
import sys
import time

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behood_query.py")
SOCK = "/tmp/bq-bench.sock"
QUESTION = "Which city have both Jean and John visited?"


def query_via_new_process(sock_path, question):
    """One query from a FRESH python process: connect, handshake, ask."""
    code = (
        "import json, socket, sys;"
        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM);"
        "s.settimeout(30);"
        f"s.connect({sock_path!r});"
        "f = s.makefile('rb');"
        "hs = json.loads(f.readline());"
        "assert hs.get('ready') is True, hs;"
        f"s.sendall(json.dumps({{'question': {question!r}}}).encode() + b'\\n');"
        "resp = json.loads(f.readline());"
        "print('entities:', resp.get('entities'));"
        "s.close()"
    )
    subprocess.run([sys.executable, "-c", code], check=True,
                   capture_output=True, timeout=60)


def main():
    if os.path.exists(SOCK):
        os.unlink(SOCK)

    # 1. Cold: spawn the daemon, then query from a fresh process.
    t0 = time.monotonic()
    subprocess.run(
        [sys.executable, SCRIPT, "--serve-socket", SOCK, "--idle-timeout", "60"],
        check=True, capture_output=True, timeout=60,
    )
    # Wait for the daemon to be ready (poll the handshake).
    ready = False
    for _ in range(100):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(SOCK)
            hs = json.loads(s.makefile("rb").readline())
            s.close()
            if hs.get("ready") is True:
                ready = True
                break
        except OSError:
            time.sleep(0.1)
    assert ready, "daemon never became ready"
    t_spawn_ready = time.monotonic() - t0

    t0 = time.monotonic()
    query_via_new_process(SOCK, QUESTION)
    t_cold_query = time.monotonic() - t0

    # 2. Warm: daemon already up; fresh process connects and asks.
    t0 = time.monotonic()
    query_via_new_process(SOCK, QUESTION)
    t_warm_query = time.monotonic() - t0

    t0 = time.monotonic()
    query_via_new_process(SOCK, QUESTION)
    t_warm_query2 = time.monotonic() - t0

    # 3. One-shot baseline (cold interpreter, no daemon).
    t0 = time.monotonic()
    subprocess.run([sys.executable, SCRIPT, QUESTION],
                   capture_output=True, timeout=120)
    t_oneshot = time.monotonic() - t0

    # 4. Pure IPC: one persistent process, fresh connection per query
    # (what the Rust client pays — no interpreter startup).
    import time as _t
    ipc_times = []
    for _ in range(5):
        t0 = _t.monotonic()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(SOCK)
        f = s.makefile("rb")
        json.loads(f.readline())  # handshake
        s.sendall(json.dumps({"question": QUESTION}).encode() + b"\n")
        json.loads(f.readline())  # response
        s.close()
        ipc_times.append((_t.monotonic() - t0) * 1000)
    t_ipc = min(ipc_times)

    print(f"daemon spawn + ready (double-fork, no spaCy here): {t_spawn_ready*1000:.0f} ms")
    print(f"first query from fresh process (includes spawn above): {t_cold_query*1000:.0f} ms")
    print(f"warm query from fresh process (incl. python startup): {t_warm_query*1000:.0f} ms")
    print(f"warm query from fresh process (repeat):               {t_warm_query2*1000:.0f} ms")
    print(f"pure IPC per query (fresh conn, no interp startup):   {t_ipc:.1f} ms")
    print(f"one-shot baseline (cold interpreter):                 {t_oneshot*1000:.0f} ms")

    # Cleanup: ask the daemon to go away by removing the socket is not
    # enough (it only exits on idle); the 60s idle timeout handles it.
    # For a clean bench environment, wait briefly then check no hang.
    print("done (daemon exits on its 60s idle timeout)")


if __name__ == "__main__":
    main()
