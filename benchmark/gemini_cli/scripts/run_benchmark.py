#!/usr/bin/env python3
"""Run a reproducible local Gemini CLI/MCP integration smoke benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def run_server(binary: Path, root: Path, requests: list[dict]) -> tuple[list[dict], float]:
    payload = "".join(json.dumps(item) + "\n" for item in requests)
    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary), "--gemini-cli-serve", str(root)],
        cwd=root,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Gemini MCP process failed")
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()], elapsed_ms


def run_hook(binary: Path, root: Path, event: str, prompt: str) -> float:
    payload = {
        "session_id": "benchmark-gemini-session",
        "cwd": str(root),
        "hook_event_name": event,
        "prompt": prompt,
    }
    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary), "--gemini-cli-hook", event.replace("BeforeAgent", "before-agent").replace("SessionStart", "session-start")],
        cwd=root,
        input=json.dumps(payload) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"{event} hook failed")
    json.loads(completed.stdout)
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_dir = (args.results_dir or repo_root / "benchmark" / "gemini_cli" / "results" / "local-smoke").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    binary = repo_root / "target" / "debug" / "lint-ai"
    # macOS commonly exposes /var as a symlink to /private/var. Lint-AI
    # intentionally rejects symlinked parent paths, so keep the benchmark
    # workspace below the canonical repository root.
    temp_root = Path(tempfile.mkdtemp(prefix=".lint-ai-gemini-benchmark.", dir=repo_root))
    project = temp_root / "project"
    project.mkdir()
    try:
        subprocess.run(["cargo", "build", "--quiet", "--features", "gemini-cli", "--bin", "lint-ai"], cwd=repo_root, check=True)
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "enable_lint_ai", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "record_session", "arguments": {"action": "start"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "info", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "benchmark memory", "top_k": 3}}},
        ]
        responses, total_ms = run_server(binary, project, requests)
        hook_times = {
            "SessionStart": run_hook(binary, project, "SessionStart", "start benchmark session"),
            "BeforeAgent": run_hook(binary, project, "BeforeAgent", "retrieve benchmark memory"),
        }
        events_path = project / ".lint-ai" / "gemini-cli-sessions" / "benchmark-gemini-session" / "events.jsonl"
        report = {
            "schema_version": 1,
            "provider": "gemini-cli",
            "benchmark": "local-mcp-hooks-smoke",
            "gemini_version": subprocess.run(["gemini", "--version"], capture_output=True, text=True, check=False).stdout.strip(),
            "lint_ai_version": "local cargo build",
            "mcp_requests": len(requests),
            "mcp_responses": len(responses),
            "mcp_roundtrip_ms": round(total_ms, 3),
            "hook_latency_ms": {key: round(value, 3) for key, value in hook_times.items()},
            "recording_event_count": sum(1 for line in events_path.read_text().splitlines() if line.strip()) if events_path.exists() else 0,
            "token_usage": None,
            "notes": "Local protocol smoke benchmark; no authenticated Gemini model turn was run.",
        }
        (results_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
