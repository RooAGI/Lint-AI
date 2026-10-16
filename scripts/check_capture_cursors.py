#!/usr/bin/env python3
"""Read-only alignment and catch-up checks for Claude/Codex transcript cursors.

Discovers the resident's durable capture requests by default. This checks cursor
positions against transcript bytes, not document persistence or search quality.
Uses only the Python standard library and never starts a resident or a hook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

PROVIDERS = ("claude", "codex")
BLOCK_BYTES = 64 * 1024


def cursor_key(session_id: str) -> str:
    """Match src/ids.rs::stable_doc_id_from_source (including its separator)."""
    canonical = " ".join(session_id.split()).lower().encode("utf-8")
    value = 0xCBF29CE484222325
    for byte in canonical + b"\xff":
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def discover(project: Path, provider: str | None, session_id: str | None,
             transcript: Path | None) -> list[dict[str, Any]]:
    if transcript is not None:
        return [{"provider": provider, "session_id": session_id,
                 "transcript": str(transcript.resolve())}]
    jobs = []
    directory = project / ".membrick/runtime/transcript-captures"
    for path in sorted(directory.glob("*.json")):
        try:
            record = read_object(path)
            request = record["request"]
            name, sid = request["provider"], request["input"]["session_id"]
            if (provider and name != provider) or (session_id and sid != session_id):
                continue
            if name not in PROVIDERS or not isinstance(sid, str) or not sid.strip():
                raise ValueError("invalid provider or session_id")
            expected = f"{name}-{cursor_key(sid)}.json"
            if path.name != expected:
                raise ValueError(f"capture request filename should be {expected}")
            transcript_path = Path(request["input"]["transcript_path"])
            if not transcript_path.is_absolute():
                raise ValueError("capture transcript path must be absolute")
            jobs.append({"provider": name, "session_id": sid,
                         "transcript": str(transcript_path), "job_path": str(path),
                         "target_bytes": record["target_bytes"],
                         "worker_last_error": record.get("last_error")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            jobs.append({"job_path": str(path), "discovery_error": str(error)})
    return jobs


def complete_end(stream, length: int) -> int:
    """Find the last newline without loading a potentially huge final line."""
    position = length
    while position:
        start = max(0, position - BLOCK_BYTES)
        stream.seek(start)
        block = stream.read(position - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0


def check(project: Path, job: dict[str, Any], validate_json: bool = False) -> dict[str, Any]:
    report = dict(job)
    report.update(aligned=False, caught_up=False, errors=[], warnings=[])
    errors = report["errors"]
    if "discovery_error" in job:
        errors.append(job["discovery_error"])
        report.update(ok=False, state="invalid")
        return report
    cursor = project / f".membrick/{job['provider']}-state/capture-cursors/{cursor_key(job['session_id'])}.json"
    report["cursor_path"] = str(cursor)
    try:
        with cursor.open(encoding="utf-8") as source:
            value = json.load(source)
            progress_time = os.fstat(source.fileno()).st_mtime
        offset = value.get("offset") if isinstance(value, dict) else None
        if type(offset) is not int or offset < 0:
            raise ValueError("cursor offset must be a nonnegative integer")
        report["cursor_bytes"] = offset
        report["last_progress_age_seconds"] = round(max(0, time.time() - progress_time), 3)
        with Path(job["transcript"]).open("rb") as stream:
            stat = os.fstat(stream.fileno())
            length = stat.st_size
            end = complete_end(stream, length)
            # Cursor publication may race an append; refresh a short snapshot
            # before calling an otherwise legitimate new position out of range.
            if offset > length:
                stat = os.fstat(stream.fileno())
                length = stat.st_size
                end = complete_end(stream, length)
            report.update(transcript_bytes=length, complete_bytes=end,
                          trailing_partial_bytes=length - end,
                          file_identity=[stat.st_dev, stat.st_ino],
                          remaining_bytes=max(0, length - offset),
                          uncaptured_complete_bytes=max(0, end - offset))
            if offset > length:
                errors.append(f"cursor {offset} is beyond transcript EOF {length}")
            elif offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    errors.append(f"cursor {offset} is inside a JSONL line, not after a newline")
            anchor = value.get("transcript_anchor")
            if anchor is not None and offset <= length:
                stream.seek(0)
                prefix = stream.read(min(offset, 64))
                stream.seek(max(0, offset - 64))
                boundary = stream.read(min(offset, 64))
                actual = {"prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                          "boundary_sha256": hashlib.sha256(boundary).hexdigest()}
                if os.name == "posix":
                    actual.update(device=stat.st_dev, inode=stat.st_ino)
                if anchor != actual:
                    errors.append("transcript identity or checkpoint bytes changed")
                report["anchor_verified"] = anchor == actual
            elif anchor is None:
                report["warnings"].append("legacy cursor has no transcript fingerprint; only offset alignment can be checked")
            report["aligned"] = not errors
            report["caught_up"] = report["aligned"] and offset == end
            report["captured_percent"] = round(100 * min(offset, length) / length, 3) if length else (100.0 if offset == 0 else 0.0)
            if validate_json:
                stream.seek(0)
                position, count = 0, 0
                while position < end:
                    line = stream.readline(end - position)
                    try:
                        json.loads(line)
                    except (ValueError, UnicodeError) as error:
                        errors.append(f"invalid complete JSONL record at byte {position}: {error}")
                        break
                    count += 1
                    position += len(line)
                report["json_records_validated"] = count
        if job.get("worker_last_error"):
            report["warnings"].append(f"worker reports: {job['worker_last_error']}")
        report["state"] = "invalid" if errors else (
            "waiting-for-newline" if report["caught_up"] and report["trailing_partial_bytes"] else
            "caught-up" if report["caught_up"] else "pending")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
        report["state"] = "invalid"
    report["ok"] = not errors
    return report


def check_progress(previous: dict[str, Any], current: dict[str, Any]) -> None:
    if "cursor_bytes" not in previous or "cursor_bytes" not in current:
        return
    current["advanced_bytes"] = current["cursor_bytes"] - previous["cursor_bytes"]
    changed = current.get("file_identity") != previous.get("file_identity")
    shrunk = current.get("transcript_bytes", 0) < previous.get("transcript_bytes", 0)
    if changed or shrunk:
        current["warnings"].append("transcript was replaced or shrank between samples")
    elif current["advanced_bytes"] < 0:
        current["errors"].append("cursor regressed while the transcript neither shrank nor changed identity")
        current["ok"] = False
        current["state"] = "invalid"


def display(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    for entry in report["captures"]:
        label = f"{entry.get('provider', '?')}/{entry.get('session_id', entry.get('job_path', '?'))}"
        if "cursor_bytes" in entry and "transcript_bytes" in entry:
            print(f"{label}: {entry['state']} | cursor {entry['cursor_bytes']:,} / {entry['transcript_bytes']:,} bytes "
                  f"({entry['captured_percent']:.1f}%) | complete backlog {entry['uncaptured_complete_bytes']:,} bytes "
                  f"| partial tail {entry['trailing_partial_bytes']:,} bytes", flush=True)
        for message in entry["errors"]:
            print(f"{label}: ERROR: {message}", flush=True)
        for message in entry["warnings"]:
            print(f"{label}: WARNING: {message}", flush=True)
    for message in report["errors"]:
        print(f"ERROR: {message}", flush=True)
    if report.get("wait_finished"):
        print(f"Wait result: {report['wait_finished']}", flush=True)


def run(args) -> int:
    project = args.project.resolve()
    previous, goals, progress_at = {}, {}, {}
    started = time.monotonic()
    while True:
        now = time.monotonic()
        jobs = discover(project, args.provider, args.session_id, args.transcript)
        captures = [check(project, job, args.validate_json) for job in jobs]
        report = {"project": str(project), "elapsed_seconds": round(now - started, 3),
                  "captures": captures, "errors": []}
        if not captures:
            report["errors"].append("No matching capture requests found. Supply --provider, --session-id and --transcript for a session without a durable job.")
        for entry in captures:
            key = entry.get("cursor_path", entry.get("job_path"))
            if key in previous:
                check_progress(previous[key], entry)
            if "complete_bytes" in entry:
                goals.setdefault(key, entry["complete_bytes"])
                progress_at.setdefault(key, now)
                if entry.get("advanced_bytes", 0) > 0:
                    progress_at[key] = now
                entry["goal_bytes"] = goals[key]
                entry["goal_reached"] = entry.get("aligned", False) and entry["cursor_bytes"] >= goals[key]
                if args.wait_seconds and args.stall_seconds and not entry["goal_reached"] and now - progress_at[key] >= args.stall_seconds:
                    entry["errors"].append(f"no cursor progress for {args.stall_seconds:g} seconds")
                    entry["ok"] = False
                    entry["state"] = "stalled"
            previous[key] = entry
        if args.resident_url:
            try:
                with urllib.request.urlopen(args.resident_url.rstrip("/") + "/api/captures", timeout=5) as response:
                    report["resident"] = json.load(response)
            except (OSError, ValueError) as error:
                report["errors"].append(f"resident status unavailable: {error}")
        invalid = bool(report["errors"]) or any(not c.get("ok", False) for c in captures)
        reached = bool(captures) and all(c.get("goal_reached", False) for c in captures)
        if invalid:
            result = 1
        elif args.wait_seconds and reached:
            report["wait_finished"] = "initial complete transcript bytes captured"
            result = 0
        elif args.wait_seconds and time.monotonic() - started >= args.wait_seconds:
            report["wait_finished"] = "timed out before initial complete transcript bytes were captured"
            result = 2
        elif not args.wait_seconds:
            result = 2 if args.require_caught_up and any(not c["caught_up"] for c in captures) else 0
        else:
            result = None
        report["ok"] = not invalid
        report["all_caught_up"] = bool(captures) and all(c.get("caught_up", False) for c in captures)
        report["exit_code"] = result
        display(report, args.json)
        if result is not None:
            return result
        time.sleep(min(args.interval_seconds, max(0, args.wait_seconds - (time.monotonic() - started))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--session-id")
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit one JSON report per sample")
    parser.add_argument("--validate-json", action="store_true", help="Validate every complete JSONL record; scans the whole file per sample")
    parser.add_argument("--require-caught-up", action="store_true", help="Exit 2 when complete transcript bytes remain uncaptured")
    parser.add_argument("--wait-seconds", type=float, default=0, help="Wait for the initial complete-byte watermark, without chasing subsequent appends")
    parser.add_argument("--interval-seconds", type=float, default=2)
    parser.add_argument("--stall-seconds", type=float, default=0, help="While waiting, fail after this many seconds without cursor progress; 0 disables")
    parser.add_argument("--resident-url", help="Optionally also read MemoryService capture status, for example http://127.0.0.1:8465")
    args = parser.parse_args(argv)
    if args.transcript is not None and not (args.provider and args.session_id):
        parser.error("--transcript requires --provider and --session-id")
    if args.session_id is not None and not args.session_id.strip():
        parser.error("--session-id must not be empty")
    if (not all(math.isfinite(v) for v in (args.wait_seconds, args.stall_seconds, args.interval_seconds))
            or args.wait_seconds < 0 or args.stall_seconds < 0 or args.interval_seconds <= 0):
        parser.error("wait/stall seconds must be nonnegative and interval must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
