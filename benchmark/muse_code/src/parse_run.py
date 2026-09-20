#!/usr/bin/env python3
"""Extract privacy-preserving performance metrics from a Muse Code run.

`muse exec --json` emits JSONL session events on stdout. The model's final
answer is the `run.terminal.completed` record's `payload.text`; scoring must
use that text only, never the raw transcript, so tool-result content cannot
contaminate fact matching.

Two honest limitations:

- Token usage is not reported in the exec JSONL stream (verified against the
  echo provider). Token fields are left empty so the shared canonical
  contract records them as null instead of guessing.
- Muse documents no tool-call payload type for the exec stream. Tool calls
  are counted defensively from task-lifecycle records that look like tool
  executions, deduped by task id. Treat `tool_calls` as best-effort until
  validated against a real tool-calling run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TERMINAL_PAYLOAD_TYPE = "run.terminal.completed"


def _iter_records(stdout_text: str) -> Any:
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def extract_muse_last_message(stdout_text: str) -> str | None:
    """Return the final assistant answer from Muse's exec JSONL output."""
    last_text: str | None = None
    for record in _iter_records(stdout_text):
        if record.get("payload_type") != TERMINAL_PAYLOAD_TYPE:
            continue
        payload = record.get("payload")
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(text, str) and text:
            last_text = text
    return last_text


def _is_tool_record(record: dict[str, Any]) -> str | None:
    """Return a dedupe key when the record looks like a tool execution."""
    payload_type = str(record.get("payload_type") or "")
    payload = record.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    task_kind = str(payload.get("task_kind") or "")
    if ".tool." in payload_type or task_kind.startswith("tool."):
        task_id = payload.get("task_id") or record.get("id")
        return str(task_id) if task_id is not None else payload_type
    return None


def parse_muse_output(stdout_path: Path) -> dict[str, Any]:
    """Parse one `muse exec --json` stdout log into provider metrics."""
    text = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
    tool_keys: set[str] = set()
    record_count = 0
    for record in _iter_records(text):
        record_count += 1
        key = _is_tool_record(record)
        if key is not None:
            tool_keys.add(key)
    return {
        "parent_tokens": {},
        "all_model_tokens": {},
        "subagent_tokens": None,
        "tool_calls": len(tool_keys),
        "repeated_tool_calls": None,
        "subagent_count": None,
        "injected_context_bytes": 0,
        "retrieved_documents": None,
        "exact_revision_memories": None,
        "hook_events": None,
        "hook_latency_ms": None,
        "unknown_events": None,
        "selected_segments": None,
        "record_count": record_count,
    }
