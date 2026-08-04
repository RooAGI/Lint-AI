#!/usr/bin/env python3
"""Extract privacy-preserving performance metrics from Codex benchmark output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
MODEL_TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
    "output_tokens": "outputTokens",
}


def _token_total(tokens: dict[str, int | None]) -> int | None:
    values = [tokens[field] for field in TOKEN_FIELDS]
    return sum(values) if all(value is not None for value in values) else None


def _parent_tokens(result: dict[str, Any]) -> dict[str, int | None]:
    usage = result.get("usage")
    tokens = {
        field: usage.get(field) if isinstance(usage, dict) else None
        for field in TOKEN_FIELDS
    }
    tokens["total"] = _token_total(tokens)
    return tokens


def _all_model_tokens(result: dict[str, Any]) -> dict[str, int | None]:
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict):
        tokens = {field: None for field in TOKEN_FIELDS}
    else:
        tokens = {
            field: sum(
                usage.get(model_field, 0)
                for usage in model_usage.values()
                if isinstance(usage, dict)
            )
            for field, model_field in MODEL_TOKEN_FIELDS.items()
        }
    tokens["total"] = _token_total(tokens)
    return tokens


def _token_difference(
    all_model: dict[str, int | None], parent: dict[str, int | None]
) -> dict[str, int | None]:
    difference: dict[str, int | None] = {}
    for field in TOKEN_FIELDS:
        left, right = all_model[field], parent[field]
        difference[field] = (
            max(left - right, 0) if left is not None and right is not None else None
        )
    difference["total"] = _token_total(difference)
    return difference


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content or [] if isinstance(block, dict)]


def _context_strings(attachment: dict[str, Any]) -> list[str]:
    content = attachment.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    strings = []
    for item in content:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            strings.append(item["text"])
    return strings


def _empty_tokens() -> dict[str, int | None]:
    tokens = {field: None for field in TOKEN_FIELDS}
    tokens["total"] = None
    return tokens


def _token_usage_from_exec_log(events: list[dict[str, Any]]) -> dict[str, int | None]:
    """Normalize Codex's streamed usage fields to the shared metric schema.

    Codex reports cached input as a subset of ``input_tokens``. The shared
    schema records uncached input and cache reads separately, so the two are
    not double-counted in ``total``.
    """
    uncached_input = 0
    cache_creation = 0
    cache_read = 0
    output = 0
    usage_seen = False
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        usage_seen = True
        raw_input = int(usage.get("input_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        uncached_input += max(raw_input - cached, 0)
        cache_read += cached
        cache_creation += int(usage.get("cache_write_input_tokens", 0) or 0)
        output += int(usage.get("output_tokens", 0) or 0)
    if not usage_seen:
        return _empty_tokens()
    tokens = {
        "input_tokens": uncached_input,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output,
    }
    tokens["total"] = _token_total(tokens)
    return tokens


def parse_codex_exec_log(log_path: Path) -> dict[str, Any]:
    """Parse the JSONL stream emitted by ``codex exec --json``.

    The result deliberately has the same metrics shape as ``parse_run`` so
    the reporter does not need agent-specific token or tool extraction.
    """
    events: list[dict[str, Any]] = []
    unknown_events = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            unknown_events += 1
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            unknown_events += 1

    tool_item_types = {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search_call",
    }
    tool_calls = 0
    repeated_tool_calls = 0
    seen_tool_fingerprints: set[str] = set()
    for event in events:
        item = event.get("item")
        if (
            event.get("type") != "item.completed"
            or not isinstance(item, dict)
            or item.get("type") not in tool_item_types
        ):
            continue
        tool_calls += 1
        fingerprint = json.dumps(
            {"type": item.get("type"), "command": item.get("command")},
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint in seen_tool_fingerprints:
            repeated_tool_calls += 1
        else:
            seen_tool_fingerprints.add(fingerprint)

    parent = _token_usage_from_exec_log(events)
    return {
        "schema_version": 1,
        "parent_tokens": parent,
        "all_model_tokens": parent.copy(),
        "subagent_tokens": None,
        "unattributed_non_parent_tokens": {
            **{field: 0 for field in TOKEN_FIELDS},
            "total": 0,
        },
        "subagent_count": 0,
        "delegations": [],
        "tool_calls": tool_calls,
        "repeated_tool_calls": repeated_tool_calls,
        "injected_context_bytes": 0,
        "retrieved_documents": 0,
        "exact_revision_memories": 0,
        "hook_events": 0,
        "hook_latency_ms": None,
        "unknown_events": unknown_events,
        "selected_segments": 0,
    }


def parse_run(result_path: Path, transcript_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    parent = _parent_tokens(result)
    all_model = _all_model_tokens(result)

    delegations: dict[str, dict[str, Any]] = {}
    tool_calls = 0
    repeated_tool_calls = 0
    seen_tool_fingerprints: set[str] = set()
    injected_context_bytes = 0
    retrieved_documents = 0
    exact_revision_memories = 0
    hook_latency_ms = 0
    hook_events = 0
    unknown_events = 0

    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            unknown_events += 1
            continue

        for block in _content_blocks(event):
            if block.get("type") != "tool_use":
                continue
            tool_calls += 1
            fingerprint = json.dumps(
                {
                    "name": block.get("name"),
                    "input": block.get("input"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if fingerprint in seen_tool_fingerprints:
                repeated_tool_calls += 1
            else:
                seen_tool_fingerprints.add(fingerprint)

            tool_name = block.get("name")
            tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            if tool_name != "Agent":
                continue
            prompt = tool_input.get("prompt", "")
            tool_use_id = str(block.get("id", ""))
            delegations[tool_use_id] = {
                "tool_use_id": tool_use_id,
                "agent_id": None,
                "subagent_type": tool_input.get("subagent_type"),
                "purpose": tool_input.get("description"),
                "resolved_model": None,
                "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
            }

        tool_result = event.get("toolUseResult")
        tool_use_id = event.get("tool_use_id")
        if isinstance(tool_result, dict) and not tool_use_id:
            tool_use_id = next(
                (
                    block.get("tool_use_id")
                    for block in _content_blocks(event)
                    if block.get("type") == "tool_result"
                ),
                None,
            )
        if isinstance(tool_result, dict) and tool_use_id in delegations:
            delegations[tool_use_id]["agent_id"] = tool_result.get("agentId")
            delegations[tool_use_id]["resolved_model"] = tool_result.get("resolvedModel")

        attachment = event.get("attachment")
        if not isinstance(attachment, dict):
            continue
        attachment_type = attachment.get("type")
        if attachment_type == "hook_additional_context":
            for context in _context_strings(attachment):
                injected_context_bytes += len(context.encode("utf-8"))
                retrieved_documents += sum(
                    line.lstrip().startswith("- Source:") for line in context.splitlines()
                )
                exact_revision_memories += context.count("Revision status: exact-match")
        elif attachment_type == "hook_success":
            hook_events += 1
            duration = attachment.get("durationMs")
            if isinstance(duration, (int, float)):
                hook_latency_ms += duration

    non_parent = _token_difference(all_model, parent)
    has_delegations = bool(delegations)
    return {
        "schema_version": 1,
        "parent_tokens": parent,
        "all_model_tokens": all_model,
        "subagent_tokens": non_parent if has_delegations else None,
        "unattributed_non_parent_tokens": None if has_delegations else non_parent,
        "subagent_count": len(delegations),
        "delegations": list(delegations.values()),
        "tool_calls": tool_calls,
        "repeated_tool_calls": repeated_tool_calls,
        "injected_context_bytes": injected_context_bytes,
        "retrieved_documents": retrieved_documents,
        "exact_revision_memories": exact_revision_memories,
        "hook_events": hook_events,
        "hook_latency_ms": hook_latency_ms,
        "unknown_events": unknown_events,
        "selected_segments": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--codex-exec-log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.codex_exec_log:
        if args.result or args.transcript:
            raise SystemExit("--codex-exec-log cannot be combined with --result or --transcript")
        metrics = parse_codex_exec_log(args.codex_exec_log)
    elif args.result and args.transcript:
        metrics = parse_run(args.result, args.transcript)
    else:
        raise SystemExit("pass --codex-exec-log or both --result and --transcript")
    output = json.dumps(metrics, indent=2) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
