"""Normalize Antigravity output into the shared benchmark metric shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _find_usage(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        for key in ("usage", "tokenUsage", "token_usage", "usageMetadata"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                numbers = {
                    "input_tokens": candidate.get("inputTokens", candidate.get("input_tokens", candidate.get("promptTokens", candidate.get("promptTokenCount")))),
                    "cache_read_input_tokens": candidate.get("cachedContentTokenCount", candidate.get("cacheReadInputTokens", candidate.get("cache_read_input_tokens", candidate.get("cache_read_tokens")))),
                    "output_tokens": candidate.get("outputTokens", candidate.get("output_tokens", candidate.get("completionTokens", candidate.get("candidatesTokenCount")))),
                    "total_tokens": candidate.get("totalTokens", candidate.get("total_tokens", candidate.get("totalTokenCount"))),
                }
                if any(isinstance(number, int) for number in numbers.values()):
                    return {key: value for key, value in numbers.items() if isinstance(value, int)}
        for child in value.values():
            found = _find_usage(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_usage(child)
            if found:
                return found
    return None


def parse_agy_output(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: list[Any] = []
    for line in text.splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # The stream contains intermediate step usage followed by a terminal
    # result usage object. Prefer the terminal record for per-turn totals.
    usage = next((_find_usage(value) for value in reversed(values) if _find_usage(value)), {})
    return {
        "provider": "agy",
        "parent_tokens": {
            "input_tokens": usage.get("input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total": usage.get("total_tokens"),
        },
        "all_model_tokens": {"total": usage.get("total_tokens")},
        "subagent_tokens": None,
        "subagent_count": 0,
        "tool_calls": None,
        "repeated_tool_calls": None,
        "retrieved_documents": None,
        "hook_latency_ms": None,
    }
