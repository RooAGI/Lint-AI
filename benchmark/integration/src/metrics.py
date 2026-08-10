"""Canonical per-phase metrics contract shared by all agent benchmarks."""

from __future__ import annotations

from typing import Any

TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def token_total(tokens: dict[str, Any]) -> int | None:
    values = [tokens.get(field) for field in TOKEN_FIELDS]
    return sum(values) if all(isinstance(value, int) for value in values) else None


def empty_tokens() -> dict[str, int | None]:
    tokens = {field: None for field in TOKEN_FIELDS}
    tokens["total"] = None
    return tokens


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def normalize_tokens(value: Any) -> dict[str, int | None]:
    """Normalize provider usage fields without guessing missing values."""
    if not isinstance(value, dict):
        return empty_tokens()
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens", "promptTokens", "prompt_tokens"),
        "cache_creation_input_tokens": (
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_creation_tokens",
        ),
        "cache_read_input_tokens": (
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_read_tokens",
            "cachedInputTokens",
        ),
        "output_tokens": ("output_tokens", "outputTokens", "completionTokens", "completion_tokens"),
    }
    result: dict[str, int | None] = {}
    for field, keys in aliases.items():
        result[field] = next((_int(value.get(key)) for key in keys if _int(value.get(key)) is not None), None)
    result["total"] = _int(value.get("total")) or _int(value.get("total_tokens")) or _int(value.get("totalTokens"))
    if result["total"] is None:
        result["total"] = token_total(result)
    return result


def canonical_metrics(metrics: Any, *, provider: str) -> dict[str, Any]:
    """Return the schema consumed by the shared report aggregator.

    Provider parsers may expose additional fields, but these fields are stable
    across Claude, Codex, and AGY. Missing provider telemetry stays null.
    """
    source = metrics if isinstance(metrics, dict) else {}
    normalized = dict(source)
    normalized["schema_version"] = 1
    normalized["provider"] = provider
    normalized["parent_tokens"] = normalize_tokens(source.get("parent_tokens"))
    normalized["all_model_tokens"] = normalize_tokens(source.get("all_model_tokens"))
    normalized["subagent_tokens"] = (
        normalize_tokens(source["subagent_tokens"])
        if isinstance(source.get("subagent_tokens"), dict)
        else None
    )
    normalized.setdefault("unattributed_non_parent_tokens", None)
    for field in (
        "subagent_count",
        "tool_calls",
        "repeated_tool_calls",
        "injected_context_bytes",
        "retrieved_documents",
        "exact_revision_memories",
        "hook_events",
        "hook_latency_ms",
        "unknown_events",
        "selected_segments",
    ):
        normalized.setdefault(field, None)
    return normalized
