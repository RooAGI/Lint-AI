#!/usr/bin/env python3
"""Codex benchmark scoring scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioScore:
    scenario_id: str
    arm: str
    repetition: int
    success: bool
    expected_fact_ids_found: list[str]
    forbidden_fact_ids_found: list[str]
    invalid_reason: str | None
    validator_count: int


def _match_fact_ids(facts: list[dict[str, Any]], text: str) -> list[str]:
    text_lower = text.lower()
    matched: list[str] = []
    for fact in facts:
        fact_id = str(fact.get("id", ""))
        match_any = fact.get("match_any", [])
        if not fact_id or not isinstance(match_any, list):
            continue
        if any(str(pattern).lower() in text_lower for pattern in match_any):
            matched.append(fact_id)
    return matched


def _read_text(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def score_result(result: dict[str, Any], scenario: dict[str, Any] | None = None) -> ScenarioScore:
    continuation = result.get("continuation")
    continuation_text = ""
    if isinstance(continuation, dict):
        continuation_text = _read_text(continuation.get("last_path"))
        if not continuation_text:
            continuation_text = _read_text(continuation.get("stdout_path"))
    expected_fact_ids_found: list[str] = []
    forbidden_fact_ids_found: list[str] = []
    if scenario is not None:
        expected_fact_ids_found = _match_fact_ids(
            list(scenario.get("expected_facts", [])), continuation_text
        )
        forbidden_fact_ids_found = _match_fact_ids(
            list(scenario.get("forbidden_facts", [])), continuation_text
        )
    return ScenarioScore(
        scenario_id=str(result.get("scenario_id", "")),
        arm=str(result.get("arm", "lint-ai")),
        repetition=int(result.get("repetition", 0) or 0),
        success=bool(result.get("success", False)),
        expected_fact_ids_found=expected_fact_ids_found
        if scenario is not None
        else list(result.get("expected_fact_ids_found", [])),
        forbidden_fact_ids_found=forbidden_fact_ids_found
        if scenario is not None
        else list(result.get("forbidden_fact_ids_found", [])),
        invalid_reason=result.get("invalid_reason"),
        validator_count=len(result.get("validators", []))
        if isinstance(result.get("validators", []), list)
        else 0,
    )
