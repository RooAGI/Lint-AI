#!/usr/bin/env python3
"""Codex benchmark reporting scaffold."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def _phase_elapsed(result: dict[str, Any], phase: str) -> float | None:
    value = result.get(phase)
    if not isinstance(value, dict):
        return None
    elapsed = value.get("elapsed_ms")
    return float(elapsed) if isinstance(elapsed, (int, float)) else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent_delta(baseline: Any, lint_ai: Any) -> float | None:
    if not isinstance(baseline, (int, float)) or not isinstance(lint_ai, (int, float)):
        return None
    if baseline == 0:
        return None
    return (lint_ai - baseline) / baseline


def _continuation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Load the normalized metrics artifact written by the benchmark runner."""
    continuation = result.get("continuation")
    path_value = continuation.get("metrics_path") if isinstance(continuation, dict) else None
    if not isinstance(path_value, str) or not Path(path_value).exists():
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "reasoning_output_tokens": None,
            "tool_calls": None,
            "repeated_tool_calls": None,
        }
    metrics = json.loads(Path(path_value).read_text(encoding="utf-8"))
    parent = metrics.get("parent_tokens")
    parent = parent if isinstance(parent, dict) else {}
    uncached_input = parent.get("input_tokens")
    cached_input = parent.get("cache_read_input_tokens")
    input_tokens = (
        uncached_input + cached_input
        if isinstance(uncached_input, int) and isinstance(cached_input, int)
        else None
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input,
        "output_tokens": parent.get("output_tokens"),
        "reasoning_output_tokens": None,
        "tool_calls": metrics.get("tool_calls"),
        "repeated_tool_calls": metrics.get("repeated_tool_calls"),
    }


def _hook_timing_summary(result: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_result = result.get(phase)
    path_value = phase_result.get("hook_timings_path") if isinstance(phase_result, dict) else None
    if not isinstance(path_value, str) or not Path(path_value).exists():
        return {"events": 0, "elapsed_ms": 0.0, "retrieve_ms": 0.0, "capture_ms": 0.0}
    records = []
    for line in Path(path_value).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    def total(operation: str | None = None) -> float:
        return sum(
            float(record["elapsed_ms"])
            for record in records
            if isinstance(record.get("elapsed_ms"), (int, float))
            and (operation is None or record.get("operation") == operation)
        )
    return {
        "events": len(records),
        "elapsed_ms": total(),
        "retrieve_ms": total("retrieve"),
        "capture_ms": total("capture"),
    }


def build_final_results(
    results: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return the directly reportable outcome of the benchmark run."""
    score_by_key = {
        (str(score.get("scenario_id", "")), int(score.get("repetition", 0) or 0)): score
        for score in scored
    }
    scenarios: list[dict[str, Any]] = []
    setup_times: list[float] = []
    continuation_times: list[float] = []
    validator_times: list[float] = []
    input_tokens: list[float] = []
    cached_input_tokens: list[float] = []
    output_tokens: list[float] = []
    tool_call_counts: list[float] = []
    repeated_tool_call_counts: list[float] = []
    setup_hook_times: list[float] = []
    continuation_hook_times: list[float] = []
    for result in results:
        scenario_id = str(result.get("scenario_id", ""))
        repetition = int(result.get("repetition", 0) or 0)
        setup_ms = _phase_elapsed(result, "setup")
        continuation_ms = _phase_elapsed(result, "continuation")
        validator_ms = sum(
            float(validator["elapsed_ms"])
            for validator in result.get("validators", [])
            if isinstance(validator, dict)
            and isinstance(validator.get("elapsed_ms"), (int, float))
        )
        if setup_ms is not None:
            setup_times.append(setup_ms)
        if continuation_ms is not None:
            continuation_times.append(continuation_ms)
        validator_times.append(validator_ms)
        score = score_by_key.get((scenario_id, repetition), {})
        usage = _continuation_metrics(result)
        setup_hooks = _hook_timing_summary(result, "setup")
        continuation_hooks = _hook_timing_summary(result, "continuation")
        setup_hook_times.append(setup_hooks["elapsed_ms"])
        continuation_hook_times.append(continuation_hooks["elapsed_ms"])
        for values, key in (
            (input_tokens, "input_tokens"),
            (cached_input_tokens, "cached_input_tokens"),
            (output_tokens, "output_tokens"),
            (tool_call_counts, "tool_calls"),
            (repeated_tool_call_counts, "repeated_tool_calls"),
        ):
            if usage[key] is not None:
                values.append(float(usage[key]))
        expected_count = len(score.get("expected_fact_ids_found", []))
        expected_total = (
            expected_counts.get(scenario_id, expected_count)
            if expected_counts is not None
            else expected_count
        )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "arm": str(result.get("arm", "lint-ai")),
                "repetition": repetition,
                "success": bool(result.get("success", False)),
                "invalid_reason": result.get("invalid_reason"),
                "setup_elapsed_ms": setup_ms,
                "continuation_elapsed_ms": continuation_ms,
                # The follow-up prompt is the comparable user-visible interaction.
                "interaction_round_latency_ms": continuation_ms,
                "setup_hooks": setup_hooks,
                "continuation_hooks": continuation_hooks,
                "validator_elapsed_ms": validator_ms,
                "expected_fact_ids_found": score.get("expected_fact_ids_found", []),
                "forbidden_fact_ids_found": score.get("forbidden_fact_ids_found", []),
                "input_tokens": usage["input_tokens"],
                "cached_input_tokens": usage["cached_input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_output_tokens": usage["reasoning_output_tokens"],
                "tool_calls": usage["tool_calls"],
                "repeated_tool_calls": usage["repeated_tool_calls"],
                "recall": {
                    "found": expected_count,
                    "expected": expected_total,
                    "rate": expected_count / expected_total if expected_total else None,
                },
            }
        )

    successful = sum(1 for result in results if result.get("success"))
    arms = {str(result.get("arm", "lint-ai")) for result in results}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in sorted(arms):
        arm_scenarios = [item for item in scenarios if item["arm"] == arm]
        found = sum(item["recall"]["found"] for item in arm_scenarios)
        expected = sum(item["recall"]["expected"] for item in arm_scenarios)
        arm_summaries[arm] = {
            "runs": len(arm_scenarios),
            "successful_runs": sum(1 for item in arm_scenarios if item["success"]),
            "recall": {
                "found": found,
                "expected": expected,
                "rate": found / expected if expected else None,
            },
            "average_input_tokens": _average(
                [item["input_tokens"] for item in arm_scenarios if item["input_tokens"] is not None]
            ),
            "average_cached_input_tokens": _average(
                [
                    item["cached_input_tokens"]
                    for item in arm_scenarios
                    if item["cached_input_tokens"] is not None
                ]
            ),
            "average_output_tokens": _average(
                [item["output_tokens"] for item in arm_scenarios if item["output_tokens"] is not None]
            ),
            "average_tool_calls": _average(
                [item["tool_calls"] for item in arm_scenarios if item["tool_calls"] is not None]
            ),
            "average_repeated_tool_calls": _average(
                [
                    item["repeated_tool_calls"]
                    for item in arm_scenarios
                    if item["repeated_tool_calls"] is not None
                ]
            ),
            "average_setup_hook_elapsed_ms": _average(
                [item["setup_hooks"]["elapsed_ms"] for item in arm_scenarios]
            ),
            "average_continuation_hook_elapsed_ms": _average(
                [item["continuation_hooks"]["elapsed_ms"] for item in arm_scenarios]
            ),
            "average_continuation_elapsed_ms": _average(
                [
                    item["continuation_elapsed_ms"]
                    for item in arm_scenarios
                    if item["continuation_elapsed_ms"] is not None
                ]
            ),
            "average_interaction_round_latency_ms": _average(
                [
                    item["interaction_round_latency_ms"]
                    for item in arm_scenarios
                    if item["interaction_round_latency_ms"] is not None
                ]
            ),
        }
    comparison: list[dict[str, Any]] = []
    reference_arm = "codex-native" if "codex-native" in arms else "baseline"
    candidate_arms = sorted(arms - {reference_arm})
    if reference_arm in arms and candidate_arms:
        by_key = {
            (str(item["scenario_id"]), int(item["repetition"]), str(item["arm"])): item
            for item in scenarios
        }
        metrics = (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "tool_calls",
            "repeated_tool_calls",
            "interaction_round_latency_ms",
        )
        for candidate_arm in candidate_arms:
            keys = sorted(
                {
                    (scenario_id, repetition)
                    for scenario_id, repetition, arm in by_key
                    if arm == reference_arm
                }
                & {
                    (scenario_id, repetition)
                    for scenario_id, repetition, arm in by_key
                    if arm == candidate_arm
                }
            )
            for scenario_id, repetition in keys:
                baseline = by_key[(scenario_id, repetition, reference_arm)]
                candidate = by_key[(scenario_id, repetition, candidate_arm)]
                reference_values = {
                    "recall": baseline["recall"],
                    **{metric: baseline[metric] for metric in metrics},
                }
                candidate_values = {
                    "recall": candidate["recall"],
                    **{metric: candidate[metric] for metric in metrics},
                }
                delta = {
                    metric: candidate[metric] - baseline[metric]
                    if isinstance(candidate[metric], (int, float))
                    and isinstance(baseline[metric], (int, float))
                    else None
                    for metric in metrics
                }
                comparison.append(
                    {
                        "scenario_id": scenario_id,
                        "repetition": repetition,
                        "reference_arm": reference_arm,
                        "candidate_arm": candidate_arm,
                        "reference": reference_values,
                        "candidate": candidate_values,
                        "candidate_minus_reference": delta,
                        "candidate_percent_delta_from_reference": {
                            metric: _percent_delta(baseline[metric], candidate[metric])
                            for metric in metrics
                        },
                    }
                )
    return {
        "status": "passed" if successful == len(results) else "failed",
        "total_runs": len(results),
        "successful_runs": successful,
        "invalid_runs": len(results) - successful,
        "success_rate": successful / len(results) if results else 0.0,
        "average_setup_elapsed_ms": _average(setup_times),
        "average_continuation_elapsed_ms": _average(continuation_times),
        "average_interaction_round_latency_ms": _average(continuation_times),
        "average_validator_elapsed_ms": _average(validator_times),
        "average_input_tokens": _average(input_tokens),
        "average_cached_input_tokens": _average(cached_input_tokens),
        "average_output_tokens": _average(output_tokens),
        "average_tool_calls": _average(tool_call_counts),
        "average_repeated_tool_calls": _average(repeated_tool_call_counts),
        "comparison_note": (
            f"This report compares {', '.join(candidate_arms)} against {reference_arm}."
            if reference_arm in arms and candidate_arms
            else (
                "This report contains only the lint-ai arm. A baseline arm has not "
                "been run, so percentage comparisons are not valid."
                if arms == {"lint-ai"}
                else "A native reference arm has not been run, so percentage comparisons "
                "are not valid."
            )
        ),
        "arm_summaries": arm_summaries,
        "scenarios": scenarios,
        "comparison": comparison,
    }


def summarize_results(
    results: list[dict[str, Any]],
    scored: list[dict[str, Any]] | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    scored = scored or []
    by_scenario: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    success_count = 0
    invalid_count = 0
    for result in results:
        scenario_id = str(result.get("scenario_id", ""))
        arm = str(result.get("arm", "lint-ai"))
        by_scenario[scenario_id][arm] += 1
        if result.get("success"):
            success_count += 1
        else:
            invalid_count += 1
    return {
        "result_count": len(results),
        "success_count": success_count,
        "invalid_count": invalid_count,
        "scenarios": {
            scenario_id: dict(arms) for scenario_id, arms in sorted(by_scenario.items())
        },
        "scores": scored,
        "final_results": build_final_results(results, scored, expected_counts),
    }


def combine_reports(
    reports: list[dict[str, Any]], expected_counts: dict[str, int]
) -> dict[str, Any]:
    """Combine independently executed arms into one comparison report."""
    if not reports:
        raise ValueError("at least one report is required")
    runs = [
        run
        for report in reports
        for run in report.get("execution", {}).get("runs", [])
    ]
    scores = [
        score
        for report in reports
        for score in report.get("report", {}).get("summary", {}).get("scores", [])
    ]
    first = reports[0]
    return {
        "benchmark_root": first["benchmark_root"],
        "repo_root": first["repo_root"],
        "scenario_count": first["scenario_count"],
        "repetitions": first["run_plan"]["repetitions"],
        "summary": summarize_results(runs, scores, expected_counts),
    }
