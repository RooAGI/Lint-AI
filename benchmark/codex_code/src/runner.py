#!/usr/bin/env python3
"""Codex benchmark runner scaffold.

This runner sets up isolated git worktrees, executes scenario phases through a
configurable command, runs validators, and writes structured run artifacts.
The actual Codex CLI contract is intentionally left configurable so the same
runner can support the local Codex binary or a wrapper script.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REQUIRED_SCENARIO_KEYS = {
    "schema_version",
    "id",
    "category",
    "description",
    "repository",
    "setup_messages",
    "continuation_prompt",
    "expected_facts",
    "forbidden_facts",
    "validators",
    "limits",
}


def log(message: str) -> None:
    print(f"[codex-bench] {message}", flush=True)


@dataclass(frozen=True)
class ScenarioRef:
    id: str
    path: str
    category: str
    negative_control: bool


@dataclass(frozen=True)
class RunPlan:
    scenarios: list[ScenarioRef]
    repetitions: int


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    last_path: str | None
    metrics_path: str | None
    hook_timings_path: str | None
    elapsed_ms: float


@dataclass(frozen=True)
class ValidatorResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_ms: float


@dataclass(frozen=True)
class ScenarioRunRecord:
    arm: str
    scenario_id: str
    repetition: int
    repository_revision: str
    dirty_diff_digest: str
    worktree_path: str
    setup: PhaseResult | None
    continuation: PhaseResult | None
    validators: list[ValidatorResult]
    success: bool
    invalid_reason: str | None


def load_scenario(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_scenario(scenario: dict[str, Any], source: Path | None = None) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_SCENARIO_KEYS - scenario.keys())
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if scenario.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    category = scenario.get("category")
    if not isinstance(category, str) or not category:
        errors.append("category must be a non-empty string")

    repo = scenario.get("repository")
    if not isinstance(repo, dict):
        errors.append("repository must be an object")
    else:
        for key in ("path", "revision"):
            if not isinstance(repo.get(key), str) or not repo.get(key):
                errors.append(f"repository.{key} must be a non-empty string")

    setup_messages = scenario.get("setup_messages")
    if not isinstance(setup_messages, list) or not setup_messages:
        errors.append("setup_messages must be a non-empty array")
    if not isinstance(scenario.get("continuation_prompt"), str) or not scenario[
        "continuation_prompt"
    ]:
        errors.append("continuation_prompt must be a non-empty string")
    if not isinstance(scenario.get("validators"), list):
        errors.append("validators must be an array")
    if not isinstance(scenario.get("limits"), dict):
        errors.append("limits must be an object")

    if source is not None and source.suffix != ".json":
        errors.append(f"{source} must use a .json extension")
    return errors


def discover_scenarios(root: Path) -> list[Path]:
    scenarios_dir = root / "scenarios"
    if not scenarios_dir.exists():
        return []
    return sorted(path for path in scenarios_dir.glob("*.json") if path.is_file())


def build_run_plan(scenarios: list[Path], repetitions: int) -> RunPlan:
    refs = []
    for path in scenarios:
        scenario = load_scenario(path)
        refs.append(
            ScenarioRef(
                id=str(scenario["id"]),
                path=str(path),
                category=str(scenario["category"]),
                negative_control=bool(scenario.get("negative_control", False)),
            )
        )
    return RunPlan(scenarios=refs, repetitions=repetitions)


def git_output(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def canonical_repo_root(repo_root: Path) -> Path:
    return Path(git_output(repo_root, ["rev-parse", "--show-toplevel"]).strip())


def resolve_revision(repo_root: Path, revision: str) -> str:
    return git_output(repo_root, ["rev-parse", revision]).strip()


def dirty_diff_digest(repo_root: Path) -> str:
    status = git_output(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    diff = git_output(repo_root, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    payload = f"{status}\0{diff}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_worktree(repo_root: Path, revision: str, run_root: Path) -> Path:
    worktree_path = run_root / "worktree"
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_path), revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree_path


def cleanup_worktree(repo_root: Path, worktree_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_peer_module(filename: str, module_name: str, base_dir: Path | None = None):
    module_path = (base_dir or Path(__file__).resolve().parent) / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module {filename}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdin_text: str,
    timeout_seconds: int,
    output_dir: Path,
    phase: str,
) -> PhaseResult:
    stdout_path = output_dir / f"{phase}.stdout.log"
    stderr_path = output_dir / f"{phase}.stderr.log"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return PhaseResult(
        phase=phase,
        command=command,
        returncode=completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        last_path=str(output_dir / f"{phase}.last"),
        metrics_path=None,
        hook_timings_path=str(output_dir / f"{phase}.hook-timings.jsonl"),
        elapsed_ms=elapsed_ms,
    )


def run_validators(
    validators: list[dict[str, Any]], cwd: Path, env: dict[str, str], output_dir: Path
) -> list[ValidatorResult]:
    results: list[ValidatorResult] = []
    for validator in validators:
        command = [str(part) for part in validator["command"]]
        log(f"running validator {validator['name']}")
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        result = ValidatorResult(
            name=str(validator["name"]),
            command=command,
            returncode=completed.returncode,
            elapsed_ms=elapsed_ms,
        )
        results.append(result)
        log(
            f"validator {validator['name']} finished with returncode {result.returncode} "
            f"in {result.elapsed_ms:.1f} ms"
        )
        write_json(
            output_dir / f"validator-{result.name}.json",
            {
                "name": result.name,
                "command": result.command,
                "returncode": result.returncode,
                "elapsed_ms": result.elapsed_ms,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    return results


def build_phase_env(
    base_env: dict[str, str],
    *,
    benchmark_root: Path,
    scenario: dict[str, Any],
    scenario_path: Path,
    run_dir: Path,
    worktree_path: Path,
    revision: str,
    dirty_digest: str,
    phase: str,
) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "LINT_AI_BENCHMARK_PHASE": phase,
            "LINT_AI_BENCHMARK_ROOT": str(benchmark_root),
            "LINT_AI_BENCHMARK_SCENARIO_ID": str(scenario["id"]),
            "LINT_AI_BENCHMARK_SCENARIO_PATH": str(scenario_path),
            "LINT_AI_BENCHMARK_RUN_DIR": str(run_dir),
            "LINT_AI_BENCHMARK_WORKTREE": str(worktree_path),
            "LINT_AI_BENCHMARK_REVISION": revision,
            "LINT_AI_BENCHMARK_DIRTY_DIFF_DIGEST": dirty_digest,
            "LINT_AI_HOOK_TIMINGS_PATH": str(run_dir / f"{phase}.hook-timings.jsonl"),
        }
    )
    return env


def execute_scenario(
    scenario: dict[str, Any],
    scenario_path: Path,
    *,
    arm: str,
    benchmark_root: Path,
    repo_root: Path,
    repetitions: int,
    agent_command: list[str],
    results_root: Path,
    timeout_scale: float,
    metrics_mode: str,
) -> list[ScenarioRunRecord]:
    repo_root = canonical_repo_root(repo_root)
    revision = resolve_revision(repo_root, str(scenario["repository"]["revision"]))
    dirty_digest = dirty_diff_digest(repo_root)
    outputs: list[ScenarioRunRecord] = []
    setup_prompt = "\n\n".join(message["prompt"] for message in scenario["setup_messages"])
    continuation_prompt = str(scenario["continuation_prompt"])
    timeout_seconds = int(scenario["limits"]["timeout_seconds"])
    timeout_seconds = max(1, int(timeout_seconds * timeout_scale))

    log(
        f"scenario {scenario['id']} ({scenario['category']}) "
        f"arm={arm} revision={revision} repetitions={repetitions}"
    )
    for repetition in range(1, repetitions + 1):
        run_dir = results_root / str(scenario["id"]) / f"rep-{repetition:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"scenario {scenario['id']} repetition {repetition}: preparing worktree")
        worktree_path = prepare_worktree(repo_root, revision, run_dir)
        skill_source = os.environ.get("LINT_AI_BENCHMARK_SKILL_PATH")
        if skill_source:
            skill_target = worktree_path / ".claude" / "skills" / "lint-ai-memory" / "SKILL.md"
            skill_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_source, skill_target)
            log(f"installed benchmark skill at {skill_target}")
        try:
            base_env = os.environ.copy()
            phase_env = build_phase_env(
                base_env,
                benchmark_root=benchmark_root,
                scenario=scenario,
                scenario_path=scenario_path,
                run_dir=run_dir,
                worktree_path=worktree_path,
                revision=revision,
                dirty_digest=dirty_digest,
                phase="setup",
            )
            log(
                f"scenario {scenario['id']} repetition {repetition}: running setup "
                f"phase with timeout={timeout_seconds}s"
            )
            setup = run_command(
                agent_command,
                cwd=worktree_path,
                env=phase_env,
                stdin_text=setup_prompt,
                timeout_seconds=timeout_seconds,
                output_dir=run_dir,
                phase="setup",
            )
            log(
                f"scenario {scenario['id']} repetition {repetition}: setup returned "
                f"{setup.returncode} in {setup.elapsed_ms:.1f} ms"
            )
            continuation = None
            invalid_reason = None
            validators: list[ValidatorResult] = []
            if setup.returncode != 0:
                invalid_reason = f"setup command exited {setup.returncode}"
            else:
                phase_env = build_phase_env(
                    base_env,
                    benchmark_root=benchmark_root,
                    scenario=scenario,
                    scenario_path=scenario_path,
                    run_dir=run_dir,
                    worktree_path=worktree_path,
                    revision=revision,
                    dirty_digest=dirty_digest,
                    phase="continuation",
                )
                log(
                    f"scenario {scenario['id']} repetition {repetition}: running "
                    f"continuation phase with timeout={timeout_seconds}s"
                )
                continuation = run_command(
                    agent_command,
                    cwd=worktree_path,
                    env=phase_env,
                    stdin_text=continuation_prompt,
                    timeout_seconds=timeout_seconds,
                    output_dir=run_dir,
                    phase="continuation",
                )
                log(
                    f"scenario {scenario['id']} repetition {repetition}: continuation "
                    f"returned {continuation.returncode} in {continuation.elapsed_ms:.1f} ms"
                )
                if continuation.returncode != 0:
                    invalid_reason = f"continuation command exited {continuation.returncode}"
                else:
                    if metrics_mode in ("codex", "claude"):
                        parser = load_peer_module(
                            "parse_run.py",
                            f"{metrics_mode}_parse_run",
                            benchmark_root / "src",
                        )
                        if metrics_mode == "codex":
                            metrics = parser.parse_codex_exec_log(Path(continuation.stdout_path))
                        else:
                            metrics = parser.parse_run(
                                Path(continuation.stdout_path),
                                Path(continuation.stdout_path),
                            )
                        metrics_path = run_dir / "continuation.metrics.json"
                        write_json(metrics_path, metrics)
                        continuation = replace(continuation, metrics_path=str(metrics_path))
                        log(
                            f"scenario {scenario['id']} repetition {repetition}: wrote normalized "
                            "continuation metrics"
                        )
                    log(
                        f"scenario {scenario['id']} repetition {repetition}: running "
                        f"{len(scenario['validators'])} validator(s)"
                    )
                    validators = run_validators(
                        list(scenario["validators"]),
                        cwd=worktree_path,
                        env=base_env,
                        output_dir=run_dir,
                    )
                    failed_validators = [validator for validator in validators if validator.returncode != 0]
                    if failed_validators:
                        invalid_reason = ", ".join(
                            f"{validator.name} exited {validator.returncode}"
                            for validator in failed_validators
                        )

            success = invalid_reason is None
            log(
                f"scenario {scenario['id']} repetition {repetition}: "
                f"{'success' if success else f'failed: {invalid_reason}'}"
            )
            record = ScenarioRunRecord(
                arm=arm,
                scenario_id=str(scenario["id"]),
                repetition=repetition,
                repository_revision=revision,
                dirty_diff_digest=dirty_digest,
                worktree_path=str(worktree_path),
                setup=setup,
                continuation=continuation,
                validators=validators,
                success=success,
                invalid_reason=invalid_reason,
            )
            write_json(
                run_dir / "run-record.json",
                {
                    **asdict(record),
                    "setup": asdict(setup) if setup else None,
                    "continuation": asdict(continuation) if continuation else None,
                    "validators": [asdict(validator) for validator in validators],
                },
            )
            log(f"scenario {scenario['id']} repetition {repetition}: wrote run record")
            outputs.append(record)
        finally:
            log(f"scenario {scenario['id']} repetition {repetition}: cleaning worktree")
            cleanup_worktree(repo_root, worktree_path)
    return outputs


def build_report_payload(
    *,
    benchmark_root: Path,
    repo_root: Path,
    scenarios: list[dict[str, Any]],
    scenario_paths: list[Path],
    repetitions: int,
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    scorer = load_peer_module("scorer.py", "codex_scorer")
    report = load_peer_module("report.py", "codex_report")
    scenario_by_id = {str(scenario["id"]): scenario for scenario in scenarios}
    scores = [
        asdict(scorer.score_result(record, scenario_by_id.get(str(record.get("scenario_id", "")))))
        for record in run_records
    ]
    expected_counts = {
        str(scenario["id"]): len(scenario.get("expected_facts", []))
        for scenario in scenarios
    }
    summary = report.summarize_results(
        run_records, scored=scores, expected_counts=expected_counts
    )
    return {
        "benchmark_root": str(benchmark_root),
        "repo_root": str(repo_root),
        "scenario_count": len(scenario_paths),
        "repetitions": repetitions,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Codex benchmark root",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root to benchmark",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate scenarios and print the discovered run plan",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the configured agent command for each scenario",
    )
    parser.add_argument(
        "--agent-command",
        nargs="+",
        help="Command used for setup and continuation phases",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to execute; repeat to select multiple scenarios",
    )
    parser.add_argument("--arm", default="lint-ai")
    parser.add_argument("--metrics", choices=("codex", "claude", "none"), default="codex")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    scenario_paths = discover_scenarios(args.benchmark_root)
    if args.scenario:
        selected = set(args.scenario)
        scenario_paths = [
            path for path in scenario_paths if str(load_scenario(path).get("id", "")) in selected
        ]
        missing = selected - {str(load_scenario(path).get("id", "")) for path in scenario_paths}
        if missing:
            raise SystemExit(f"unknown scenario id(s): {', '.join(sorted(missing))}")
    errors: dict[str, list[str]] = {}
    scenarios: list[dict[str, Any]] = []
    for path in scenario_paths:
        scenario = load_scenario(path)
        scenarios.append(scenario)
        validation_errors = validate_scenario(scenario, path)
        if validation_errors:
            errors[str(path)] = validation_errors

    payload: dict[str, Any] = {
        "benchmark_root": str(args.benchmark_root),
        "repo_root": str(args.repo_root),
        "scenario_count": len(scenario_paths),
        "validation_errors": errors,
        "run_plan": asdict(build_run_plan(scenario_paths, args.repetitions)),
        "status": "ok" if not errors else "invalid",
    }

    if args.execute:
        if not args.agent_command:
            raise SystemExit("--execute requires --agent-command")
        results_dir = args.results_dir or (args.benchmark_root / "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        log(f"results directory: {results_dir}")
        run_records: list[dict[str, Any]] = []
        for scenario, path in zip(scenarios, scenario_paths):
            validation_errors = validate_scenario(scenario, path)
            if validation_errors:
                log(f"skipping invalid scenario {path}: {', '.join(validation_errors)}")
                continue
            log(f"executing scenario file {path}")
            run_records.extend(
                asdict(
                    record,
                )
                for record in execute_scenario(
                    scenario,
                    path,
                    arm=args.arm,
                    benchmark_root=args.benchmark_root,
                    repo_root=args.repo_root,
                    repetitions=args.repetitions,
                    agent_command=args.agent_command,
                    results_root=results_dir,
                    timeout_scale=args.timeout_scale,
                    metrics_mode=args.metrics,
                )
            )
        payload["execution"] = {
            "agent_command": args.agent_command,
            "results_dir": str(results_dir),
            "runs": run_records,
        }
        log("building report payload")
        report_payload = build_report_payload(
            benchmark_root=args.benchmark_root,
            repo_root=args.repo_root,
            scenarios=scenarios,
            scenario_paths=scenario_paths,
            repetitions=args.repetitions,
            run_records=run_records,
        )
        payload["report"] = report_payload
        report_path = args.report_out or (results_dir / "report.json")
        log(f"writing report to {report_path}")
        write_json(report_path, report_payload)
        payload["report_path"] = str(report_path)
        payload["status"] = "ok" if not errors and all(record["success"] for record in run_records) else "invalid"
    elif not args.validate_only:
        payload["note"] = "pass --execute and --agent-command to run scenarios"

    output = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
