#!/usr/bin/env python3
"""Codex benchmark runner scaffold.

This runner sets up isolated git worktrees, executes scenario phases through a
configurable command, runs validators, and writes structured run artifacts.
The actual Codex CLI contract is intentionally left configurable so the same
runner can support the local Codex binary or a wrapper script.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import importlib.util
import hashlib
import json
import os
import pty
import select
import signal
import shutil
import subprocess
import termios
import tempfile
import time
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)


def extract_session_id(output_text: str) -> str | None:
    """Extract a resumable session/conversation ID from JSON lines or text."""
    for line in reversed(output_text.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                if "conversation_id" in payload and isinstance(payload["conversation_id"], str):
                    return payload["conversation_id"]
                if "result" in payload and isinstance(payload["result"], dict) and "conversation_id" in payload["result"]:
                    return str(payload["result"]["conversation_id"])
                for key in ("session_id", "sessionId", "thread_id", "threadId", "conversation_id", "conversationId"):
                    if key in payload and isinstance(payload[key], str):
                        return payload[key]
        except json.JSONDecodeError:
            continue
    ids = UUID_RE.findall(output_text)
    return ids[-1] if ids else None

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
    cleanup_worktree(repo_root, worktree_path)
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_path), revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree_path


def cleanup_worktree(repo_root: Path, worktree_path: Path) -> None:
    """Remove a worktree left over from a prior (uncleaned) run of the same rep dir."""
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


def extract_claude_last_message(stdout_text: str) -> str | None:
    """Return the text of the final assistant message from Claude's
    stream-json output, so scoring sees only the model's own final answer
    instead of the full transcript (which includes tool-result content that
    can quote scenario/scoring text verbatim and contaminate fact matching).
    """
    last_text: str | None = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "assistant":
            continue
        content = record.get("message", {}).get("content", [])
        text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]
        if text_blocks:
            last_text = "".join(text_blocks)
    return last_text


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
    last_path = output_dir / f"{phase}.last"
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
    if not last_path.exists():
        # The wrapped CLI didn't write its own final-message file (only Codex's
        # --output-last-message does); extract it from stream-json output so
        # scoring never falls back to the raw tool-call transcript.
        last_message = extract_claude_last_message(completed.stdout)
        if last_message is not None:
            last_path.write_text(last_message, encoding="utf-8")
    return PhaseResult(
        phase=phase,
        command=command,
        returncode=completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        last_path=str(last_path),
        metrics_path=None,
        hook_timings_path=str(output_dir / f"{phase}.hook-timings.jsonl"),
        elapsed_ms=elapsed_ms,
    )


def run_pty_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdin_text: str,
    timeout_seconds: int,
    output_dir: Path,
    phase: str,
) -> PhaseResult:
    """Run an interactive CLI through a pseudo-terminal.

    AGY only emits its lifecycle hooks from the interactive execution loop.
    The prompt is followed by EOT so one benchmark phase remains bounded and
    can still be launched as a fresh conversation or resumed by its wrapper.
    """
    stdout_path = output_dir / f"{phase}.stdout.log"
    stderr_path = output_dir / f"{phase}.stderr.log"
    started = time.monotonic()
    master, slave = pty.openpty()
    chunks: list[bytes] = []
    timed_out = False
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(slave, termios.TIOCSCTTY, 0),
        )
        os.close(slave)
        os.write(master, stdin_text.encode("utf-8"))
        if not stdin_text.endswith("\n"):
            os.write(master, b"\n")
        # AGY enters its interactive prompt asynchronously. Sending EOT before
        # that transition is consumed as part of the initial prompt instead of
        # closing the phase.
        time.sleep(2.0)
        os.write(master, b"\x04")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.kill()
                break
            ready, _, _ = select.select([master], [], [], min(0.25, remaining))
            if ready:
                try:
                    chunks.append(os.read(master, 65536))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
            if process.poll() is not None and not ready:
                break
            # Some AGY versions keep the interactive shell alive after EOT.
            # Once the response has had time to flush, interrupt only that
            # shell; the captured transcript remains valid for scoring.
            # AGY's lifecycle hooks (and the model turn itself) may not
            # complete within a short fixed window, so give the loop most of
            # the phase timeout budget rather than a small hardcoded floor,
            # leaving a margin before the hard deadline kills the process.
            interrupt_after = max(20.0, timeout_seconds - 15.0)
            if process.poll() is None and time.monotonic() - started > min(interrupt_after, timeout_seconds):
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (PermissionError, ProcessLookupError):
                        pass
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
                break
        returncode = process.wait(timeout=5) if process.poll() is None else process.returncode
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if 'slave' in locals():
            try:
                os.close(slave)
            except OSError:
                pass
    elapsed_ms = (time.monotonic() - started) * 1000.0
    output = b"".join(chunks).decode("utf-8", errors="replace")
    stdout_path.write_text(output, encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    if timed_out:
        returncode = 124
    return PhaseResult(
        phase=phase,
        command=command,
        returncode=returncode,
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


def build_turn_prompt(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    """Build a phase's stdin payload from one or more turn messages.

    A single message is sent as plain text (unchanged legacy behavior). More
    than one message is delivered as sequential stream-json user-turn lines
    into one live session (see run_command's --multiturn contract), so
    scenarios that need multiple turns within a single phase get a real
    multi-turn conversation instead of one flattened prompt.
    """
    if len(messages) > 1:
        prompt = "".join(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": message["prompt"]}],
                    },
                }
            )
            + "\n"
            for message in messages
        )
        return prompt, True
    return "\n\n".join(message["prompt"] for message in messages), False


# Providers whose CLI can accept a stream of turn messages on one process's
# stdin (Claude's --input-format stream-json) and therefore never need more
# than one process for a multi-turn phase. Everything else (Codex, AGY) has
# no such mode -- multi-turn there means one process per turn, chained by
# passing the resume/session id back in as an explicit --resume argument.
STREAM_MULTITURN_PROVIDERS = {"claude"}


def run_turn_phase(
    *,
    phase_runner: Any,
    agent_command: list[str],
    messages: list[dict[str, Any]],
    metrics_mode: str,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    output_dir: Path,
    phase: str,
    scenario_id: str,
    repetition: int,
) -> PhaseResult:
    """Run one phase (setup or continuation) as its own process/session.

    Setup and continuation differ only in which messages they carry and
    which phase_env/output paths apply -- the actual "run this turn (or
    sequence of turns) as one session" logic is identical, so both call
    sites share this one function instead of duplicating it. A single
    message is always one plain-text call, regardless of provider. More
    than one message goes through whichever multi-turn strategy the
    provider actually supports (see STREAM_MULTITURN_PROVIDERS).
    """
    if len(messages) > 1 and metrics_mode not in STREAM_MULTITURN_PROVIDERS:
        return run_resume_chain_phase(
            phase_runner=phase_runner,
            agent_command=agent_command,
            messages=messages,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            phase=phase,
            scenario_id=scenario_id,
            repetition=repetition,
        )
    stdin_text, multiturn = build_turn_prompt(messages)
    command = agent_command + (["--multiturn"] if multiturn else [])
    log(
        f"scenario {scenario_id} repetition {repetition}: running {phase} "
        f"phase with timeout={timeout_seconds}s"
    )
    result = phase_runner(
        command,
        cwd=cwd,
        env=env,
        stdin_text=stdin_text,
        timeout_seconds=timeout_seconds,
        output_dir=output_dir,
        phase=phase,
    )
    log(
        f"scenario {scenario_id} repetition {repetition}: {phase} returned "
        f"{result.returncode} in {result.elapsed_ms:.1f} ms"
    )
    return result


def run_resume_chain_phase(
    *,
    phase_runner: Any,
    agent_command: list[str],
    messages: list[dict[str, Any]],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    output_dir: Path,
    phase: str,
    scenario_id: str,
    repetition: int,
) -> PhaseResult:
    """Multi-turn via a resume id chained between separate processes.

    Used by CLIs (Codex, AGY) with no way to stream several turns into one
    live process. Each message is its own process; the session/thread id
    that process reports is passed to the next one as an explicit
    "--resume <id>" argument -- never through the environment or a file the
    wrapper has to poll for. Intermediate turns get their own phase name
    (e.g. "setup-turn-1") for diagnostics; the final turn's result is
    returned relabeled as `phase` so callers see the same PhaseResult shape
    a single-turn phase would have produced.
    """
    resume_id: str | None = None
    result: PhaseResult | None = None
    for index, message in enumerate(messages, start=1):
        command = list(agent_command)
        if resume_id:
            command += ["--resume", resume_id]
        turn_phase = phase if index == len(messages) else f"{phase}-turn-{index}"
        log(
            f"scenario {scenario_id} repetition {repetition}: running {turn_phase} "
            f"phase with timeout={timeout_seconds}s"
        )
        result = phase_runner(
            command,
            cwd=cwd,
            env=env,
            stdin_text=message["prompt"],
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            phase=turn_phase,
        )
        log(
            f"scenario {scenario_id} repetition {repetition}: {turn_phase} returned "
            f"{result.returncode} in {result.elapsed_ms:.1f} ms"
        )
        if result.returncode != 0 or index == len(messages):
            break
        turn_output = Path(result.stdout_path).read_text(encoding="utf-8", errors="replace")
        resume_id = extract_session_id(turn_output)
        if not resume_id:
            log(
                f"scenario {scenario_id} repetition {repetition}: {turn_phase} did not "
                "report a resumable session id; remaining turns will start fresh"
            )
            resume_id = None
            continue
    assert result is not None
    return replace(result, phase=phase)


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
    execution_mode: str,
    metrics_root: Path | None,
) -> list[ScenarioRunRecord]:
    repo_root = canonical_repo_root(repo_root)
    revision = resolve_revision(repo_root, str(scenario["repository"]["revision"]))
    dirty_digest = dirty_diff_digest(repo_root)
    outputs: list[ScenarioRunRecord] = []
    setup_messages = list(scenario["setup_messages"])
    # continuation_messages is an optional array (mirroring setup_messages)
    # for scenarios that need continuation itself to span several turns. It
    # always starts a session of its own, separate from setup's session --
    # only turns *within* continuation share state, never turns from setup.
    # Scenarios without this field keep the single continuation_prompt they
    # already have, wrapped as a one-message list so run_turn_phase sees the
    # same shape either way.
    continuation_messages_field = scenario.get("continuation_messages")
    if continuation_messages_field:
        continuation_messages = list(continuation_messages_field)
    else:
        continuation_messages = [{"prompt": str(scenario["continuation_prompt"])}]
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
            phase_runner = run_pty_command if execution_mode == "pty" else run_command
            setup = run_turn_phase(
                phase_runner=phase_runner,
                agent_command=agent_command,
                messages=setup_messages,
                metrics_mode=metrics_mode,
                cwd=worktree_path,
                env=phase_env,
                timeout_seconds=timeout_seconds,
                output_dir=run_dir,
                phase="setup",
                scenario_id=str(scenario["id"]),
                repetition=repetition,
            )
            continuation = None
            invalid_reason = None
            validators: list[ValidatorResult] = []
            if setup.returncode != 0:
                invalid_reason = f"setup command exited {setup.returncode}"
            else:
                setup_output = Path(setup.stdout_path).read_text(encoding="utf-8", errors="replace")
                conversation_id = extract_session_id(setup_output)
                if conversation_id:
                    (run_dir / "conversation.id").write_text(conversation_id + "\n", encoding="utf-8")
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
                continuation = run_turn_phase(
                    phase_runner=phase_runner,
                    agent_command=agent_command,
                    messages=continuation_messages,
                    metrics_mode=metrics_mode,
                    cwd=worktree_path,
                    env=phase_env,
                    timeout_seconds=timeout_seconds,
                    output_dir=run_dir,
                    phase="continuation",
                    scenario_id=str(scenario["id"]),
                    repetition=repetition,
                )
                if continuation.returncode != 0:
                    invalid_reason = f"continuation command exited {continuation.returncode}"
                else:
                    if metrics_mode in ("codex", "claude", "agy"):
                        parser = load_peer_module(
                            "parse_run.py",
                            f"{metrics_mode}_parse_run",
                            (metrics_root or benchmark_root) / "src",
                        )
                        if metrics_mode == "codex":
                            metrics = parser.parse_codex_exec_log(Path(continuation.stdout_path))
                        elif metrics_mode == "claude":
                            metrics = parser.parse_run(
                                Path(continuation.stdout_path),
                                Path(continuation.stdout_path),
                            )
                        else:
                            metrics = parser.parse_agy_output(Path(continuation.stdout_path))
                        shared_metrics = load_peer_module(
                            "metrics.py",
                            "integration_metrics",
                            Path(__file__).resolve().parents[2] / "integration" / "src",
                        )
                        metrics = shared_metrics.canonical_metrics(
                            metrics, provider=metrics_mode
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
            log(
                f"scenario {scenario['id']} repetition {repetition}: leaving worktree "
                f"in place at {worktree_path} for inspection (removed automatically on next run)"
            )
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
    parser.add_argument("--metrics", choices=("codex", "claude", "agy", "none"), default="codex")
    parser.add_argument("--execution-mode", choices=("pipe", "pty"), default="pipe")
    parser.add_argument("--metrics-root", type=Path, default=None)
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
                    execution_mode=args.execution_mode,
                    metrics_root=args.metrics_root,
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
