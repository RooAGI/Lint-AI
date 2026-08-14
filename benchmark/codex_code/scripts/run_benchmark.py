#!/usr/bin/env python3
"""Run the Codex benchmark suite and keep the temp tree for inspection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CLAUDE_EQUIVALENT_HOOK_EVENTS = {
    "SessionStart",
    "UserPromptSubmit",
    "UserPromptExpansion",
    "PreCompact",
    "Stop",
    "SessionEnd",
}


def copy_codex_auth(source: Path, destination: Path) -> None:
    auth = source / "auth.json"
    if not auth.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    # Keep credentials required by `codex exec`, but never import host sessions,
    # memories, caches, plugins, or configuration into a benchmark arm.
    shutil.copy2(auth, destination / "auth.json")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_codex_home(
    source: Path,
    destination: Path,
    *,
    memories_enabled: bool,
    mcp_feature_enabled: bool,
) -> None:
    copy_codex_auth(source, destination)
    source_config = source / "config.toml"
    model_lines: list[str] = []
    if source_config.exists():
        for line in source_config.read_text(encoding="utf-8").splitlines():
            if re.match(r"^(model|model_reasoning_effort)\s*=", line):
                model_lines.append(line)
    (destination / "config.toml").write_text(
        "\n".join(
            model_lines
            + [
                "",
                "[features]",
                f"memories = {str(memories_enabled).lower()}",
                f"mcp_2026_07_28 = {str(mcp_feature_enabled).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not memories_enabled:
        (destination / "hooks.json").unlink(missing_ok=True)


def load_report_module(path: Path):
    spec = importlib.util.spec_from_file_location("codex_report_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Codex report module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restrict_to_claude_equivalent_hooks(codex_home: Path) -> None:
    """Keep the Lint-AI benchmark lifecycle aligned with the Claude arm."""
    settings_path = codex_home / "hooks.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        raise RuntimeError("Codex hook settings must contain a hooks object")
    settings["hooks"] = {
        event_name: entries
        for event_name, entries in hooks.items()
        if event_name in CLAUDE_EQUIVALENT_HOOK_EVENTS
    }
    write_json(settings_path, settings)


def remove_lint_ai_mcp_server(codex_home: Path) -> None:
    """Keep the memory benchmark from measuring one-shot MCP code indexing."""
    config_path = codex_home / "config.toml"
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    retained: list[str] = []
    in_lint_ai_server = False
    for line in lines:
        if line.strip() == "[mcp_servers.lint-ai]":
            in_lint_ai_server = True
            continue
        if in_lint_ai_server and line.lstrip().startswith("["):
            in_lint_ai_server = False
        if not in_lint_ai_server:
            retained.append(line)
    config_path.write_text("".join(retained).rstrip() + "\n", encoding="utf-8")


def configure_mcp_diagnostics(codex_home: Path, executable: Path, trace_path: Path) -> None:
    """Make the MCP launch observable without changing the server protocol."""
    config_path = codex_home / "config.toml"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    in_lint_ai_server = False
    env_written = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[mcp_servers.lint-ai]":
            in_lint_ai_server = True
        elif in_lint_ai_server and stripped.startswith("["):
            if not env_written:
                updated.append(
                    f'env = {{ LINT_AI_MCP_TRACE_PATH = {json.dumps(str(trace_path))} }}'
                )
                env_written = True
            in_lint_ai_server = False
        if in_lint_ai_server and stripped.startswith("command"):
            updated.append(f"command = {json.dumps(str(executable))}")
            continue
        updated.append(line)
    if in_lint_ai_server and not env_written:
        updated.append(f'env = {{ LINT_AI_MCP_TRACE_PATH = {json.dumps(str(trace_path))} }}')
    config_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def capture_codex_diagnostics(codex_home: Path, env: dict[str, str], output_dir: Path) -> None:
    """Capture the client-side view before starting a benchmark phase."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "environment.json",
        {
            "HOME": env.get("HOME"),
            "CODEX_HOME": env.get("CODEX_HOME"),
            "PATH": env.get("PATH"),
            "config_path": str(codex_home / "config.toml"),
            "hooks_path": str(codex_home / "hooks.json"),
        },
    )
    shutil.copy2(codex_home / "config.toml", output_dir / "config.toml")
    hooks_path = codex_home / "hooks.json"
    if hooks_path.exists():
        shutil.copy2(hooks_path, output_dir / "hooks.json")
    for name, command in (
        ("codex-version", ["codex", "--version"]),
        ("codex-mcp-list", ["codex", "mcp", "list"]),
        ("codex-features-list", ["codex", "features", "list"]),
    ):
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (output_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        write_json(
            output_dir / f"{name}.json",
            {"command": command, "returncode": completed.returncode},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root to benchmark",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Benchmark root",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("hooks-only", "production"),
        default="hooks-only",
        help="Measure memory hooks only or the full MCP-backed integration",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("codex-native", "lint-ai", "lint-ai-with-codex-memory"),
        default=("codex-native", "lint-ai", "lint-ai-with-codex-memory"),
        help="Benchmark arms to execute (default: both)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to execute; repeat to select multiple scenarios",
    )
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory that receives the preserved report and run artifacts",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    results_dir = (args.results_dir or (benchmark_root / "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    host_home = Path.home()
    host_cargo_home = host_home / ".cargo"
    host_rustup_home = host_home / ".rustup"
    temp_root = Path(tempfile.mkdtemp(prefix="lint-ai-codex-perf."))
    runner = repo_root / "benchmark" / "codex_code" / "src" / "runner.py"
    report_module = load_report_module(benchmark_root / "src" / "report.py")
    launcher_summary = results_dir / "launcher.json"
    launch_events: list[dict[str, str]] = []

    print(f"temp_root={temp_root}", flush=True)
    arm_reports: list[dict[str, object]] = []
    try:
        print("step: build Codex-enabled binary", flush=True)
        build_env = os.environ.copy()
        build_env.update(
            {
                "CARGO_HOME": str(host_cargo_home),
                "RUSTUP_HOME": str(host_rustup_home),
                "RUSTUP_TOOLCHAIN": "stable",
            }
        )
        subprocess.run(
            ["cargo", "+stable", "build", "--features", "codex", "--quiet"],
            cwd=repo_root,
            env=build_env,
            check=True,
        )
        print("step complete: Codex-enabled binary built", flush=True)
        for arm in args.arms:
            arm_root = temp_root / arm
            worktree = arm_root / "worktree"
            home = arm_root / "home"
            codex_home = home / ".codex"
            wrapper = arm_root / "agent-wrapper.sh"
            codex_report = arm_root / "report.json"
            runner_out = arm_root / "runner.json"
            arm_results = results_dir / arm
            arm_results.mkdir(parents=True, exist_ok=True)

            print(f"step: create {arm} disposable worktree", flush=True)
            launch_events.append({"step": f"create {arm} disposable worktree", "status": "started"})
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree), "HEAD"],
                check=True,
            )
            print(f"step complete: {arm} worktree={worktree}", flush=True)
            launch_events.append({"step": f"create {arm} disposable worktree", "status": "completed"})
            print(f"step: prepare {arm} Codex configuration", flush=True)
            prepare_codex_home(
                Path.home() / ".codex",
                codex_home,
                memories_enabled=arm != "lint-ai",
                mcp_feature_enabled=args.profile == "production" and arm != "codex-native",
            )
            launch_events.append({"step": f"prepare {arm} Codex configuration", "status": "completed"})

            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                # The disposable CODEX_HOME isolates persisted transcripts while allowing
                # Stop and SessionEnd to capture setup memory for the continuation.
                "codex_args=(exec --json --dangerously-bypass-approvals-and-sandbox "
                "--dangerously-bypass-hook-trust --cd \"$LINT_AI_BENCHMARK_WORKTREE\" "
                "--output-last-message \"$LINT_AI_BENCHMARK_RUN_DIR/${LINT_AI_BENCHMARK_PHASE}.last\" -)\n"
                "if [[ \"${LINT_AI_CODEX_ENABLE_MCP_FEATURE:-0}\" == \"1\" ]]; then\n"
                "  codex_args=(--enable mcp_2026_07_28 \"${codex_args[@]}\")\n"
                "fi\n"
                "printf '[codex-wrapper] phase=%s CODEX_HOME=%s command=codex' "
                "\"$LINT_AI_BENCHMARK_PHASE\" \"$CODEX_HOME\" >&2\n"
                "printf ' %q' \"${codex_args[@]}\" >&2\n"
                "printf '\\n' >&2\n"
                "exec codex \"${codex_args[@]}\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "CODEX_HOME": str(codex_home),
                    "CARGO_HOME": str(host_cargo_home),
                    "RUSTUP_HOME": str(host_rustup_home),
                    "PATH": f"{repo_root / 'target' / 'debug'}:{env.get('PATH', '')}",
                    "RUSTUP_TOOLCHAIN": "stable",
                    "LINT_AI_CODEX_ENABLE_MCP_FEATURE": (
                        "1" if args.profile == "production" and arm != "codex-native" else "0"
                    ),
                }
            )

            if arm != "codex-native":
                print("step: install Codex integration into lint-ai worktree", flush=True)
                subprocess.run(
                    [str(repo_root / "target" / "debug" / "lint-ai"), "--codex-install", str(worktree)],
                    cwd=repo_root,
                    env=env,
                    check=True,
                )
                restrict_to_claude_equivalent_hooks(codex_home)
                if args.profile == "hooks-only":
                    remove_lint_ai_mcp_server(codex_home)
                print(
                    "step complete: restricted Lint-AI to Claude-equivalent memory hooks "
                    + ("without the code-indexing MCP server" if args.profile == "hooks-only" else "with MCP enabled"),
                    flush=True,
                )

            if args.profile == "production" and arm != "codex-native":
                trace_path = arm_results / "mcp-trace.log"
                configure_mcp_diagnostics(
                    codex_home,
                    repo_root / "target" / "debug" / "lint-ai",
                    trace_path,
                )
            capture_codex_diagnostics(codex_home, env, arm_results / "client-diagnostics")
            print(
                f"step complete: captured {arm} Codex diagnostics in "
                f"{arm_results / 'client-diagnostics'}",
                flush=True,
            )

            print(f"step: run Codex benchmark suite ({arm})", flush=True)
            launch_events.append({"step": f"run Codex benchmark suite ({arm})", "status": "started"})
            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--benchmark-root",
                    str(benchmark_root),
                    "--repo-root",
                    str(worktree),
                    "--execute",
                    "--repetitions",
                    str(args.repetitions),
                    *[argument for scenario in args.scenario for argument in ("--scenario", scenario)],
                    "--results-dir",
                    str(arm_results),
                    "--timeout-scale",
                    str(args.timeout_scale),
                    "--agent-command",
                    str(wrapper),
                    "--arm",
                    arm,
                    "--report-out",
                    str(codex_report),
                    "--out",
                    str(runner_out),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            arm_payload = json.loads(runner_out.read_text(encoding="utf-8"))
            arm_reports.append(arm_payload)
            shutil.copy2(codex_report, arm_results / "report.json")
            shutil.copy2(runner_out, arm_results / "runner.json")
            print(f"step complete: Codex benchmark suite ({arm}) finished", flush=True)
            launch_events.append({"step": f"run Codex benchmark suite ({arm})", "status": "completed"})
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree)],
                check=False,
                capture_output=True,
                text=True,
            )

        scenarios = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((benchmark_root / "scenarios").glob("*.json"))
        ]
        expected_counts = {str(scenario["id"]): len(scenario.get("expected_facts", [])) for scenario in scenarios}
        combined = report_module.combine_reports(arm_reports, expected_counts)
        write_json(results_dir / "report.json", combined)
        print(f"results_dir={results_dir}", flush=True)
        print(f"preserved_report={results_dir / 'report.json'}", flush=True)
        write_json(
            launcher_summary,
            {
                "temp_root": str(temp_root),
                "results_dir": str(results_dir),
                "preserved_report": str(results_dir / "report.json"),
                "arms": list(args.arms),
                "profile": args.profile,
                "events": launch_events,
            },
        )
    finally:
        print("step: clean up temporary worktrees", flush=True)
        for worktree in temp_root.glob("*/worktree"):
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree)],
                check=False,
                capture_output=True,
                text=True,
            )
        print("step complete: cleanup attempted", flush=True)
        launch_events.append({"step": "clean up temporary worktrees", "status": "completed"})


if __name__ == "__main__":
    main()
