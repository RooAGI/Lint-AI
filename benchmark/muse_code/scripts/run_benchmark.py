#!/usr/bin/env python3
"""Run Muse Code scenarios through the shared benchmark orchestration.

Arms:
  muse-native   - no Lint-AI hooks, MCP, or memory policy
  muse-lint-ai  - `lint-ai --muse-install` plus session recording, per worktree

Each arm runs with an isolated HOME and XDG_CONFIG_HOME, mirroring the
Codex benchmark. The runner checks out a fresh git worktree per
repetition, so project-local install state (AGENTS.md policy, recording
state) is established by the wrapper once per worktree, guarded by a
marker file; global hook/MCP config lives in the arm's isolated
XDG_CONFIG_HOME. Muse has no validated resume API, so every benchmark
turn runs as a fresh `muse exec` process; the wrapper accepts (and
ignores) the runner's `--resume` argument.

Authentication for real (non-echo) runs: set LINT_AI_MUSE_API_KEY, which
the wrapper feeds to `muse exec --api-key-stdin`. Without it, the wrapper
relies on whatever credentials `muse` can find; the isolated HOME hides
the user's stored login, so a real benchmark needs the env key.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_report_module(path: Path):
    spec = importlib.util.spec_from_file_location("muse_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Runs inside the wrapper before `muse exec`, once per repetition worktree.
# The runner checks out a fresh worktree per repetition, so the install
# cannot be done once up front: AGENTS.md and the recording state live in
# the project. The one-time cost lands in the first (setup) phase timing.
MUSE_INSTALL_BLOCK = """\
# Project-local install: the runner uses a fresh worktree per repetition,
# so AGENTS.md policy and recording state must be established here.
marker="$LINT_AI_BENCHMARK_WORKTREE/.lint-ai/.bench-muse-installed"
if [ ! -f "$marker" ]; then
  lint-ai --muse-install --muse-config "$XDG_CONFIG_HOME/muse/settings.json" \\
    "$LINT_AI_BENCHMARK_WORKTREE" >&2
  # Hook events are only recorded while recording is enabled;
  # --muse-install leaves it off by default (opt-in capture).
  printf '%s\\n' '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"record_session","arguments":{"action":"start"}}}' \\
    | lint-ai --muse-serve "$LINT_AI_BENCHMARK_WORKTREE" >/dev/null
  mkdir -p "$(dirname "$marker")"
  touch "$marker"
fi
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("muse-native", "muse-lint-ai"),
        default=("muse-native", "muse-lint-ai"),
        help="Benchmark arms: no Lint-AI vs Lint-AI installed",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("LINT_AI_MUSE_PROVIDER", "meta"),
        help="Muse provider: meta or echo (echo needs no auth, for smoke tests)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LINT_AI_MUSE_MODEL", ""),
        help="Model id for non-echo providers",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    benchmark_root = Path(__file__).resolve().parents[1]
    results_dir = (args.results_dir or benchmark_root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="lint-ai-muse-perf."))
    binary = repo_root / "target" / "debug" / "lint-ai"
    runner = repo_root / "benchmark" / "codex_code" / "src" / "runner.py"
    report_module = load_report_module(repo_root / "benchmark" / "codex_code" / "src" / "report.py")
    host_home = Path.home()
    host_cargo_home = host_home / ".cargo"
    host_rustup_home = host_home / ".rustup"
    cargo_bin = host_cargo_home / "bin"
    base_path = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    arm_reports: list[dict] = []

    try:
        print("step: build Muse-enabled binary", flush=True)
        subprocess.run(
            ["cargo", "+stable", "build", "--features", "muse-code", "--quiet"],
            cwd=repo_root,
            env={**os.environ, "PATH": base_path},
            check=True,
        )

        for arm in args.arms:
            arm_root = temp_root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            home = arm_root / "home"
            config_home = home / ".config"
            settings_path = config_home / "muse" / "settings.json"
            arm_results = results_dir / arm
            arm_results.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(config_home),
                    "PATH": f"{repo_root / 'target' / 'debug'}:{base_path}",
                    # Validators run cargo in the isolated HOME; point them at
                    # the host toolchain instead of re-downloading per arm.
                    "CARGO_HOME": str(host_cargo_home),
                    "RUSTUP_HOME": str(host_rustup_home),
                    "RUSTUP_TOOLCHAIN": "stable",
                    "LINT_AI_MUSE_PROVIDER": args.provider,
                    "LINT_AI_MUSE_MODEL": args.model,
                }
            )
            api_key = os.environ.get("LINT_AI_MUSE_API_KEY")
            if api_key:
                env["LINT_AI_MUSE_API_KEY"] = api_key

            if arm != "muse-native":
                # Validate the install up front against a scratch project so a
                # broken installer fails fast; the wrapper repeats the
                # project-local install per repetition worktree.
                print(f"step: validate Muse integration install ({arm})", flush=True)
                scratch = arm_root / "install-scratch"
                scratch.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [str(binary), "--muse-install", "--muse-config", str(settings_path), str(scratch)],
                    cwd=repo_root,
                    env=env,
                    check=True,
                    capture_output=True,
                )
                installed = json.loads(settings_path.read_text(encoding="utf-8"))
                hook_events = installed.get("hooks", {})
                assert hook_events, f"{arm}: expected lint-ai hooks in {settings_path}"
                print(
                    f"step complete: install validated for events {sorted(hook_events)}",
                    flush=True,
                )

            install_block = MUSE_INSTALL_BLOCK if arm != "muse-native" else ""
            wrapper = arm_root / "muse-wrapper.sh"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "# No validated resume API: each turn is a fresh `muse exec` process.\n"
                "# The runner passes --resume <id> for multi-turn phases; accept and\n"
                "# ignore it so phases stay comparable.\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"${1:-}\" = \"--resume\" ]; then shift 2; else shift; fi\n"
                "done\n"
                + install_block +
                "prompt_file=\"$(mktemp)\"\n"
                "trap 'rm -f \"$prompt_file\"' EXIT\n"
                "cat > \"$prompt_file\"\n"
                "muse_args=(exec --json --provider \"${LINT_AI_MUSE_PROVIDER:-meta}\" --approval-mode never --prompt-file \"$prompt_file\")\n"
                "if [ -n \"${LINT_AI_MUSE_MODEL:-}\" ]; then\n"
                "  muse_args+=(--model \"$LINT_AI_MUSE_MODEL\")\n"
                "fi\n"
                "if [ -n \"${LINT_AI_MUSE_API_KEY:-}\" ]; then\n"
                "  printf '%s' \"$LINT_AI_MUSE_API_KEY\" | muse \"${muse_args[@]}\" --api-key-stdin\n"
                "else\n"
                "  muse \"${muse_args[@]}\"\n"
                "fi\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

            print(f"step: run Muse benchmark suite ({arm})", flush=True)
            runner_out = arm_results / "runner.json"
            report_out = arm_results / "report.json"
            command = [
                sys.executable,
                str(runner),
                "--benchmark-root",
                str(benchmark_root),
                "--repo-root",
                str(repo_root),
                "--execute",
                "--repetitions",
                str(args.repetitions),
                "--agent-command",
                str(wrapper),
                "--arm",
                arm,
                "--metrics",
                "muse",
                "--metrics-root",
                str(benchmark_root),
                "--results-dir",
                str(arm_results),
                "--timeout-scale",
                str(args.timeout_scale),
                "--report-out",
                str(report_out),
                "--out",
                str(runner_out),
            ]
            for scenario in args.scenario:
                command.extend(["--scenario", scenario])
            subprocess.run(command, cwd=repo_root, env=env, check=True)
            arm_reports.append(json.loads(runner_out.read_text(encoding="utf-8")))
            print(f"step complete: Muse benchmark suite ({arm}) finished", flush=True)

        scenarios = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((benchmark_root / "scenarios").glob("*.json"))
        ]
        expected_counts = {
            str(scenario["id"]): len(scenario.get("expected_facts", [])) for scenario in scenarios
        }
        combined = report_module.combine_reports(arm_reports, expected_counts)
        write_json(results_dir / "report.json", combined)
        print(f"results_dir={results_dir}", flush=True)
        print(f"preserved_report={results_dir / 'report.json'}", flush=True)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
