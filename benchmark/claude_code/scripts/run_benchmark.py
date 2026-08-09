#!/usr/bin/env python3
"""Run Claude scenarios through the shared benchmark orchestration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def remove_lint_ai_hooks(settings_path: Path) -> None:
    """Keep unrelated Claude hooks while disabling only Lint-AI hooks."""
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    hooks_by_event = settings.get("hooks", {})
    if not isinstance(hooks_by_event, dict):
        return
    for event_name, entries in list(hooks_by_event.items()):
        if not isinstance(entries, list):
            continue
        retained_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                retained_entries.append(entry)
                continue
            nested_hooks = entry.get("hooks")
            if not isinstance(nested_hooks, list):
                retained_entries.append(entry)
                continue
            nested_hooks = [
                hook
                for hook in nested_hooks
                if not (
                    isinstance(hook, dict)
                    and "--claude-code-hook" in str(hook.get("command", ""))
                )
            ]
            if nested_hooks:
                updated = dict(entry)
                updated["hooks"] = nested_hooks
                retained_entries.append(updated)
        hooks_by_event[event_name] = retained_entries
    write_json(settings_path, settings)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("hooks-only", "production", "mcp-only"),
        default="hooks-only",
        help="Measure hooks, the full integration, or MCP without Lint-AI hooks",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("claude-native", "claude-lint-ai", "claude-both"),
        default=("claude-native", "claude-lint-ai", "claude-both"),
        help="Benchmark arms: native memory, Lint-AI only, or both (default: all)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LINT_AI_CLAUDE_MODEL", "claude-sonnet-5"),
        help="Claude model used by both benchmark arms",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    benchmark_root = Path(__file__).resolve().parents[1]
    results_dir = (args.results_dir or benchmark_root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="lint-ai-claude-perf."))
    binary = repo_root / "target" / "debug" / "lint-ai"
    runner = repo_root / "benchmark" / "codex_code" / "src" / "runner.py"
    cargo_bin = Path.home() / ".cargo" / "bin"
    base_path = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    reports: dict[str, dict[str, object]] = {}
    try:
        print("step: build Claude-enabled binary", flush=True)
        subprocess.run(
            ["cargo", "+stable", "build", "--features", "claude-code", "--quiet"],
            cwd=repo_root,
            env={**os.environ, "PATH": base_path},
            check=True,
        )
        # Read auth fields from the user's global settings so isolated arms can authenticate.
        user_settings_path = Path.home() / ".claude" / "settings.json"
        user_settings: dict = {}
        if user_settings_path.exists():
            try:
                user_settings = json.loads(user_settings_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        auth_fields: dict = {}
        if "apiKeyHelper" in user_settings:
            auth_fields["apiKeyHelper"] = user_settings["apiKeyHelper"]
        if "apiKeyHelperTTLMs" in user_settings:
            auth_fields["apiKeyHelperTTLMs"] = user_settings["apiKeyHelperTTLMs"]
        if "sandbox" in user_settings:
            auth_fields["sandbox"] = user_settings["sandbox"]
        # Carry through auth-related env vars (e.g. Apple TTL and custom headers).
        auth_env_keys = {
            "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
            "ANTHROPIC_CUSTOM_HEADERS",
            "CLAUDE_CODE_DISABLE_SANDBOX",
        }
        user_env = user_settings.get("env", {})
        if isinstance(user_env, dict):
            auth_fields.setdefault("env", {})
            for key in auth_env_keys:
                if key in user_env:
                    auth_fields["env"][key] = user_env[key]
        if not auth_fields.get("env"):
            auth_fields.pop("env", None)

        for arm in args.arms:
            arm_root = temp_root / arm
            claude_home = Path.home()
            claude_config_dir = arm_root / "claude-config"
            settings_path = claude_config_dir / "settings.json"
            claude_config = arm_root / "claude.json"
            mcp_config_path = arm_root / "mcp.json"
            auto_memory_dir = arm_root / "auto-memory"
            claude_config_dir.mkdir(parents=True, exist_ok=True)
            write_json(claude_config, {})
            arm_settings = {
                **(
                    {"autoMemoryDirectory": str(auto_memory_dir)}
                    if arm in ("claude-native", "claude-both")
                    else {"autoMemoryEnabled": False}
                ),
                **auth_fields,
            }
            write_json(settings_path, arm_settings)
            env = os.environ.copy()
            # Propagate sandbox and auth env vars from user settings into the process env.
            # sandbox_apply runs before Claude Code reads settings, so these must be real env vars.
            for key in auth_env_keys:
                if key in user_env:
                    env[key] = user_env[key]
            env.update({
                "HOME": str(claude_home),
                "CLAUDE_CONFIG_DIR": str(claude_config_dir),
                "LINT_AI_CLAUDE_SETTINGS": str(settings_path),
                "LINT_AI_CLAUDE_MODEL": args.model,
                "LINT_AI_CLAUDE_MCP_CONFIG": str(mcp_config_path),
                "LINT_AI_BENCHMARK_SKILL_PATH": str(
                    repo_root / "src" / "integrations" / "claude_code" / "skill.md"
                ),
                "PATH": f"{repo_root / 'target' / 'debug'}:{base_path}",
            })
            env.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)
            env.pop("CLAUDE_CODE_SHELL_PREFIX", None)  # Apple sandbox wrapper causes sandbox_apply failure
            if arm == "claude-lint-ai":
                env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
            if arm != "claude-native":
                subprocess.run(
                    [
                        str(binary),
                        "--claude-code-install",
                        str(repo_root),
                        "--claude-code-config",
                        str(claude_config),
                        "--claude-code-settings",
                        str(settings_path),
                    ],
                    cwd=repo_root,
                    env=env,
                    check=True,
                )
                if args.profile == "mcp-only":
                    remove_lint_ai_hooks(settings_path)
                config = json.loads(claude_config.read_text(encoding="utf-8"))
                if args.profile == "hooks-only":
                    # Hooks-only measurements must not include MCP startup or code indexing.
                    config.get("mcpServers", {}).pop("lint-ai", None)
                elif "lint-ai" in config.get("mcpServers", {}):
                    # MCP child processes may not inherit the benchmark PATH.
                    config["mcpServers"]["lint-ai"]["command"] = str(binary)
                write_json(claude_config, config)
            write_json(
                mcp_config_path,
                json.loads(claude_config.read_text(encoding="utf-8"))
                if args.profile in ("production", "mcp-only") and arm != "claude-native"
                else {"mcpServers": {}},
            )

            wrapper = arm_root / "claude-wrapper.sh"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"exec claude --print --verbose --output-format stream-json --include-hook-events "
                "--model \"$LINT_AI_CLAUDE_MODEL\" "
                "--dangerously-skip-permissions "
                "--permission-mode bypassPermissions "
                f"--strict-mcp-config --mcp-config {shlex.quote(str(mcp_config_path))} "
                "-- -\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            arm_results = results_dir / arm
            command = [
                sys.executable,
                str(runner),
                "--benchmark-root",
                str(benchmark_root),
                "--repo-root",
                str(repo_root),
                "--execute",
                "--agent-command",
                str(wrapper),
                "--arm",
                arm,
                "--metrics",
                "claude",
                "--repetitions",
                str(args.repetitions),
                "--results-dir",
                str(arm_results),
                "--report-out",
                str(arm_results / "report.json"),
                "--out",
                str(arm_results / "runner.json"),
            ]
            for scenario in args.scenario:
                command.extend(["--scenario", scenario])
            print(f"step: run {arm}", flush=True)
            subprocess.run(command, cwd=repo_root, env=env, check=True)
            reports[arm] = json.loads((arm_results / "report.json").read_text(encoding="utf-8"))

        final_results = {
            arm: report["summary"]["final_results"] for arm, report in reports.items()
        }
        native = final_results.get("claude-native")
        comparisons = {}
        if native:
            native_latency = native.get("average_interaction_round_latency_ms")
            for arm, final in final_results.items():
                if arm == "claude-native":
                    continue
                latency = final.get("average_interaction_round_latency_ms")
                comparisons[arm] = {
                    "native_latency_ms": native_latency,
                    "arm_latency_ms": latency,
                    "latency_delta_ms": (
                        latency - native_latency
                        if isinstance(latency, (int, float))
                        and isinstance(native_latency, (int, float))
                        else None
                    ),
                    "native_recall": native["arm_summaries"]["claude-native"]["recall"],
                    "arm_recall": final["arm_summaries"][arm]["recall"],
                }
        write_json(
            results_dir / "comparison.json",
            {
                "scenario": args.scenario,
                "profile": args.profile,
                "reports": reports,
                "arm_summaries": final_results,
                "continuation_latency_ms": {
                    arm: final["average_interaction_round_latency_ms"]
                    for arm, final in final_results.items()
                },
                "comparison_vs_claude_native": comparisons,
            },
        )
        print(f"results_dir={results_dir}", flush=True)
        print(f"comparison_report={results_dir / 'comparison.json'}", flush=True)
    finally:
        for worktree in temp_root.glob("*/worktree"):
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree)],
                check=False,
                capture_output=True,
            )


if __name__ == "__main__":
    main()
