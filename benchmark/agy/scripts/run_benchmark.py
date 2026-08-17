#!/usr/bin/env python3
"""Run AGY scenarios through the shared Claude/Codex benchmark harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def set_integration_state(binary: Path, project: Path, tool: str, env: dict[str, str], arguments: dict | None = None) -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    }
    completed = subprocess.run(
        [str(binary), "--agy-serve", str(project)],
        cwd=project,
        env=env,
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"AGY MCP tool {tool} failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--arms", nargs="+",
        choices=("agy-native", "agy-lint-ai", "agy-lint-ai-disabled", "agy-mcp-only"),
        default=("agy-native", "agy-lint-ai", "agy-lint-ai-disabled", "agy-mcp-only"),
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    benchmark_root = repo / "benchmark" / "agy"
    results = (args.results_dir or benchmark_root / "results" / "shared").resolve()
    results.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="lint-ai-agy-perf."))
    binary = repo / "target" / "debug" / "lint-ai"
    host_gemini = Path.home() / ".gemini"
    host_settings = host_gemini / "antigravity-cli" / "settings.json"
    host_hooks = host_gemini / "config" / "hooks.json"
    host_mcp = host_gemini / "config" / "mcp_config.json"
    saved_host_files = {
        path: path.read_bytes() if path.exists() else None
        for path in (host_settings, host_hooks, host_mcp)
    }
    host_settings.parent.mkdir(parents=True, exist_ok=True)
    settings = {}
    if saved_host_files[host_settings]:
        settings = json.loads(saved_host_files[host_settings].decode("utf-8"))
    permissions = settings.setdefault("permissions", {})
    allow = permissions.setdefault("allow", [])
    if "command(*)" not in allow:
        allow.append("command(*)")
    host_settings.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    runner = repo / "benchmark" / "codex_code" / "src" / "runner.py"
    try:
        subprocess.run(["cargo", "+stable", "build", "--features", "agy", "--quiet"], cwd=repo, check=True)
        reports = {}
        for arm in args.arms:
            arm_root = temp / arm
            worktree = arm_root / "worktree"
            worktree.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), "HEAD"], check=True, capture_output=True)
            arm_env = os.environ.copy()
            arm_env.update({
                # AGY authenticates through the host profile/keychain. Keep
                # worktrees isolated without creating a fresh OAuth profile.
                "HOME": str(Path.home()),
                "PATH": f"{repo / 'target' / 'debug'}:{arm_env.get('PATH', '')}",
            })
            if arm != "agy-native":
                subprocess.run([str(binary), "--agy-install", str(worktree)], cwd=repo, env=arm_env, check=True)
                if arm == "agy-mcp-only":
                    host_hooks.parent.mkdir(parents=True, exist_ok=True)
                    host_hooks.write_text("{}\n", encoding="utf-8")
                else:
                    # Keep the hook/recording A/B arms focused on lifecycle
                    # behavior. MCP exposure is measured by agy-mcp-only.
                    host_mcp.write_text(json.dumps({"mcpServers": {}}, indent=2) + "\n", encoding="utf-8")
            wrapper = arm_root / "agy-wrapper.sh"
            wrapper.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "prompt=$(cat)\n"
                "prompt=\"Benchmark instruction: answer the user directly from the conversation context. Do not call tools, inspect files, inspect configuration, or access MCP servers. $prompt\"\n"
                "conversation_file=\"${LINT_AI_BENCHMARK_RUN_DIR}/conversation.id\"\n"
                "resume_arg=\"\"\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = \"--resume\" ] && [ $# -ge 2 ]; then\n"
                "    shift\n"
                "    resume_arg=\"$1\"\n"
                "    shift\n"
                "    continue\n"
                "  fi\n"
                "  shift\n"
                "done\n"
                "set -- --print \"$prompt\" --disable-slash-commands --output-format stream-json --dangerously-skip-permissions\n"
                "if [ -n \"$resume_arg\" ]; then\n"
                "  set -- \"$@\" --conversation \"$resume_arg\"\n"
                "elif [ -s \"$conversation_file\" ]; then\n"
                "  conversation=$(tr -d '\\n' < \"$conversation_file\")\n"
                "  set -- \"$@\" --conversation \"$conversation\"\n"
                "fi\n"
                "exec agy \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            arm_results = results / arm
            command = [
                "python3", str(runner), "--benchmark-root", str(benchmark_root),
                "--repo-root", str(worktree), "--execute",
                "--metrics", "agy", "--metrics-root", str(Path(__file__).resolve().parents[1]), "--agent-command", str(wrapper), "--arm", arm,
                "--repetitions", str(args.repetitions), "--results-dir", str(arm_results),
                "--report-out", str(arm_results / "report.json"), "--out", str(arm_results / "runner.json"),
                "--timeout-scale", str(args.timeout_scale),
            ]
            for scenario in args.scenario:
                command.extend(["--scenario", scenario])
            env = arm_env
            if arm == "agy-lint-ai-disabled":
                set_integration_state(binary, worktree, "disable_lint_ai", env)
                set_integration_state(binary, worktree, "record_session", env, {"action": "start"})
            elif arm == "agy-lint-ai":
                set_integration_state(binary, worktree, "enable_lint_ai", env)
                set_integration_state(binary, worktree, "record_session", env, {"action": "start"})
            subprocess.run(command, cwd=repo, env=env, check=True)
            reports[arm] = json.loads((arm_results / "report.json").read_text(encoding="utf-8"))
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)], check=False, capture_output=True)
        (results / "comparison.json").write_text(json.dumps({"arms": list(reports), "reports": reports}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"arms": list(reports), "results_dir": str(results)}, indent=2))
    finally:
        for path, content in saved_host_files.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        for worktree in temp.glob("*/worktree"):
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)], check=False, capture_output=True)


if __name__ == "__main__":
    main()
