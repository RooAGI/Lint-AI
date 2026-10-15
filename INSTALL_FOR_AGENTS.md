# Install Membrick with an AI coding agent

This file is the canonical installation contract for Codex, Claude Code, Gemini CLI, AGY, ChatGPT Work, and other agents that can operate a local terminal with the user's permission.

## Goal

Install the official Membrick binary, configure it only for the agent the user requested, verify the MCP runtime, and report exactly what changed.

Do not modify unrelated project files. Do not disable existing MCP servers, hooks, instructions, or user configuration.

## 1. Identify the requested agent

Use the agent explicitly named by the user or the environment you are currently operating in.

Supported values:

| User / environment | Installer value | Install command | Verification command |
|---|---|---|---|
| Codex | `codex` | `membrick install codex` | `membrick verify codex` |
| Claude Code | `claude` | `membrick install claude` | `membrick verify claude` |
| Gemini CLI | `gemini` | `membrick install gemini` | `membrick verify gemini` |
| Antigravity CLI / AGY | `agy` | `membrick install agy` | `membrick verify agy` |

If the user has not identified an agent and the environment is ambiguous, ask which integration they want instead of changing several agent configurations.

## 2. Confirm the project directory

Use the repository or project the user asked you to configure. If you are already operating inside that repository, use its root.

Before installation, confirm the directory exists. Membrick's provider installers canonicalize the project path and will refuse a missing path.

## 3. Install on macOS or Linux

Download the installer to a temporary file before executing it:

```bash
installer="$(mktemp)"
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh -o "$installer"
sh "$installer" --agent codex --project "$PWD"
rm -f "$installer"
```

Replace `codex` with `claude`, `gemini`, or `agy` when appropriate.

The installer:

1. detects the supported OS/CPU,
2. downloads the matching official GitHub Release asset,
3. downloads its `.sha256` sidecar,
4. verifies the SHA-256 checksum before installation,
5. installs `membrick` under `~/.local/bin` by default,
6. runs the provider-specific configuration command, and
7. runs the provider-specific MCP verification command.

A custom binary directory can be supplied with `--install-dir` or `MEMBRICK_INSTALL_DIR`.

## 4. Install on Windows

Download the PowerShell installer first, then execute it:

```powershell
$installer = Join-Path $env:TEMP 'membrick-install.ps1'
Invoke-WebRequest -UseBasicParsing `
  -Uri 'https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.ps1' `
  -OutFile $installer
& $installer -Agent codex -Project (Get-Location).Path
Remove-Item $installer
```

Replace `codex` with `claude`, `gemini`, or `agy` when appropriate.

The Windows installer verifies the release SHA-256 checksum and adds its default install directory (`$HOME\.local\bin`) to the user's PATH when needed.

## 5. Source-install fallback

A newly merged installer may be newer than the most recent tagged release. If the latest release does not yet contain the checksum sidecar or the required platform asset, **do not install an unverified binary**.

If Rust/Cargo is already available, use the source-install fallback:

```bash
cargo install --git https://github.com/RooAGI/Membrick --features agent-integrations --force
```

Then configure and verify the requested provider. For Codex:

```bash
membrick install codex "$PWD"
membrick verify codex "$PWD"
```

Use the corresponding commands from the table above for other providers.

## 6. Success criteria

Do not report success only because a binary was downloaded.

Installation is successful when all of the following are true:

- `membrick --version` exits successfully,
- the requested provider installation command exits successfully,
- the provider MCP verification command exits successfully, and
- no unrelated configuration was overwritten.

The MCP verification launches the installed Membrick provider server, performs the MCP `initialize` handshake, calls `tools/list`, and requires a valid Membrick response.

## 7. Report back to the user

Report:

- the installed Membrick version,
- the binary path,
- the project path,
- the configured agent,
- whether MCP verification passed, and
- any PATH change or restart/reload required by the provider.

If a step fails, stop at that step and show the error. Do not hide a failed checksum, failed provider configuration, or failed MCP verification.

## Example prompt for Codex

```text
Install Membrick in this project. Follow INSTALL_FOR_AGENTS.md from RooAGI/Membrick. Use the official release, verify its checksum, configure the Codex integration, run the MCP verification, and tell me exactly what changed. Do not modify unrelated project files.
```

## Example prompt for ChatGPT Work

```text
Install Membrick in my current project by following RooAGI/Membrick's INSTALL_FOR_AGENTS.md. Configure it for the coding agent I use, verify the official release and MCP runtime, and report the installed version and files/configuration changed. Do not modify unrelated project files.
```
