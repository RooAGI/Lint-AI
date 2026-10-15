# Connect Membrick to your AI agent

Membrick can give supported AI agents memory that carries across work sessions.
Install it once for a project and the agent can look up earlier decisions and
save useful outcomes as it works. Memories from connected agents are kept in
the project's `.membrick/memory` folder.

## Choose your agent

| Agent | Setup |
|---|---|
| Codex | Run the installer below with `--agent codex`. |
| Claude Code | Run the installer below with `--agent claude`. |
| Gemini CLI | Run the installer below with `--agent gemini`. |
| Antigravity CLI | Run the installer below with `--agent agy`. |
| OpenClaw | Follow the [OpenClaw setup](openclaw.md). |
| Hermes | Follow the [Hermes setup](hermes.md). |
| Muse Code | Follow the [Muse Code setup](muse-code.md). |
| RooAGI AgentFlow | Follow the [RooAGI AgentFlow setup](roo-runtime.md). |

## Install for Codex, Claude Code, Gemini CLI, or Antigravity

Open a terminal in the project folder and run the command for your agent. The
installer downloads and checksum-verifies the official Membrick release,
configures the selected agent, and checks that its MCP connection works.

```bash
# Codex
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh -s -- --agent codex --project .

# Claude Code
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh -s -- --agent claude --project .

# Gemini CLI
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh -s -- --agent gemini --project .

# Antigravity CLI
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh -s -- --agent agy --project .
```

Restart the agent if it was already open. Then ask a question about a past
project decision, or ask the agent to remember an outcome from the current
session. The agent uses Membrick's memory tools and lifecycle integration in
the background.

If you already installed the `membrick` executable, use the shared provider
commands for setup and maintenance:

```bash
membrick install codex .
membrick install claude .
membrick install gemini .
membrick install agy .
membrick serve codex .
membrick verify codex .
membrick uninstall codex .
```

Use `membrick hook <provider> <event> [PATH]` to run a lifecycle hook, such as
`membrick hook claude session-start .`. `setup` remains an alias for `install`.
Older provider-specific flags remain accepted for installed agent configs and
existing scripts.

Advanced path overrides use `--provider-config`, `--provider-settings`,
`--provider-skills-dir`, `--provider-hooks-dir`, and
`--provider-plugin-dir`. Use `--force-skill` with `install` when you
intentionally want to replace a customized skill.

## Uninstall Claude Code or Codex

From the project directory, remove the Membrick integration with:

```bash
membrick uninstall claude .
membrick uninstall codex .
```

Uninstall removes only Membrick or legacy Lint-AI MCP entries, hooks, and
project memory guidance for that provider. It preserves unrelated agent
settings, customized Claude skills, the project's `.membrick` memory data, and
the Membrick executable. Restart the agent after uninstalling. The Codex hook
feature flag is left unchanged because other Codex hooks may use it.

## Migrate from Lint-AI

Existing Lint-AI users can install Membrick and update one agent connection in
one step. From the project directory, run:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/migrate-lint-ai.sh | sh -s -- --agent codex --project .
```

Replace `codex` with `claude`, `gemini`, or `agy` for another supported agent.
The migration installs Membrick, updates that agent's configuration while
preserving unrelated settings, and verifies the MCP connection. During agent
setup, existing `.lint-ai` project state is moved to `.membrick` with its
contents preserved. Membrick also checks for legacy state when it first accesses
the project. If both directories exist, migration merges non-conflicting entries
and moves conflicting legacy entries into `.membrick/migration-archive`, without
overwriting current state. It removes the project-root `.lint-ai` directory when
migration completes. The old `lint-ai` executable is left installed; once you have
confirmed Membrick works, remove it using the package manager that installed
it.

## OpenClaw and Hermes

OpenClaw and Hermes have their own setup steps because each host loads its
integrations differently. Their guides walk through the install and explain
what is automatic:

- [Set up OpenClaw](openclaw.md): install the MCP connection, memory skill,
  recall hook, and capture plugin, then restart the Gateway.
- [Set up Hermes](hermes.md): install the MCP connection and memory skill.
  The optional hooks plugin adds automatic recall and capture and requires the
  Membrick server to be running.

Both integrations use the same project memory folder, `.membrick/memory`, as
the other agents. The agents do not need to be switched to a different memory
provider.

## What happens after setup

- The agent can search project memory when it needs earlier context.
- Supported lifecycle integrations can recall relevant memories at the start
  of work and capture outcomes when work finishes.
- You can add or remove memories yourself by editing files in
  `.membrick/memory`.
- You can turn memory behavior on or off with the agent's Membrick controls;
  this does not delete saved memories.

See [Quickstart](quickstart.md) to try memory directly from the command line,
or [Agent integrations](agents.md) for the implementation details and shared
memory behavior.
