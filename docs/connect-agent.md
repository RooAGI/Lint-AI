# Connect Lint-AI to your AI agent

Lint-AI can give supported AI agents memory that carries across work sessions.
Install it once for a project and the agent can look up earlier decisions and
save useful outcomes as it works. Memories from connected agents are kept in
the project's `.lint-ai/memory` folder.

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
installer downloads and checksum-verifies the official Lint-AI release,
configures the selected agent, and checks that its MCP connection works.

```bash
# Codex
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh -s -- --agent codex --project .

# Claude Code
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh -s -- --agent claude --project .

# Gemini CLI
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh -s -- --agent gemini --project .

# Antigravity CLI
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh -s -- --agent agy --project .
```

Restart the agent if it was already open. Then ask a question about a past
project decision, or ask the agent to remember an outcome from the current
session. The agent uses Lint-AI's memory tools and lifecycle integration in
the background.

## OpenClaw and Hermes

OpenClaw and Hermes have their own setup steps because each host loads its
integrations differently. Their guides walk through the install and explain
what is automatic:

- [Set up OpenClaw](openclaw.md): install the MCP connection, memory skill,
  recall hook, and capture plugin, then restart the Gateway.
- [Set up Hermes](hermes.md): install the MCP connection and memory skill.
  The optional hooks plugin adds automatic recall and capture and requires the
  Lint-AI server to be running.

Both integrations use the same project memory folder, `.lint-ai/memory`, as
the other agents. The agents do not need to be switched to a different memory
provider.

## What happens after setup

- The agent can search project memory when it needs earlier context.
- Supported lifecycle integrations can recall relevant memories at the start
  of work and capture outcomes when work finishes.
- You can add or remove memories yourself by editing files in
  `.lint-ai/memory`.
- You can turn memory behavior on or off with the agent's Lint-AI controls;
  this does not delete saved memories.

See [Quickstart](quickstart.md) to try memory directly from the command line,
or [Agent integrations](agents.md) for the implementation details and shared
memory behavior.
