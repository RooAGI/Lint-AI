# Use Lint-AI with Codex

Connect Lint-AI to a project so Codex can find earlier decisions and save
useful outcomes as work continues. Codex memory is stored with the project in
`.lint-ai/memory/` and can be shared with other connected agents.

## 1. Install Lint-AI

On macOS or Linux, download and verify the official release with:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh
```

On Windows, download `lint-ai-windows-x86_64.exe` from the
[release page](https://github.com/RooAGI/Lint-AI/releases/latest).

## 2. Enable Lint-AI for this project

Open a terminal in the project folder and run:

```bash
lint-ai --codex-install .
```

This enables Codex to use Lint-AI in the project. It configures the MCP connection and hooks, adds the project memory instructions, and preserves your other Codex settings.

On Windows, run this in PowerShell from the project folder (adjust the path if
you saved the executable elsewhere):

```powershell
.\lint-ai-windows-x86_64.exe --codex-install .
```

If Codex asks whether to trust the project's configuration or hooks, approve them so the integration can run.

## 3. Try it

Ask Codex to look up a decision from earlier work, or continue work in a new
session. Lint-AI can bring relevant project memory into the conversation and
save useful outcomes at session boundaries.

## 4. If memory is not available

Check that the project is trusted in Codex and start a new session. You can
check the MCP connection with:

```bash
lint-ai --codex-verify-mcp .
```

For details, see [all agent integrations](agents.md),
[Codex performance measurements](codex-performance-tests.md), and
[session recording](session-recording-design.md).
