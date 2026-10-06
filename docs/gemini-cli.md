# Use Lint-AI with Gemini CLI

Connect Lint-AI to a project so Gemini CLI can find earlier decisions and
save useful outcomes as work continues. Project memory is stored in
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
lint-ai --gemini-cli-install .
```

This enables Gemini CLI to use Lint-AI in the project. It configures the MCP connection and hooks and preserves your other Gemini settings.

On Windows, run this in PowerShell from the project folder (adjust the path if
you saved the executable elsewhere):

```powershell
.\lint-ai-windows-x86_64.exe --gemini-cli-install .
```

Restart Gemini CLI and open the project so it loads the new settings.

## 3. Try it

Ask Gemini CLI to look up a decision from earlier work, or continue work in a
new session. Lint-AI can bring relevant project memory into the conversation
and save useful outcomes as sessions finish.

## 4. If memory is not available

Restart Gemini CLI and make sure you opened the project where you installed
Lint-AI. You can check the MCP connection with:

```bash
lint-ai --gemini-cli-verify-mcp .
```

For details, see [all agent integrations](agents.md) and
[session recording](session-recording-design.md).
