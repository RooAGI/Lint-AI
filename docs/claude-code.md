# Use Membrick with Claude Code

Connect Membrick to a project so Claude Code can find earlier decisions and
save useful outcomes as work continues. Project memory is stored in
`.membrick/memory/` and can be shared with other connected agents.

## 1. Install Membrick

On macOS or Linux, download and verify the official release with:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh
```

On Windows, download `membrick-windows-x86_64.exe` from the
[release page](https://github.com/RooAGI/Membrick/releases/latest).

## 2. Enable Membrick for this project

Open a terminal in the project folder and run:

```bash
membrick --claude-code-install .
```

This enables Claude Code to use Membrick in the project. It configures the MCP connection and hooks, installs the project memory skill, and preserves your other Claude Code settings.

On Windows, run this in PowerShell from the project folder (adjust the path if
you saved the executable elsewhere):

```powershell
.\membrick-windows-x86_64.exe --claude-code-install .
```

Restart Claude Code and open the project so it loads the new settings.

## 3. Try it

Ask Claude Code to look up a decision from earlier work, or continue work in a
new session. Membrick can bring relevant project memory into the conversation
and save useful outcomes as sessions finish.

## 4. If memory is not available

Restart Claude Code and make sure you opened the project where you installed
Membrick. You can check the MCP connection with:

```bash
membrick verify claude .
```

For details, see [all agent integrations](agents.md),
[Claude Code performance measurements](claude-code-performance-tests.md), and
[session recording](session-recording-design.md).

## Uninstall

From the project directory, run:

```bash
membrick uninstall claude .
```

This removes the Membrick MCP entry and hook commands from Claude's config,
plus the generated project memory skill when it has not been edited. Customized
skills are kept. Other settings, `.membrick` data, and the executable remain.
Restart Claude Code to reload its settings.
