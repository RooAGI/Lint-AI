# Use Membrick with Antigravity CLI

Connect Membrick to a project so Antigravity CLI (AGY) can find earlier
decisions and save useful outcomes as work continues. Project memory is stored
in `.membrick/memory/` and can be shared with other connected agents.

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
membrick install agy .
```

This enables AGY to use Membrick in the project. It configures the MCP connection and hooks, installs the project memory skill, and preserves your other AGY settings.

On Windows, run this in PowerShell from the project folder (adjust the path if
you saved the executable elsewhere):

```powershell
.\membrick-windows-x86_64.exe install agy .
```

Restart AGY and open the project so it loads the new settings.

At `Stop`, Membrick stores every non-empty transcript record in source order
as an individual `SourceDocument`, including records that are not plain text.
A linked session context is reconciled through `MemoryService`. AGY supplies
the transcript path in its hook payload. Project-local transcripts need no
extra setting; for AGY's centralized transcript directory, configure it in the
project's `.membrick/settings.toml`:

```toml
[hooks]
agy_transcript_dir = "/path/to/.gemini/antigravity-cli/brain"
```

This replaces environment-based transcript path discovery.

## 3. Try it

Ask AGY to look up a decision from earlier work, or continue work in a new
session. Membrick can bring relevant project memory into the conversation and
save useful outcomes as work finishes.

## 4. If memory is not available

Restart AGY and make sure you opened the project where you installed Membrick.
You can check the MCP connection with:

```bash
membrick verify agy .
```

For details, see [all agent integrations](agents.md) and
[session recording](session-recording-design.md).
