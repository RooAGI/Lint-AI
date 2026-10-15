# Use Membrick with Codex

Connect Membrick to a project so Codex can find earlier decisions and save
useful outcomes as work continues. Codex memory is stored with the project in
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
membrick --codex-install .
```

This enables Codex to use Membrick in the project. It configures the MCP connection and hooks, adds the project memory instructions, and preserves your other Codex settings.

Codex hooks use the transcript path supplied in the hook payload. If Codex
doesn't supply a usable path, Membrick can locate the session transcript using
the project setting below. Set `codex_transcript_dir` to the Codex sessions
directory; relative paths are resolved from the project root.

```toml
[hooks]
codex_transcript_dir = "/path/to/codex/sessions"
# Optional hook timing log; relative paths resolve from the project root.
timings_path = ".membrick/hook-timings.jsonl"
```

Transcript discovery does not read `CODEX_HOME` or `HOME` environment
variables.

On Windows, run this in PowerShell from the project folder (adjust the path if
you saved the executable elsewhere):

```powershell
.\membrick-windows-x86_64.exe --codex-install .
```

If Codex asks whether to trust the project's configuration or hooks, approve them so the integration can run.

## 3. Try it

Ask Codex to look up a decision from earlier work, or continue work in a new
session. Membrick can bring relevant project memory into the conversation and
save useful outcomes at session boundaries.

## 4. If memory is not available

Check that the project is trusted in Codex and start a new session. You can
check the MCP connection with:

```bash
membrick verify codex .
```

For details, see [all agent integrations](agents.md),
[Codex performance measurements](codex-performance-tests.md), and
[session recording](session-recording-design.md).

## Uninstall

From the project directory, run:

```bash
membrick uninstall codex .
```

This removes the Membrick MCP entries and hook commands, plus the Membrick
policy block from `AGENTS.md`. Other settings, `.membrick` data, and the
executable remain. Codex's hooks feature flag is left unchanged because other
Codex hooks may use it. Restart Codex to reload its settings.
