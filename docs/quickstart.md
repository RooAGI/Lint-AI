# Get started

Download Membrick, save a few things you want your assistant to remember, and
ask questions in plain language. You do not need to build the project or learn
special memory syntax.

## 1. Download Membrick

Get the latest published version from the
[Membrick releases page](https://github.com/RooAGI/Membrick/releases/latest).
Choose the download for your computer:

| Computer | Release download |
|---|---|
| Mac with Apple silicon | `membrick-macos-aarch64` |
| Mac with Intel | `membrick-macos-x86_64` |
| Linux, 64-bit | `membrick-linux-x86_64` |
| Windows, 64-bit | `membrick-windows-x86_64.exe` |

On macOS or Linux, you can instead run the installer, which downloads the
official release and checks its SHA-256 checksum:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Membrick/main/scripts/install.sh | sh
```

Installations made by the official installer check for updates once per day
when you run Membrick interactively. Membrick asks before downloading and
replacing its executable. MCP and hook processes do not check for updates.
Pass `--no-update-check` to skip the check for one invocation. Binaries
installed with Cargo or a package manager should be updated with that tool.

On Windows, download `membrick-windows-x86_64.exe` from the releases page. In
PowerShell, run it from the download folder; in the commands below, use
`.\membrick-windows-x86_64.exe` instead of `membrick`. You can create, edit, and
remove memory files with Notepad or File Explorer.

Open a new terminal if needed, then confirm it is ready:

```bash
membrick --version
```

## 2. Add something to remember

Create a folder for your memories. Save each thing you want to remember in a
plain text or Markdown file. Write naturally; Membrick searches the words and
ideas in the file.

```bash
mkdir -p ~/my-assistant-memory
cat > ~/my-assistant-memory/preferences.md <<'EOF'
I prefer concise answers with a short summary first.
When explaining code, include a small example.
EOF
```

You can add another memory by creating another file or editing an existing one.
These files stay on your computer unless you choose to connect them to another
service.

## 3. Ask a question

Ask Membrick a question about the memories in that folder:

```bash
membrick query "How do I prefer answers to be written?" ~/my-assistant-memory
```

To print a short answer context that you can paste into an AI assistant, use
the `context` command:

```bash
membrick context "How do I prefer answers to be written?" ~/my-assistant-memory
```

Ask a new question whenever you want to look something up. To show more
matches, add `--count 10`.

To see which indexes already exist in the current project, run
`membrick status .`. This only inspects existing index stores; it does not scan
project files.

## 4. Change or forget a memory

Edit a file to change what it says. Run your question again to search the
updated memory. To forget a memory, remove its file:

```bash
rm ~/my-assistant-memory/preferences.md
```

You can also keep a memory by moving its file outside the memory folder. Each
file is one piece of information, so it is easy to update or remove just that
memory.

## Use Membrick inside an AI tool

Membrick connects to Codex, Claude Code, Gemini CLI, Antigravity CLI, OpenClaw,
Hermes, Muse Code, and RooAGI AgentFlow. Choose your agent in the
[agent setup guide](connect-agent.md) for the right installation steps.

## More ways to use Membrick

The command line is the simplest way to try personal memory. Developers can
connect applications through the [HTTP API](server.md), [MCP](mcp.md),
[Python](python-migration-0.3.0.md), [Node.js](node.md), or
[Rust](memory-service-api.md) interfaces.
