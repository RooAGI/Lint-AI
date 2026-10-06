# Get started

Download Lint-AI, save a few things you want your assistant to remember, and
ask questions in plain language. You do not need to build the project or learn
special memory syntax.

## 1. Download Lint-AI

Get the latest published version from the
[Lint-AI releases page](https://github.com/RooAGI/Lint-AI/releases/latest).
Choose the download for your computer:

| Computer | Release download |
|---|---|
| Mac with Apple silicon | `lint-ai-macos-aarch64` |
| Mac with Intel | `lint-ai-macos-x86_64` |
| Linux, 64-bit | `lint-ai-linux-x86_64` |
| Windows, 64-bit | `lint-ai-windows-x86_64.exe` |

On macOS or Linux, you can instead run the installer, which downloads the
official release and checks its SHA-256 checksum:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh
```

On Windows, download `lint-ai-windows-x86_64.exe` from the releases page. In
PowerShell, run it from the download folder; in the commands below, use
`.\lint-ai-windows-x86_64.exe` instead of `lint-ai`. You can create, edit, and
remove memory files with Notepad or File Explorer.

Open a new terminal if needed, then confirm it is ready:

```bash
lint-ai --version
```

## 2. Add something to remember

Create a folder for your memories. Save each thing you want to remember in a
plain text or Markdown file. Write naturally; Lint-AI searches the words and
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

Ask Lint-AI a question about the memories in that folder:

```bash
lint-ai --query "How do I prefer answers to be written?" ~/my-assistant-memory
```

To print a short answer context that you can paste into an AI assistant, use
`--llm-context`:

```bash
lint-ai --llm-context "How do I prefer answers to be written?" ~/my-assistant-memory
```

Ask a new question whenever you want to look something up. To show more
matches, add `--result-count 10`.

## 4. Change or forget a memory

Edit a file to change what it says. Run your question again to search the
updated memory. To forget a memory, remove its file:

```bash
rm ~/my-assistant-memory/preferences.md
```

You can also keep a memory by moving its file outside the memory folder. Each
file is one piece of information, so it is easy to update or remove just that
memory.

## Use Lint-AI inside an AI tool

Lint-AI connects to Codex, Claude Code, Gemini CLI, Antigravity CLI, OpenClaw,
Hermes, Muse Code, and RooAGI AgentFlow. Choose your agent in the
[agent setup guide](connect-agent.md) for the right installation steps.

## More ways to use Lint-AI

The command line is the simplest way to try personal memory. Developers can
connect applications through the [HTTP API](server.md), [MCP](mcp.md),
[Python](python-migration-0.3.0.md), or [Rust](memory-service-api.md)
interfaces.
