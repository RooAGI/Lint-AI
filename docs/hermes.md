# Set up Lint-AI with Hermes

The basic setup lets Hermes search and manage Lint-AI memories when it needs
them. It uses the Lint-AI release binary and does not require a separate
server to stay running.

If you also want Hermes to recall memories automatically before each model
call and save turns in the background, follow the optional setup at the end.
That feature needs a small Lint-AI server to keep running on your computer.

Lint-AI works alongside Hermes' own notes and memory. It does not replace
Hermes' existing memory provider.

## Basic setup

### 1. Install Lint-AI

On macOS or Linux, install the latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/RooAGI/Lint-AI/main/scripts/install.sh | sh
```

For Windows or another download option, use the
[Lint-AI releases page](https://github.com/RooAGI/Lint-AI/releases/latest).

### 2. Connect Hermes to your project

Open a terminal in your project folder and run:

```bash
lint-ai --hermes-install .
```

This adds the Lint-AI connection to Hermes and installs a memory skill that
explains when Hermes should look up project memories. It leaves your other
Hermes settings and memory provider in place.

### 3. Restart Hermes and check the connection

Restart Hermes so it loads the new connection, then check it:

```bash
lint-ai --hermes-verify-mcp
hermes mcp test
```

The Hermes check should show Lint-AI as connected. Start a new conversation
in the project and ask Hermes to look up a decision from earlier work.

With this basic setup, Hermes can use Lint-AI's memory tools when needed. It
does not automatically save every turn. The optional setup below adds that.

## Optional: automatic recall and capture

Add this if you want Lint-AI to look up context before each model call and
save useful turns and tool results as you work. This option needs both the
Hermes plugin and the running Lint-AI server.

### 1. Get the plugin files

The plugin is included in the Lint-AI source repository. If you already have
the repository, use its path. Otherwise:

```bash
git clone --depth 1 https://github.com/RooAGI/Lint-AI.git
```

### 2. Install and enable the plugin

From the directory containing the `Lint-AI` folder, run:

```bash
hermes plugins install ./Lint-AI/src/integrations/hermes/plugin
hermes plugins enable lintai
```

If your repository is elsewhere, replace the path with its location.

### 3. Start the local Lint-AI server

In a separate terminal, run:

```bash
lint-ai serve
```

Keep this terminal open while using Hermes with automatic recall and capture.
The plugin connects to the local server at `http://127.0.0.1:8080`. If the
server is stopped, Hermes continues to work, but automatic recall and capture
pause until the server is available again.

### 4. Try it

Start a new Hermes conversation and work for a turn. Then ask Hermes to recall
something from that turn. Lint-AI can add relevant memories before later model
calls and save useful turns in the background.

## Where memories are stored

Project memories are stored under `.lint-ai/memory/`. Other connected agents
for the same project can use those memories too.

## If something does not work

- If `hermes mcp test` cannot connect, run `lint-ai --hermes-install .`
  again from the project folder, restart Hermes, and retry the check.
- If Hermes has the memory skill but no Lint-AI tools, check the MCP connection
  in `~/.hermes/config.yaml` and restart Hermes. The skill gives Hermes
  instructions; the MCP connection provides the callable tools.
- If automatic recall or capture is missing, confirm the plugin is enabled
  and `lint-ai serve` is still running.

For setup with another agent, see [Connect your AI agent](connect-agent.md).
