# Update notifications

Membrick checks for a newer stable release when an installed MCP server starts
and its last check was more than 24 hours ago. The check runs in the background
and does not delay MCP startup or searches. It requests release metadata from
the Membrick GitHub releases API. It does not send workspace paths, queries, or
memory content.

The release version is cached under the current user's operating system cache
directory, keyed by the installed executable. This shares one check across
projects while keeping separate installations independent. Codex and Claude
Code display a cached notice through their next session start or user prompt
hook. Gemini CLI displays it at session start. A release is announced once per
installation.

The notice names the installed and available versions, links to the release
notes, and gives an update command for the package manager when it can be
identified:

| Installation | Suggested update |
| --- | --- |
| npm | `npm update -g membrick` for a global install, or `npm update membrick` in the project |
| pip | `python -m pip install --upgrade membrick` |
| pipx | `pipx upgrade membrick` |
| Cargo | `cargo install membrick --locked --force` |
| Official installer | Rerun the official installer for the same platform |

Every provider MCP server exposes the `update_membrick` action, including
Claude Code, Codex, Muse Code, Gemini CLI, AGY, Hermes, and OpenClaw. The update
notice offers this action with the exact available version. The Codex
installer configures the tool with `approval_mode = "prompt"`. The Claude Code
installer adds `mcp__membrick__update_membrick` to `permissions.ask` in
`~/.claude/settings.json`. Other clients control tool approval through their
own permission settings. The AGY installer adds
`mcp(membrick/update_membrick)` to `permissions.ask` in
`~/.gemini/antigravity-cli/settings.json`. AGY also asks for approval for
unconfigured MCP tools by default. Configure other clients to ask before
executing `update_membrick` if you want the same explicit approval prompt. MCP
itself does not standardize a yes/no approval dialog, so the action description
also instructs the agent to ask and wait for approval.

The action uses fixed package names and arguments, does not invoke a shell, and
refuses unknown installation methods. Local npm installs are left to the user
because updating them would modify the project's dependency state.
Official-installer updates verify the release asset checksum. Restart the
agent client after a successful update to reconnect the new MCP process.

This approval limits accidental or model-initiated updates, but package
manager updates still execute package installation code and replace software
in the current user's environment. Host approval settings can usually be
changed or bypassed, so keep `update_membrick` behind a confirmation prompt if
you want approval each time. Existing standalone CLI update behavior remains
interactive and separate.
