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

Membrick only reports the update. The user chooses whether and when to install
it. Existing standalone CLI update behavior remains interactive and separate.
