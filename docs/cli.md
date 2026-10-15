# Command line

Running `membrick` with no arguments displays help. Choose an explicit command
to inspect or process a project. A folder path by itself does not start a scan.

## Project commands

| Command | Purpose |
|---|---|
| `membrick status [PATH]` | Show persisted index stores under `PATH/.membrick`; does not scan project files |
| `membrick review [PATH]` | Run project lint checks with the search index prepared |
| `membrick analyze [PATH]` | Analyze project structure |
| `membrick index [PATH] [--redacted]` | Build or refresh an index |
| `membrick query <TEXT> [PATH] [--count N]` | Search project documents |
| `membrick context <TEXT> [PATH]` | Prepare LLM context from project documents |
| `membrick recall <TEXT> [PATH] [--count N]` | Search indexed documents and recorded memories |
| `membrick recall serve [PATH]` | Serve recall requests over newline-delimited JSON |
| `membrick inspect-index <PATH> [--view VIEW]` | Inspect one existing index |
| `membrick session promote <ID> [--provider PROVIDER] [--root PATH]` | Promote a recorded session into project memory |
| `membrick session replay <ID> [--provider PROVIDER] [--root PATH]` | Replay a recorded session, optionally toggling Membrick |
| `membrick export graph <FORMAT> [PATH]` | Export a document, chunk, or entity graph |
| `membrick export ontology [PATH]` | Export ontology data |

`PATH` defaults to the current folder only after a command is supplied. The
status command lists persisted stores and their document, record, publication,
revision, and segment counts. It does not scan project files.

Graph formats are `dot`, `json`, and `cytoscape-html`. Graph levels are `doc`,
`chunk`, and `entity`. Index inspection views are `summary`,
`source-documents`, `records`, and `segments`.

## Provider commands

```sh
membrick install <provider> [PATH]
membrick serve <provider> [PATH]
membrick verify <provider> [PATH]
membrick uninstall <provider> [PATH]
membrick hook <provider> <event> [PATH]
membrick statusline <claude|codex> [PATH]
```

Uninstall currently supports `claude` and `codex`. Provider-specific paths can
be overridden with `--provider-config`, `--provider-settings`,
`--provider-skills-dir`, `--provider-hooks-dir`, and `--provider-plugin-dir`.
Use `--force-skill` with install to replace a user-modified provider skill.
`statusline` currently supports `claude` and `codex`. The MCP verification
timeout is configurable with `--timeout`.

## Advanced configuration

Index extraction and chunking defaults are configured in the project config
file. Older single-dash command modes and low-level diagnostic switches remain
accepted for compatibility, but are omitted from normal help. Use the commands
above for supported CLI workflows. `--config PATH` selects a config file and
`--no-update-check` skips the interactive update check for one invocation.
