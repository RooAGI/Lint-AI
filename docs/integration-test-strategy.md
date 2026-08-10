# Integration test strategy

Every provider integration is tested against the same capability categories. Provider-specific
hook names may differ, but an integration is not complete until every required category has a
test.

| Category | Claude Code | Codex | Gemini CLI | AGY |
| --- | --- | --- | --- | --- |
| Installation and idempotent configuration | Required | Required | Required | Required |
| Hook payload and protocol validation | Required | Required | Required | Required |
| Memory retrieval and context injection | Required | Required | Required | Required |
| Capture, redaction, persistence, and later retrieval | Required | Required | Required | Required |
| Recording, enable/disable controls, and status | Required | Required | Required | Required |
| MCP tools: validation, search, memory synchronization, and list memories | Required | Required | Required | Required |
| Per-provider memory, recording, and MCP-index isolation | Required | Required | Required | Required |
| Replay and metrics archive behavior | Required | Required | Required when replay is supported | Required when replay is supported |

## Test levels

- **Unit tests** validate parsers, redaction, payload limits, and provider-specific documents.
- **Contract tests** validate the lifecycle behavior above using temporary project roots and
  persisted indexes. These are the main CI gate for integrations.
- **MCP protocol tests** send JSON-RPC requests directly to the server implementation. They
  validate the public tool surface without requiring a provider binary.
- **External smoke tests** run manually against installed and authenticated provider CLIs. They
  validate the provider's current hook format, configuration loading, and one real session.

## Required commands

Run the integration contract suite with every integration enabled:

```bash
cargo test --features 'claude-code,codex,gemini-cli,agy' --lib
```

Run the core black-box suite separately:

```bash
cargo test --test integration
```

External smoke tests are intentionally not part of the default suite: they require a local
provider installation, authentication, and can change independently of this crate. A provider
integration change must include a deterministic contract test; external smoke validation is an
additional release check.
