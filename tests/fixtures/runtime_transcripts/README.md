# Runtime transcript fixtures

The Codex, Claude Code, and AGY JSONL samples and the OpenClaw `before_reset` message payload are sanitized extracts from local runtime transcripts or hook-probe output. The provider event structure and ordering are retained; conversation text, identifiers, timestamps, and paths are replaced with test values. No unredacted transcripts are stored in this repository.

Hermes has no saved local session transcript, and RooAGI Runtime has no local runtime transcript directory. Their JSON files are sanitized callback payloads matching the integration hook contracts. These exercise each provider's actual input shape but are not represented as captured runtime sessions.

- `codex.jsonl`: Codex rollout `session_meta` and `response_item` records, including messages and a tool round-trip.
- `claude.jsonl`: Claude Code JSONL user/assistant messages with text, tool-use, and tool-result blocks.
- `agy.jsonl`: ordered AGY transcript event records.
- `openclaw.json`: OpenClaw typed-hook envelope containing session messages.
- `hermes.json`: Hermes `post_llm_call` callback fields.
- `roo_runtime.json`: RooAGI Runtime hook invocation envelope.
