# Antigravity CLI integration benchmark

This benchmark is the AGY counterpart to the Claude and Codex integration
benchmarks. It uses the shared integration metric contract and measures the
local MCP server, AGY hook latency, memory retrieval, durable session
recording, and—when AGY emits structured usage telemetry—continuation-turn
token usage.

Run the shared-harness AGY arms with:

```bash
python3 benchmark/agy/scripts/run_benchmark.py \
  --arms agy-native agy-lint-ai agy-lint-ai-disabled agy-mcp-only \
  --scenario index-store-segmented-routing \
  --repetitions 1 \
  --timeout-scale 0.5 \
  --results-dir benchmark/agy/results/shared
```

The launcher reuses the scenarios, validators, scoring, and report schema from
the shared runner in `benchmark/codex_code/src/runner.py`. AGY owns its scenario,
fixture, schema, parser, test, and result directories; AGY-specific parsing
lives in `benchmark/agy/src`.
The shared runner's PTY mode preserves the interactive AGY lifecycle and saves
the conversation ID between setup and continuation.

Available arms:

- `agy-native`: no Lint-AI installation.
- `agy-lint-ai`: Lint-AI memory, hooks, and recording enabled.
- `agy-lint-ai-disabled`: integration and recording enabled, memory injection disabled.
- `agy-mcp-only`: Lint-AI MCP installed with lifecycle hooks removed.

## Local protocol benchmark result

The local MCP and hook smoke track completed successfully on 2026-08-09 with
AGY 1.1.11 and the authenticated local installation:

| Measurement | Result |
| --- | ---: |
| MCP requests / responses | 6 / 6 |
| MCP roundtrip | 1,013.2 ms |
| `PreInvocation` hook | 16.6 ms |
| `PreToolUse` hook | 17.0 ms |
| `PostToolUse` hook | 6.8 ms |
| `Stop` hook | 6.0 ms |
| Recorded lifecycle events | 4 |

These numbers measure the Lint-AI MCP and hook path only. They do not measure
model quality, token savings, or AGY tool-selection behavior.

## Shared scenario benchmark result

The four-arm scenario run completed successfully on 2026-08-09 with AGY 1.1.11,
one repetition, and the `index-store-segmented-routing` scenario. Each arm
completed setup, continuation, and the segmented-index validator.

| Arm | Success | Recall | Setup | Continuation | Validator |
| --- | ---: | ---: | ---: | ---: | ---: |
| `agy-native` | 1/1 | 1/3 (33.3%) | 16,367.0 ms | 16,077.3 ms | 68,189.9 ms |
| `agy-lint-ai` | 1/1 | 1/3 (33.3%) | 16,365.9 ms | 16,070.6 ms | 54,880.0 ms |
| `agy-lint-ai-disabled` | 1/1 | 1/3 (33.3%) | 16,382.7 ms | 16,074.0 ms | 85,712.6 ms |
| `agy-mcp-only` | 1/1 | 1/3 (33.3%) | 16,313.3 ms | 16,087.9 ms | 75,354.1 ms |

This particular interactive AGY run did not emit structured token-usage data,
so token columns are reported as unavailable and must not be replaced with
elapsed time. The recall score is
the shared validator score, not a model-quality score: this run found one of
the three expected facts in each arm. The native arm means AGY without a
Lint-AI installation; it is not evidence that AGY provides a separate native
memory implementation.

AGY token accounting is accepted when the provider emits `usageMetadata` or
stream-JSON usage fields (`promptTokenCount`, `cachedContentTokenCount`,
`candidatesTokenCount`, and `totalTokenCount`). The AGY parser maps those into
the shared `parent_tokens`/`all_model_tokens` contract. The interactive terminal
renderer used for this run emitted neither, so this result is a lifecycle and
recall validation—not a token A/B result.

Detailed JSON reports are written below the selected results directory, one
`report.json` per arm, plus `comparison.json`.

The authenticated interactive track should use the same scenario prompts and
fresh AGY conversations as the Claude/Codex runners. AGY lifecycle hooks require
the interactive execution loop; `agy --print` usage is useful for token
accounting but is not, by itself, evidence that hooks ran.

## Authentication and quota troubleshooting

The benchmark reuses the authenticated AGY profile from the host account. It
temporarily writes the AGY settings, hook, and MCP configuration needed by each
arm, then restores all three files in a `finally` cleanup path. The hook and
recording A/B arms intentionally clear MCP configuration so MCP discovery does
not change their behavior; MCP is measured separately by `agy-mcp-only`.

The benchmark also passes `--disable-slash-commands` and prefixes each scenario
prompt with a no-tools instruction. This prevents AGY skills, filesystem
inspection, or MCP calls from turning a memory-recall benchmark into an
uncontrolled tool-use benchmark.

AGY account quota is an external prerequisite. When AGY returns `Individual
quota reached`, every arm can fail before producing usage telemetry, including
`agy-native`. Such a run is recorded as unavailable rather than as a Lint-AI
failure; retry after the account quota resets or use an account with available
quota. The result files remain useful for diagnosing the provider response, but
must not be published as a token A/B result.

For issue #32, this documents the supported AGY integration boundary and the
fail-open benchmark setup. For issue #33, the same shared harness and token
parser are used for Gemini CLI; provider-specific authentication or quota
failures are kept separate from Lint-AI retrieval and hook measurements.
