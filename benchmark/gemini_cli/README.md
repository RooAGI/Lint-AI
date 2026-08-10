# Gemini CLI integration benchmark

This benchmark measures the local Lint-AI integration path for Gemini CLI. It
is intentionally separate from a model-quality or token-usage benchmark.

The smoke track measures:

- Gemini MCP server startup;
- MCP `initialize` and `tools/list` latency;
- warm `info` and `search` tool latency;
- Gemini hook latency for `SessionStart` and `BeforeAgent`; and
- whether hook events were durably recorded.

Run it from the repository root:

```bash
python3 benchmark/gemini_cli/scripts/run_benchmark.py
```

Write the report somewhere else with:

```bash
python3 benchmark/gemini_cli/scripts/run_benchmark.py \
  --results-dir benchmark/gemini_cli/results/local-smoke
```

The output is diagnostic integration evidence. It does not measure Gemini
model tokens, task quality, or agent tool selection. Those require a controlled
authenticated Gemini CLI session and should be reported separately.
