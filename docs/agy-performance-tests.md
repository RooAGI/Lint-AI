# Antigravity (AGY) Integration Benchmark

AGY has its own benchmark track for native, full Lint-AI, hooks-disabled, and
MCP-only arms. It covers memory injection, lifecycle recording, supersession,
oversized transcripts, negative controls, and MCP behavior.

The complete scenarios, parser, schemas, and commands are in the
[`benchmark/agy/README.md`](https://github.com/RooAGI/Lint-AI/blob/main/benchmark/agy/README.md).

Typical run:

```bash
python3 benchmark/agy/scripts/run_benchmark.py \
  --arms agy-native agy-lint-ai agy-lint-ai-disabled agy-mcp-only \
  --scenario index-store-segmented-routing \
  --repetitions 1 --timeout-scale 0.5 \
  --results-dir benchmark/agy/results/shared
```

The launcher preserves the host AGY profile and restores temporary hook and
MCP configuration when it exits.
