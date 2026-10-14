# Performance

This page summarizes measured performance across Membrick's agent
integrations. Results are grouped by provider because token accounting,
models, and task transcripts differ. Each comparison uses that provider's
native memory behavior as its baseline.

## Overview

Membrick automatically retrieves saved project context during an agent
session and adds it to the agent's context. The agent can use that context
without calling a Membrick tool. This automatic context feature uses agent
hooks. “Native memory” means Claude's or Codex's own memory feature. Tool-call
counts below refer to tools called by the agent.

!!! success "Claude Code: lower token and tool use with automatic Membrick context"

    - **Fresh input tokens:** 128 → 32 (75.0% fewer)
    - **Overall tokens:** 2,884,478 → 605,686 (79.0% fewer)
    - **Overall tokens per fresh input token:** 22,535 → 18,928 (16.0% fewer)
    - **Agent tool calls:** 65 → 14 (78.5% fewer)
    - **Claude response turns:** 131 → 36 (72.5% fewer)

    Recall was 3/3 in every run. This was one scenario repeated three times.
    Response turns count Claude's replies across those repetitions.

!!! success "Codex: lower token and tool use with automatic Membrick context"

    - **Fresh input tokens:** 46,035 → 36,731 (20.2% fewer)
    - **Overall tokens:** 313,066 → 196,102 (37.4% fewer)
    - **Overall tokens per fresh input token:** 6.80 → 5.34 (21.5% fewer)
    - **Agent tool calls:** 7 → 4 (42.9% fewer)
    - **Continuation time:** 27.95 s → 13.80 s (50.6% less)

    Recall stayed at 3/3. Both setups used one continuation round, so this run
    did not reduce the number of rounds. This was one run per setup.

## Claude Code token use

### Save tokens by giving Claude remembered project context

Each row below combines three runs. Parentheses show the reduction compared
with Claude native.

| Setup | Claude response turns | Tool calls | Fresh input tokens | Overall tokens | Overall tokens per fresh input token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude native memory only | 131 | 65 | 128 | 2,884,478 | 22,535 |
| Membrick hooks only | 36 (72.5% fewer) | 14 (78.5% fewer) | 32 (75.0% fewer) | 605,686 (79.0% fewer) | 18,928 (16.0% fewer) |
| Claude native memory + Membrick hooks | 33 (74.8% fewer) | 13 (80.0% fewer) | 32 (75.0% fewer) | 587,723 (79.6% fewer) | 18,366 (18.5% fewer) |

**Why Claude needed fewer turns and calls:** without Membrick, Claude had to
inspect the project to find and check the facts. Membrick supplied saved
project context automatically, so Claude needed fewer back-and-forth
responses and tool calls. Across the three runs, hooks returned 165 document
results and had no failures.

With Claude native memory and Membrick hooks enabled, Claude used 587,723
tokens, saving **2,296,755 tokens (79.6%)** against the native-only run. It
made 13 tool calls and found all three facts in each run. The detailed tables
below show how each run contributed to these totals.

On 2026-10-14, Claude Sonnet ran the `routing-decision-supersession`
scenario three times in each of three arms. The response-turn count is the
number of Claude assistant responses in the continuation transcripts. These
are conversation steps, not the three benchmark repetitions. Setup was
excluded. Each run used
a fresh worktree from the same fixture and repository revision. Claude's
normal tools remained available; the Membrick arms used lifecycle hooks only.
Each arm found all three task facts in each of its three runs. Membrick
reported no hook failures.

Total input is fresh input plus cache creation plus cached input reads.
Overall tokens add output tokens to total input. Tokens per fresh input token
is the arm's total overall tokens divided by its fresh input token count.

| Arm | Round | Fresh input | Cache creation | Cached input read | Output | Overall tokens | Tokens per fresh input | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude native | 1 | 52 | 65,232 | 1,251,777 | 9,877 | 1,326,938 | 25,518.0 | 3/3 |
| Claude native | 2 | 36 | 42,254 | 803,529 | 7,844 | 853,663 | 23,712.9 | 3/3 |
| Claude native | 3 | 40 | 68,697 | 621,323 | 13,817 | 703,877 | 17,596.9 | 3/3 |
| Membrick hooks only | 1 | 12 | 27,070 | 175,409 | 3,269 | 205,760 | 17,146.7 | 3/3 |
| Membrick hooks only | 2 | 10 | 31,823 | 143,076 | 3,001 | 177,910 | 17,791.0 | 3/3 |
| Membrick hooks only | 3 | 10 | 43,318 | 172,911 | 5,777 | 222,016 | 22,201.6 | 3/3 |
| Claude + Membrick | 1 | 10 | 32,474 | 153,524 | 2,275 | 188,283 | 18,828.3 | 3/3 |
| Claude + Membrick | 2 | 14 | 27,659 | 238,645 | 3,644 | 269,962 | 19,283.0 | 3/3 |
| Claude + Membrick | 3 | 8 | 14,561 | 113,087 | 1,822 | 129,478 | 16,184.8 | 3/3 |

¹ Total input is fresh input plus cache creation plus cached input read.
² Overall tokens are total input plus output. Cached input is counted each
time Claude reads it, so these are processed-token totals rather than unique
prompt size. ³ This is the aggregate overall token count divided by aggregate
fresh input tokens, not the mean of the three per-round ratios.

This is one task repeated three times per arm. The arms ran in separate
three-repetition batches, not an interleaved randomized experiment. Treat the
reductions as a diagnostic for this fixture and scenario, not as a general
performance claim. See the [Claude Code performance report](claude-code-performance-tests.md)
for the full methodology, hook activity, retrieved documents, and raw report
paths.

## Codex token use

### Codex used fewer tokens and tool calls with remembered project context

| Setup | Fresh input tokens | Overall tokens | Overall tokens per fresh input token | Agent tool calls | Continuation time | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Codex native memory only | 46,035 | 313,066 | 6.80 | 7 | 27.95 s | 3/3 |
| Membrick hooks | 36,731 (20.2% fewer) | 196,102 (37.4% fewer) | 5.34 (21.5% fewer) | 4 (3 fewer) | 13.80 s (50.6% less) | 3/3 |
| Membrick hooks + Codex native memory | 25,624 (44.3% fewer) | 137,333 (56.1% fewer) | 5.36 (21.2% fewer) | 2 (5 fewer) | 14.94 s (46.6% less) | 3/3 |

This was one run of `routing-decision-supersession` per setup on 2026-10-13.
Fresh input is calculated as reported input tokens minus cached input tokens.
Overall tokens are input plus output; tokens per fresh input token divides
overall tokens by fresh input tokens. Treat these figures as a single-run
diagnostic for this scenario, not a general performance claim. Token totals
use Codex's accounting and should not be compared directly with Claude's. See the
[Codex performance report](codex-performance-tests.md) for methodology and
the [machine-readable results](https://github.com/RooAGI/Membrick/blob/main/benchmark/codex_code/results/v0.5.0-routing-decision-supersession/report.json).

## Provider reports

- [Codex performance](codex-performance-tests.md)
- [Claude Code performance](claude-code-performance-tests.md)
- [Antigravity (AGY) performance](agy-performance-tests.md)
