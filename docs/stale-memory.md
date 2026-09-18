---
meta:
  - property: og:type
    content: website
  - property: og:title
    content: "Stale AI Agent Memory: Why Agents Resurface Outdated Decisions"
  - property: og:description
    content: "Your project says one thing on Monday and something different on Friday. Lint-AI keeps track of what is still true."
  - property: og:image
    content: https://rooagi.github.io/Lint-AI/assets/images/social/stale-memory.png
  - property: og:image:width
    content: "1200"
  - property: og:image:height
    content: "630"
  - property: og:url
    content: https://rooagi.github.io/Lint-AI/stale-memory/
  - name: twitter:card
    content: summary_large_image
---

# Stale AI Agent Memory: Why Agents Resurface Outdated Decisions

Your project says one thing on Monday and something different on Friday. Both versions stay relevant to the same question — and that is exactly the problem.

A retriever that only understands topical relevance will happily hand your agent an outdated decision and present it as the truth. Stale memory is not a retrieval failure. It is a state failure: the system remembered the text but lost track of what is *still true*.

If you run AI coding agents — Claude Code, Codex, Gemini CLI — against a living codebase, your agent's memory is full of these silent contradictions. Decisions get updated in docs and chat, but the old versions never stop being retrievable.

## The retry-policy walkthrough

Two documents. Same policy name, same semantic domain, no `supersedes:` metadata anywhere.

**Older — `decision-a.md`**

```markdown
# Gateway Retry Policy

Gateway timeout retry attempts: 5.
```

**Newer — `decision-b.md`**

```markdown
# Gateway Retry Policy

Gateway timeout retry attempts: 2.
```

Now ask the neutral question:

```
How many retry attempts should we use for gateway timeouts?
```

A topical retriever can return either document — or both — and leave the model to guess which value is current. **Lint-AI keeps `2` as the current answer and preserves `5` as superseded history.**

That difference matters anywhere project truth changes over time: architecture decisions, API contracts, configuration values, runbooks, ownership, terminology, and implementation plans.

## Why topical relevance does not guarantee current truth

Answering "what should we do?" from project history requires more than similarity search. A useful memory layer has to answer five questions at once:

1. **Relevance** — is this about the question?
2. **Recency** — when was this evidence true?
3. **Supersession** — did something newer replace it?
4. **Temporal intent** — is the asker asking about now, last Tuesday, or a historical state?
5. **Evidence** — where did the answer come from?

Ordinary retrieval answers the first and ignores the rest. That is how a decision from January keeps resurfacing in June as if nothing changed.

## How Lint-AI tracks what is still true

- **Time awareness.** Evidence is ranked with its timestamp, not just its text similarity.
- **Supersession tracking.** For simple configuration and value claims, Lint-AI can infer chronological replacement when documents establish the same semantic domain. Inferred supersession is domain-scoped, so unrelated settings that happen to share a field name (two different services' `timeout` values, for example) do not suppress one another. Explicit supersession metadata remains authoritative when you provide it. See [semantic supersession](semantic-supersession.md).
- **Evidence links.** Retrieved context carries its source and relationship evidence, so the agent — and the human reviewing it — can check where an answer came from.
- **History is preserved, not deleted.** A historical query can still retrieve superseded material *as history* rather than presenting it as current state.

## We ran it ourselves: the messy real data

Claims are cheap, so we built `lint-ai` (v0.2.1) from source and ran this exact scenario. (You don't need to repeat any of this — the release binary does the same thing. We went through the messy part to show our work.) The unglamorous part: the build machine had no Rust toolchain, so we installed one; the first `cargo build --release` died halfway because `/tmp` was a 512MB ramdisk; we moved the clone, rebuilt, and 7 minutes 25 seconds later had a working binary compiled from the repository's actual source.

With the older file dated January and the newer file dated June, the real query returned:

```json
{
  "result": {
    "doc_id": "decision-b.md",
    "semantic_status": "current"
  }
}
```

And the LLM-ready context — the text an agent would actually receive — contained the current value with the stale one excluded:

```json
{
  "current_context": {
    "doc_id": "decision-b.md",
    "text": "Gateway timeout retry attempts: 2."
  },
  "stale_value_in_context": false
}
```

One honest note: we set the file dates ourselves, because a fresh clone stamps both files with the same time. The repository's own [reproducible terminal demo](https://github.com/RooAGI/Lint-AI#reproducible-terminal-demo) does exactly the same thing to create its controlled scenario — run `bash scripts/run_real_terminal_demo.sh` to replay it.

The benchmark numbers get the same treatment. They are not one clean score; every result carries three independent labels — **scope** (the 500-question aggregate vs. the 133-question multi-session slice), **metric** (fractional recall vs. any-hit recall), and **cutoff** (`@5`, `@10`, `@20`) — and the documentation explicitly warns against comparing numbers across different labels. Messier to read than a single badge; harder to misuse.

The headline aggregate, on the heuristic release backend with no embedding vectors:

| Metric | Result |
| --- | --- |
| Fractional Recall@5 | **83.5%** |
| Any-hit Recall@10 | **95.6%** |
| Average query latency | **~1.88 ms** |

Single CPU. No GPU. The dataset downloader, benchmark binary, raw reports, and comparison workflows are all in the repository, so the numbers can be reproduced rather than taken on faith. See [benchmarks](benchmark.md).

## A real case from the benchmark data — including the miss

The 500-question evaluation set contains genuine stale-memory cases. Question `830ce83f` (type `knowledge-update`, asked June 13, 2023):

- **May 24 session:** the user's friend Rachel "recently moved to a new apartment in the city" (Chicago).
- **May 27 session:** Rachel "just moved back to the suburbs again."
- **Question:** "Where did Rachel move to after her recent relocation?"
- **Gold answer:** the suburbs.

We ran the project's own `haystack_scoped_benchmark` on this question. Both relevant sessions were retrieved in the top 5 (recall@5 = 1.0, MRR = 1.0 — a perfect score by the benchmark's metric, which counts either session as a hit). But the ranking tells the harder story:

1. `answer_0b1a0942_1` — the **stale** May 24 session
2. an unrelated session about a different Rachel (an author)
3. `answer_0b1a0942_2` — the **current** May 27 session

The stale session outranks the current one. On messy multi-session data, topical relevance can still beat recency: the May 24 session is literally about Rachel's relocation, so it scores highly on similarity even though its information is outdated. The system reliably retrieves the right neighborhood of evidence — that is what the 95.6% any-hit Recall@10 measures — but ordering stale-below-current *within* the results is the genuinely hard part, and this example shows it is not solved perfectly. The clean two-file demo works because chronology is the only signal; real conversations are noisier.

We include this because it is the point: a memory layer should be judged on real data, misses included. The stale-memory problem is real, the progress is measurable, and the remaining gap is visible in the project's own benchmark.

### Reproduce it

Everything below runs from a fresh clone of the repository:

```bash
# 1. Download the pinned, checksum-verified dataset (500 questions)
python3 benchmark/download_longmemeval_raw.py

# 2. Isolate the one question
python3 -c "
import json
d = json.load(open('benchmark/data/longmemeval_s_cleaned.json'))
q = [x for x in d if x['question_id'] == '830ce83f']
json.dump(q, open('/tmp/rachel-q.json', 'w'))
"

# 3. Build the benchmark binary
cargo build --release --bin haystack_scoped_benchmark

# 4. Run the real harness on it
./target/release/haystack_scoped_benchmark \
  --longmemeval /tmp/rachel-q.json \
  --out /tmp/rachel-result.json

# 5. Read the ranking
python3 -c "
import json
r = json.load(open('/tmp/rachel-result.json'))
q = r['per_query'][0]
print(q['retrieved_session_ids'][:5])
print('recall@5:', q['recall_at_k']['5'], '| MRR:', q['mrr'])
"
```

## Try it yourself — no building required

You don't need Rust, and you don't need to compile anything. Download the prebuilt binary for your platform (macOS, Linux, or Windows) from the [releases page](https://github.com/RooAGI/Lint-AI/releases), then:

```bash
# 1. Create two files that disagree, the way real project docs do
mkdir decisions && cd decisions
printf '# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.\n' > decision-a.md
sleep 2  # ensure the second file is newer, like a real update would be
printf '# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.\n' > decision-b.md

# 2. Ask the neutral question
lint-ai --query "How many retry attempts should we use for gateway timeouts?" .
```

You get `decision-b.md` back with `semantic_status: current`. The old file is still there as history — it just isn't presented as the truth anymore.

```bash
# 3. See exactly what your agent would receive
lint-ai --llm-context "How many retry attempts should we use for gateway timeouts?" .
```

The context contains the current value (**2**); the stale value (**5**) is excluded. That's the moment it clicks: your agent can't act on the outdated decision, because it never sees it as current.

Now point it at your own docs — especially the ones where you *know* a decision changed — and ask about them:

```bash
lint-ai --query "what is our current deploy process?" /path/to/repo/docs
```

Start with the [quickstart](quickstart.md), or read the [agent memory guide](agent-memory.md) for the bigger picture. If you want the full from-source reproduction with the benchmark data, the [reproduce-it steps above](#reproduce-it) have you covered.
