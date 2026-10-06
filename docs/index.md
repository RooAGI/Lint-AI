---
hide:
  - navigation
  - toc
---

<div class="hero-stack">
<section class="hero">
  <div class="hero__eyebrow">THE MEMORY LAYER FOR AI AGENTS</div>
  <h1>Give your AI a memory that lasts.</h1>
  <p class="hero__lede"><strong>Context that persists. Built for production.</strong> Lint-AI helps an assistant remember useful preferences, decisions, and history across interactions, then brings the right context back when it matters.</p>
  <p><a class="md-button md-button--primary" href="quickstart/">Get started</a> <a class="md-button" href="agent-memory/">Explore agent memory</a></p>
</section>
</div>

## From one conversation to a lasting relationship {.landing-heading}

On its own, an AI model only has the context it can see right now. Lint-AI adds
a memory that stays between conversations. As an agent saves new information or
feedback, that memory can be updated and used in future interactions. The model
does not need to be retrained; it gets better context to work with.

<div class="feature-grid">
  <article>
    <span class="feature-number">01</span>
    <h3>Remember the person and the work</h3>
    <p>Carry useful preferences, ongoing tasks, and past decisions from one
    interaction to the next. Your assistant can pick up where you left off
    instead of asking you to start over.</p>
  </article>
  <article>
    <span class="feature-number">02</span>
    <h3>Learn from what changes</h3>
    <p>When new information or feedback updates a memory, future conversations
    can use it. Lint-AI keeps the earlier history too, so an old answer does not
    quietly become the current one.</p>
  </article>
  <article>
    <span class="feature-number">03</span>
    <h3>Add memory without building it all yourself</h3>
    <p>Lint-AI handles saving, updating, and finding memories through one
    persistent service. Use the integrations and APIs to add memory to an agent
    without building a separate memory system from scratch.</p>
  </article>
</div>

## What lasting memory makes possible {.landing-heading}

Connect Lint-AI to the experience you are building:

<div class="feature-grid">
  <article>
    <h3>AI assistants</h3>
    <p>Remember preferences and past conversations to make each new interaction
    feel more personal and consistent.</p>
  </article>
  <article>
    <h3>Customer support</h3>
    <p>Bring relevant case history and past resolutions into the next support
    conversation, so people do not have to repeat themselves.</p>
  </article>
  <article>
    <h3>Autonomous systems</h3>
    <p>Carry goals, task progress, and previous outcomes across runs, so an agent
    can continue work with the state it has already built.</p>
  </article>
</div>

The same memory layer can support productivity tools and other applications
that benefit from remembering user preferences and past activity. Connect these
applications through the [HTTP API](server.md), [MCP](mcp.md),
[Python](python-migration-0.3.0.md), or [Rust](memory-service-api.md) interface.

## Memory where your agents already work {.landing-heading}

Use the same project memory inside supported agent tools. Information captured
in one can be available to the others, so people can keep using the tools they
already know.

<div class="integration-grid">
  <a href="claude-code/"><strong>Claude Code</strong><span>Memory and lifecycle hooks →</span></a>
  <a href="codex/"><strong>Codex</strong><span>Memory, hooks, and MCP →</span></a>
  <a href="gemini-cli/"><strong>Gemini CLI</strong><span>Memory and lifecycle integration →</span></a>
  <a href="agents/"><strong>More integrations</strong><span>AGY, Muse Code, OpenClaw, Hermes, and RooAGI AgentFlow →</span></a>
</div>

## Built for dependable use {.landing-heading}

Lint-AI keeps memory in a persistent store and provides a shared service for
agents and applications. It supports user and session scoping, durable writes,
and retrieval that can account for newer corrections while preserving history.
Run it locally or connect through its HTTP and MCP servers. A dashboard helps
you inspect memory and agent activity.

<h3>Benchmark highlights</h3>

<section class="proof-grid proof-grid--six" aria-label="Lint-AI benchmark highlights">
  <div><strong>96.0%</strong><span>fused temporal any-hit Recall@5 · 500 questions</span></div>
  <div><strong>94.2%</strong><span>routed top-5 any-hit Recall@5</span></div>
  <div><strong>2.63 ms</strong><span>single-index mean search latency</span></div>
  <div><strong>2,246.69/s</strong><span>read-only searches · C=10 · Oct 4</span></div>
  <div><strong>212.36/s</strong><span>single records published · <code>/add</code></span></div>
  <div><strong>1,977.12/s</strong><span>batch records published · 128 per call</span></div>
</section>

<p class="benchmark-note">Retrieval scores use 500 LongMemEval-S questions with Tantivy 0.25.0. In an experimental route comparison, fused temporal + global reached 96.0% any-hit Recall@5; routed top-5 reached 94.2%. Mean single-index search time was 2.58 ms. Read throughput is a separate C=10 read-only run on 23,366 records from October 4. Published write rates include final flush: <code>/add</code> completed 212.36 records/s; batches of 128 completed 1,977.12 records/s. The batch run was saturated and rejected about 25,500–26,700 requests per repetition, so its rate is a burst result. In the October 5 sparse mixed-load check, ten readers sustained 1,248.69 searches/s with one single-record write per second, versus 1,209.01 searches/s without writes. See the <a href="benchmark-results/">retrieval results</a> and <a href="http-server-benchmarks/">HTTP server benchmarks</a> for methods and reports.</p>

<section class="final-cta">
  <p>Context that persists.</p>
  <h2>Build agents that remember and improve over time.</h2>
  <a class="md-button md-button--primary" href="quickstart/">Read the quickstart</a>
</section>
