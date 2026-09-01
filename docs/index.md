---
hide:
  - navigation
  - toc
---

<div class="hero-stack">
<section class="hero">
  <div class="hero__eyebrow">RELIABLE PROJECT MEMORY FOR AI AGENTS</div>
  <h1>Project memory<br><span>that knows what changed.</span></h1>
  <p class="hero__lede">Lint-AI captures decisions across agent sessions, recognizes when newer evidence supersedes older guidance, and retrieves current, source-backed context for Codex, Claude Code, Gemini CLI, and AGY.</p>
</section>

<section class="scenario" aria-label="Example of an agent retrieving a superseded decision">
  <div class="scenario__timeline">
    <article class="scenario-card scenario-card--complete">
      <div><span>OLDER MEMORY</span><time>INITIAL RECOMMENDATION</time></div>
      <p>“Increase retries to recover from the intermittent timeout.”</p>
      <small>Still searchable · no longer current</small>
    </article>
    <div class="scenario__connector"><span>SUPERSEDED</span></div>
    <article class="scenario-card scenario-card--empty">
      <div><span>LATEST DECISION</span><time>SUPPORTED BY EVIDENCE</time></div>
      <p>“Retries amplify load. Cap attempts and fix the token clock skew.”</p>
      <small>Newer evidence · source-linked · current</small>
    </article>
  </div>
  <div class="failure-points" aria-label="Signals needed to retrieve the right memory">
    <span>Finding memory is not enough.</span>
    <ol>
      <li><b>01</b> Relevance</li>
      <li><b>02</b> Recency</li>
      <li><b>03</b> Supersession</li>
      <li><b>04</b> Evidence</li>
    </ol>
  </div>
  <div class="scenario__outcome">
    <span>WITHOUT LINT-AI</span>
    <p>The stale recommendation can surface as truth—sending the agent toward a fix already proven harmful.</p>
  </div>
  <div class="scenario__outcome scenario__outcome--good">
    <span>WITH LINT-AI</span>
    <p>The current decision ranks first. Superseded guidance stays out of retrieval, while timestamps and source evidence keep the answer inspectable.</p>
  </div>
</section>
</div>

<section class="solution-intro">
  <div class="hero__eyebrow">HOW LINT-AI HELPS</div>
  <h2>Give agents the right past—not all of it.</h2>
  <p>Lint-AI turns project history into ranked, current, evidence-backed context. It combines lifecycle capture, temporal reasoning, explicit supersession, and source-aware retrieval across Codex, Claude Code, Gemini CLI, and AGY—so agents can act on what is true now and teams can inspect why.</p>
  <div class="hero__actions">
    <a class="md-button md-button--primary" href="quickstart/">Start building</a>
    <a class="md-button" href="https://github.com/RooAGI/Lint-AI">View on GitHub</a>
  </div>
  <div class="install-line">
    <code>cargo install --git https://github.com/RooAGI/Lint-AI</code>
  </div>
</section>

<section class="proof-grid" aria-label="Lint-AI benchmark highlights">
  <div><strong>95.6%</strong><span>recall@10</span></div>
  <div><strong>84.0%</strong><span>MRR</span></div>
  <div><strong>1.9 ms</strong><span>average query latency</span></div>
  <div><strong>952/s</strong><span>HTTP searches at C=10</span></div>
</section>

<p class="benchmark-note">Heuristic release backend · LongMemEval-S, 500 scoped questions, fair comparison track using any-hit recall · HTTP figure uses 23,366 records</p>

## Reproducible performance comparison {.landing-heading}

Lint-AI's retrieval and server-load measurements are published with the exact
scripts, payloads, corpus sizes, and caveats needed to reproduce them. In the
normalized 23,366-record HTTP run, Lint-AI sustained **952 req/s at
concurrency 10**, compared with **171 req/s** for AgentMemory in keyless BM25
mode. This is a service-load comparison, not a claim that the two systems have
identical retrieval semantics.

[View the comparison methodology and results](comparison.md)

## Memory that understands change {.landing-heading}

Agent context is not a pile of text. Decisions supersede older decisions. Terms drift. Ownership changes. The right answer often depends on *when* something was true and *where* the evidence came from.

<div class="feature-grid">
  <article>
    <span class="feature-number">01</span>
    <h3>Retrieve the right past</h3>
    <p>Search sessions, notes, docs, and code with lexical, entity, temporal, and graph signals working together.</p>
  </article>
  <article>
    <span class="feature-number">02</span>
    <h3>Keep answers grounded</h3>
    <p>Return LLM-ready context with source evidence instead of opaque similarity matches or raw text blobs.</p>
  </article>
  <article>
    <span class="feature-number">03</span>
    <h3>Catch corpus drift</h3>
    <p>Surface contradictions, stale claims, terminology drift, orphan pages, and missing cross-references.</p>
  </article>
</div>

## One memory layer. Your agent of choice. {.landing-heading}

Lint-AI integrates with the tools your team already uses. Each provider gets isolated, project-scoped memory, lifecycle capture, and shared MCP controls.

<div class="integration-grid">
  <a href="codex/"><strong>Codex</strong><span>Hooks, MCP, replay, project memory →</span></a>
  <a href="claude-code/"><strong>Claude Code</strong><span>Hooks, MCP, status line, replay →</span></a>
  <a href="gemini-cli/"><strong>Gemini CLI</strong><span>JSON hooks and shared MCP tools →</span></a>
  <a href="agy/"><strong>Antigravity CLI</strong><span>Gemini-compatible lifecycle protocol →</span></a>
</div>

## From corpus to grounded context {.landing-heading}

<div class="pipeline" role="list" aria-label="Lint-AI processing pipeline">
  <div role="listitem"><span>01</span><strong>Ingest</strong><small>Sessions · docs · code · traces</small></div>
  <div role="listitem"><span>02</span><strong>Understand</strong><small>Facts · entities · symbols · time</small></div>
  <div role="listitem"><span>03</span><strong>Connect</strong><small>Links · ownership · co-occurrence</small></div>
  <div role="listitem"><span>04</span><strong>Retrieve</strong><small>Ranked, sourced, current context</small></div>
</div>

<section class="final-cta">
  <p>Start with a local corpus. Keep the evidence.</p>
  <h2>Make memory inspectable.</h2>
  <a class="md-button md-button--primary" href="quickstart/">Read the quickstart</a>
</section>
