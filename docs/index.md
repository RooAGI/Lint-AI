---
hide:
  - navigation
  - toc
---

<div class="hero-stack">
<section class="hero">
  <div class="hero__eyebrow">CURRENT CONTEXT FOR AI AGENTS</div>
  <h1>Relevant context<br><span>is not always true.</span></h1>
  <p class="hero__lede">Lint-AI turns scattered project history, including sessions, documents, decisions, and traces, into current, evidence-backed context at the moment an agent needs it.</p>
</section>

<section class="scenario" aria-label="Example of an agent retrieving a superseded decision">
  <div class="scenario__timeline">
    <article class="scenario-card scenario-card--complete">
      <div><span>OLDER EVIDENCE</span><time>ONCE RELEVANT</time></div>
      <p>“Increase retries to recover from the intermittent timeout.”</p>
      <small>Topically relevant · no longer current</small>
    </article>
    <div class="scenario__connector"><span>SUPERSEDED</span></div>
    <article class="scenario-card scenario-card--empty">
      <div><span>CURRENT STATE</span><time>SUPPORTED BY NEWER EVIDENCE</time></div>
      <p>“Retries amplify load. Cap attempts and fix the token clock skew.”</p>
      <small>Source-linked · time-aware · current</small>
    </article>
  </div>
  <div class="failure-points" aria-label="Signals needed to retrieve the right memory">
    <span>Finding a relevant passage is not enough.</span>
    <ol>
      <li><b>01</b> Relevance</li>
      <li><b>02</b> Recency</li>
      <li><b>03</b> Supersession</li>
      <li><b>04</b> Evidence</li>
    </ol>
  </div>
  <div class="scenario__outcome">
    <span>WITHOUT LINT-AI</span>
    <p>Ordinary retrieval can surface an old recommendation as current because it is still topically relevant.</p>
  </div>
  <div class="scenario__outcome scenario__outcome--good">
    <span>WITH LINT-AI</span>
    <p>The current state ranks first. Older guidance remains useful as history without quietly becoming today’s answer.</p>
  </div>
</section>
</div>

<section class="solution-intro">
  <div class="hero__eyebrow">THE MISSING LAYER</div>
  <h2>Move from relevant retrieval to current understanding.</h2>
  <p>Search can find the right topic. Lint-AI helps agents distinguish what is relevant from what is still true, while preserving the sources, timestamps, and history behind each answer.</p>
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

## Prevent confident staleness {.landing-heading}

Agent context is not a pile of text. Decisions supersede older decisions. Terms drift. Ownership changes. The right answer often depends on *when* something was true and *where* the evidence came from.

<div class="feature-grid">
  <article>
    <span class="feature-number">01</span>
    <h3>Know what is current</h3>
    <p>Rank current evidence ahead of older guidance and preserve historical answers when a question depends on the past.</p>
  </article>
  <article>
    <span class="feature-number">02</span>
    <h3>Show why it is current</h3>
    <p>Return source, time, and relationship signals with the context so an agent or reviewer can inspect the basis for an answer.</p>
  </article>
  <article>
    <span class="feature-number">03</span>
    <h3>Make drift visible</h3>
    <p>Surface contradictions, stale claims, terminology drift, orphan pages, and missing links before they become confident answers.</p>
  </article>
</div>

## Keep your sources. Add a current-state layer. {.landing-heading}

Lint-AI does not ask you to discard your existing project knowledge. It indexes the sessions, notes, documents, and decisions you already have, then makes their relationships and history usable at retrieval time. Each provider gets isolated, project-scoped memory, lifecycle capture, and shared MCP controls.

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
