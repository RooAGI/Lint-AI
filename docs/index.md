---
hide:
  - navigation
  - toc
---

<div class="hero-stack">
<section class="hero">
  <div class="hero__eyebrow">LINT-AI · RELIABLE MEMORY &amp; INSIGHTS</div>
  <h1>Agent memory that knows what is still true.</h1>
  <p class="hero__lede"><strong>Memory and insights for AI agents.</strong> Lint-AI turns project history — sessions, documents, decisions, traces, and code — into current, evidence-backed context, while showing what changed, what was superseded, and where the answer came from.</p>
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

## What’s new in v0.3.0 {.landing-heading}

Lint-AI v0.3.0 unifies memory access across Rust, Python, HTTP, and agent
integrations. It adds conversation-aware retrieval, stronger correction and
supersession handling, broader language support, and more agent integrations.

<div class="feature-grid">
  <article>
    <span class="feature-number">01</span>
    <h3>One memory API</h3>
    <p>Rust integrations use <code>MemoryService</code>, while Python clients use
    <code>lint_ai.Memory</code> for both local and HTTP-backed memory.</p>
  </article>
  <article>
    <span class="feature-number">02</span>
    <h3>Conversation-aware search</h3>
    <p>Session context carries across short-lived tools and helps resolve
    follow-up questions with current project evidence.</p>
  </article>
  <article>
    <span class="feature-number">03</span>
    <h3>Current facts, more agents</h3>
    <p>Correction and supersession handling keeps stale claims from resurfacing
    as current. Integrations include Claude Code, Codex, Gemini CLI, AGY, Muse,
    Hermes, OpenClaw, and <code>rooagi_runtime</code>.</p>
  </article>
</div>

<p class="benchmark-note">Integrations are feature-gated. The default build enables the supported <code>agent-integrations</code> feature set; use individual feature flags for a smaller build.</p>

[Read the full v0.3.0 release notes](releases/0.3.0.md)

<section class="proof-grid" aria-label="Lint-AI benchmark highlights">
  <div><strong>96.24%</strong><span>adaptive any-hit Recall@5</span></div>
  <div><strong>95.49%</strong><span>fixed any-hit Recall@5</span></div>
  <div><strong>1.25 ms</strong><span>fixed query latency</span></div>
  <div><strong>2,246.69/s</strong><span>Tantivy 0.25.0 single-index HTTP at C=10</span></div>
</section>

<p class="benchmark-note">Retrieval quality and HTTP throughput are separate measurements. The 2,246.69 req/s figure is the Tantivy 0.25.0 median at C=10 from a five-run version comparison using 23,366 records, 1,000 measured requests and 100 warm-ups per run. The matching five-segment routed layout measured 888.86 req/s. The older 1,512.31 req/s figure is retained as a historical v0.2.0 result. See the <a href="comparison/">throughput notes and benchmark artifacts</a> for workload details and caveats.</p>

## See the dashboard {.landing-heading}

The local dashboard gives operators a live view of project indexes, segment layout, query activity, latency, errors, and provider telemetry across the agent integrations.

<section class="dashboard-slideshow" data-dashboard-slideshow aria-label="Lint-AI dashboard screenshots">
  <div class="dashboard-slideshow__stage">
    <div class="dashboard-slideshow__track">
      <figure class="dashboard-slide is-active" data-dashboard-slide>
        <img src="assets/lint-ai-dashboard-overview.png" alt="Lint-AI dashboard overview showing project health and query activity" loading="eager">
        <figcaption><strong>Overall</strong><span>Project health, query activity, and recent events</span></figcaption>
      </figure>
      <figure class="dashboard-slide" data-dashboard-slide>
        <img src="assets/lint-ai-dashboard-index.png" alt="Lint-AI dashboard topology showing the routed project segments" loading="lazy">
        <figcaption><strong>Topology</strong><span>Segment routing and document distribution</span></figcaption>
      </figure>
      <figure class="dashboard-slide" data-dashboard-slide>
        <img src="assets/lint-ai-dashboard-providers.png" alt="Lint-AI dashboard provider telemetry showing Codex activity and token usage" loading="lazy">
        <figcaption><strong>Provider</strong><span>Provider activity, sessions, and token usage</span></figcaption>
      </figure>
    </div>
  </div>
  <div class="dashboard-slideshow__controls">
    <button type="button" data-dashboard-prev aria-label="Previous dashboard screenshot">←</button>
    <div class="dashboard-slideshow__dots" role="tablist" aria-label="Dashboard screenshot views">
      <button type="button" class="is-active" data-dashboard-dot="0" role="tab" aria-selected="true">Overall</button>
      <button type="button" data-dashboard-dot="1" role="tab" aria-selected="false">Topology</button>
      <button type="button" data-dashboard-dot="2" role="tab" aria-selected="false">Provider</button>
    </div>
    <button type="button" data-dashboard-next aria-label="Next dashboard screenshot">→</button>
  </div>
</section>

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

## Keep your sources. Add a current-state memory layer. {.landing-heading}

Lint-AI does not ask you to discard your existing project knowledge. It indexes the sessions, notes, documents, and decisions you already have, then makes their relationships and history usable at retrieval time. Each provider gets isolated, project-scoped agent memory, lifecycle capture, and shared MCP controls.

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
  <p>Current-state agent memory for AI coding agents.</p>
  <h2>AI memory that knows what is still true.</h2>
  <a class="md-button md-button--primary" href="quickstart/">Read the quickstart</a>
</section>
