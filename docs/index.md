---
hide:
  - navigation
  - toc
---

<div class="hero-stack">
<section class="hero">
  <div class="hero__eyebrow">LINT-AI · RELIABLE MEMORY &amp; INSIGHTS</div>
  <h1>Reliable AI memory<br><span>with insights you can trust.</span></h1>
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

## What’s new in v0.2.0 {.landing-heading}

Lint-AI v0.2.0 helps AI agents find useful project context faster, keep working
while memory is updated, and use that memory more safely across different tools.

<div class="feature-grid">
  <article>
    <span class="feature-number">01</span>
    <h3>Find the right memory faster</h3>
    <p>Projects are organized into searchable sections, so a query can start with
    the most relevant context and expand its search when needed.</p>
  </article>
  <article>
    <span class="feature-number">02</span>
    <h3>Keep working during updates</h3>
    <p>Agents can continue searching a trusted snapshot while new memories are
    written and checked in the background.</p>
  </article>
  <article>
    <span class="feature-number">03</span>
    <h3>Reliable memory, safer integrations</h3>
    <p>Validated updates, duplicate protection, and project boundaries keep memory
    dependable across Claude Code, Codex, Gemini CLI, and Antigravity CLI.</p>
  </article>
</div>

<p class="benchmark-note">Provider integrations remain opt-in. The default build stays lightweight, while <code>agent-integrations</code> enables all supported providers.</p>

[Read the full v0.2.0 release notes](releases/0.2.0.md)

<section class="proof-grid" aria-label="Lint-AI benchmark highlights">
  <div><strong>96.24%</strong><span>adaptive any-hit Recall@5</span></div>
  <div><strong>95.49%</strong><span>fixed any-hit Recall@5</span></div>
  <div><strong>1.25 ms</strong><span>fixed query latency</span></div>
  <div><strong>1,512.31/s</strong><span>v0.2.0 routed HTTP at C=10</span></div>
</section>

<p class="benchmark-note">v0.2.0 segmented benchmark · 133 multi-session questions · adaptive routing reached 96.24% any-hit Recall@5; fixed routing reached 95.49% at 1.25 ms average latency · Latest throughput is the cold-start median of five runs over 23,366 records on macOS M5 Pro; results vary by hardware and index layout</p>

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
