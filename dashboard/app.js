import { renderEventDetail, renderToolCall, toolDetail, toolGroup } from './event-detail.js';

const VIEWS = {
  overview: { title: 'Overview', sub: 'How the memory index is serving agents right now.' },
  index: { title: 'Index', sub: 'Segment topology and document distribution of the primary query index.' },
  providers: { title: 'Providers', sub: 'Per-integration sessions, tool use, and private memory stores.' }
};

const state = { view: 'overview', status: null, series: [], provider: null, tool: null, hover: null };

const tokenInput = document.querySelector('#token');
tokenInput.value = localStorage.getItem('lint-ai-dashboard-token') || '';
tokenInput.addEventListener('change', () => localStorage.setItem('lint-ai-dashboard-token', tokenInput.value));
document.querySelector('#refresh').addEventListener('click', refresh);

async function api(path) {
  const token = tokenInput.value.trim().replace(/^Bearer\s+/i, '');
  const response = await fetch(path, { cache: 'no-store', headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

const esc = value => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;'}[c]));
const fmt = value => Number(value || 0).toLocaleString();
const clock = value => value ? new Date(value).toLocaleTimeString() : '—';
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const seconds = ms => ms ? `${(ms / 1000).toFixed(1)}s` : '—';
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const stillness = window.matchMedia('(prefers-reduced-motion: reduce)');

// Animate a figure between readings. `render` receives the interpolated numbers,
// so composite readouts (p50 / p95) move as one.
function tween(element, next, render) {
  if (!element) return;
  const previous = JSON.parse(element.dataset.figures || 'null');
  element.dataset.figures = JSON.stringify(next);
  if (stillness.matches || !previous || previous.length !== next.length) {
    element.textContent = render(next);
    return;
  }
  cancelAnimationFrame(Number(element.dataset.frame || 0));
  const started = performance.now();
  const step = now => {
    const progress = Math.min(1, (now - started) / 520);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = render(next.map((value, index) => previous[index] + (value - previous[index]) * eased));
    if (progress < 1) element.dataset.frame = requestAnimationFrame(step);
  };
  element.dataset.frame = requestAnimationFrame(step);
}

// Second half of the window against the first: the direction the measure is moving.
function trend(values) {
  if (values.length < 6) return null;
  const mean = list => list.reduce((sum, value) => sum + value, 0) / (list.length || 1);
  const half = Math.floor(values.length / 2);
  const before = mean(values.slice(0, half));
  const after = mean(values.slice(half));
  if (!before && !after) return null;
  const change = before ? (after - before) / before : 1;
  return { change, direction: Math.abs(change) < 0.02 ? 'flat' : change > 0 ? 'up' : 'down' };
}

// `polarity` says what a rise means: 'lower-better' for latency and errors,
// 'neutral' for throughput, where neither direction is good news by itself.
function renderDelta(element, values, polarity) {
  if (!element) return;
  const moved = trend(values);
  if (!moved) { element.textContent = ''; element.className = 'delta'; return; }
  const glyph = moved.direction === 'flat' ? '→' : moved.direction === 'up' ? '↑' : '↓';
  const tone = polarity === 'neutral' || moved.direction === 'flat'
    ? ''
    : (moved.direction === 'up') === (polarity === 'lower-better') ? ' bad' : ' good';
  element.className = `delta${tone}`;
  element.textContent = `${glyph} ${Math.abs(moved.change * 100).toFixed(0)}%`;
  element.title = `${moved.direction === 'flat' ? 'Steady' : moved.direction === 'up' ? 'Up' : 'Down'} versus the first half of the window`;
}

/* ── view routing ──────────────────────────────────────────────── */

function showView(view) {
  if (!VIEWS[view]) view = 'overview';
  state.view = view;
  document.querySelectorAll('.view').forEach(section => { section.hidden = section.id !== `view-${view}`; });
  document.querySelectorAll('.navitem').forEach(item => {
    if (item.dataset.view === view) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  });
  document.querySelector('#view-title').textContent = VIEWS[view].title;
  document.querySelector('#view-sub').textContent = VIEWS[view].sub;
  if (location.hash.slice(1) !== view) history.replaceState(null, '', `#${view}`);
  document.querySelector('.viewport').scrollTop = 0;
  if (view === 'overview') drawSparks();
}

document.querySelectorAll('.navitem').forEach(item => item.addEventListener('click', () => showView(item.dataset.view)));
document.querySelectorAll('[data-goto]').forEach(item => item.addEventListener('click', () => showView(item.dataset.goto)));
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

/* ── polling ───────────────────────────────────────────────────── */

async function refresh() {
  const pill = document.querySelector('#status-pill');
  try {
    const [status, series] = await Promise.all([api('/api/status'), api('/api/timeseries')]);
    state.status = status;
    state.series = series.query_series || [];

    const healthy = status.status === 'healthy';
    pill.className = `pill ${healthy ? 'ok' : 'warn'}`;
    pill.querySelector('span').textContent = healthy ? 'Healthy' : 'Index needs attention';

    const query = status.telemetry.query_summary;
    tween(document.querySelector('#rps'), [Number(query.requests_per_second || 0)], ([rate]) => rate.toFixed(2));
    tween(document.querySelector('#latency'), [query.p50_ms, query.p95_ms], ([p50, p95]) => `${p50.toFixed(0)} / ${p95.toFixed(0)}`);
    tween(document.querySelector('#errors'), [Number(query.errors || 0)], ([errors]) => fmt(Math.round(errors)));
    tween(document.querySelector('#empty'), [Number(query.empty_results || 0)], ([empty]) => fmt(Math.round(empty)));
    document.querySelector('#errors-scope').firstChild.textContent = `${pct(query.error_rate)} of queries `;
    document.querySelector('#empty-scope').firstChild.textContent = `${pct(query.empty_result_rate)} of queries `;
    renderDelta(document.querySelector('#rps-delta'), state.series.map(point => Number(point.requests_per_second || 0)), 'neutral');
    renderDelta(document.querySelector('#latency-delta'), state.series.map(point => Number(point.p95_ms || 0)), 'lower-better');
    renderDelta(document.querySelector('#errors-delta'), state.series.map(point => Number(point.errors || 0)), 'lower-better');
    renderDelta(document.querySelector('#empty-delta'), state.series.map(point => Number(point.empty_results || 0)), 'lower-better');
    document.querySelector('#window').textContent = `${status.telemetry.window_seconds}s rolling window`;
    document.querySelector('#last-updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
    document.body.classList.remove('booting');

    const integrations = status.integrations || [];
    const projectIndex = projectIndexView(status.index, integrations);
    renderIndexBrief(projectIndex);
    renderIndexTopology(projectIndex);
    renderProviders(integrations, projectIndex);
    renderFeed(integrations);
    drawSparks();
  } catch (error) {
    pill.className = 'pill bad';
    pill.querySelector('span').textContent = 'Unavailable';
    document.querySelector('#last-updated').textContent = error.message;
  }
}

/* ── query activity: small multiples, one scale per measure ────── */

const SPARKS = {
  requests_per_second: { color: '--series-requests', label: 'Requests / sec', format: value => value.toFixed(2) },
  p95_ms: { color: '--series-latency', label: 'p95 latency', format: value => `${value.toFixed(1)} ms` },
  errors: { color: '--series-errors', label: 'Errors', format: value => fmt(Math.round(value)) }
};

function drawSparks() {
  const surface = cssVar('--surface-2');
  const grid = cssVar('--line');
  document.querySelectorAll('#sparks .spark').forEach(figure => {
    const spec = SPARKS[figure.dataset.metric];
    const values = state.series.map(point => Number(point[figure.dataset.metric] || 0));
    const index = state.hover != null && state.hover < values.length ? state.hover : values.length - 1;
    figure.querySelector('figcaption b').textContent = values.length ? spec.format(values[index]) : '—';
    const canvas = figure.querySelector('canvas');
    const end = drawSpark(canvas, values, cssVar(spec.color), surface, grid, state.hover);
    // The pulse marker is a DOM element so it can breathe in CSS rather than
    // re-rasterising three canvases every frame. Offset by the canvas's own
    // position in the figure, since the caption sits above it.
    figure.classList.toggle('has-pulse', Boolean(end) && state.hover == null);
    if (end) {
      figure.style.setProperty('--px', `${end.x + canvas.offsetLeft}px`);
      figure.style.setProperty('--py', `${end.y + canvas.offsetTop}px`);
    }
  });
  renderSparkTip();
}

// Quadratic smoothing through segment midpoints: rounder than a polyline and,
// unlike a cubic spline, it never overshoots outside the data's own range.
function tracePath(ctx, points) {
  ctx.moveTo(points[0].x, points[0].y);
  if (points.length === 1) return;
  for (let index = 0; index < points.length - 1; index++) {
    const current = points[index];
    const next = points[index + 1];
    ctx.quadraticCurveTo(current.x, current.y, (current.x + next.x) / 2, (current.y + next.y) / 2);
  }
  ctx.lineTo(points.at(-1).x, points.at(-1).y);
}

function drawSpark(canvas, values, color, surface, grid, hover) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return null;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  // Inset so the 8px end marker is never clipped by the canvas edge.
  const pad = 5;
  const top = 9;
  const bottom = height - 5;
  const plot = width - pad * 2;
  // Zero baseline: every measure here is a non-negative rate or count.
  const max = Math.max(1e-9, ...values);
  const x = index => values.length === 1 ? width / 2 : pad + index / (values.length - 1) * plot;
  const y = value => bottom - (value / max) * (bottom - top);

  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, bottom + 0.5);
  ctx.lineTo(width, bottom + 0.5);
  ctx.stroke();

  if (!values.length) return null;
  const points = values.map((value, index) => ({ x: x(index), y: y(value) }));

  const gradient = ctx.createLinearGradient(0, top, 0, bottom);
  gradient.addColorStop(0, `${color}59`);
  gradient.addColorStop(1, `${color}00`);
  ctx.beginPath();
  tracePath(ctx, points);
  ctx.lineTo(points.at(-1).x, bottom);
  ctx.lineTo(points[0].x, bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  tracePath(ctx, points);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();

  const marker = hover != null && hover < values.length ? hover : values.length - 1;
  if (hover != null && hover < values.length) {
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(Math.round(x(hover)) + 0.5, top - 6);
    ctx.lineTo(Math.round(x(hover)) + 0.5, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.beginPath();
  ctx.arc(points[marker].x, points[marker].y, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  // A 2px surface ring keeps the marker legible where it sits on the line.
  ctx.strokeStyle = surface;
  ctx.lineWidth = 2;
  ctx.stroke();
  return points[marker];
}

function renderSparkTip() {
  const tip = document.querySelector('#spark-tip');
  const point = state.hover != null ? state.series[state.hover] : null;
  if (!point) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.innerHTML = `<strong>${clock(point.start_ms)}</strong>` + Object.entries(SPARKS)
    .map(([metric, spec]) => `<span><i style="background:${cssVar(spec.color)}"></i>${spec.label}<b>${spec.format(Number(point[metric] || 0))}</b></span>`)
    .join('');
  const canvas = sparks.querySelector('canvas');
  const ratio = state.series.length > 1 ? state.hover / (state.series.length - 1) : 0.5;
  tip.style.left = `${Math.round(ratio * canvas.clientWidth)}px`;
}

const sparks = document.querySelector('#sparks');
sparks.addEventListener('mousemove', event => {
  const rect = sparks.querySelector('canvas').getBoundingClientRect();
  const count = state.series.length;
  if (!count || !rect.width) return;
  const position = Math.round((event.clientX - rect.left) / rect.width * (count - 1));
  const next = Math.min(count - 1, Math.max(0, position));
  if (next !== state.hover) { state.hover = next; drawSparks(); }
});
sparks.addEventListener('mouseleave', () => { state.hover = null; drawSparks(); });

/* ── index ─────────────────────────────────────────────────────── */

// The project view spans the shared workspace corpus and every discovered
// provider store. Keep segments qualified by their store name so identical
// local segment ids from separate indexes never look like one segment.
function projectIndexView(primary, integrations) {
  const stores = [];
  if (primary?.snapshot) stores.push({
    name: 'workspace-memory',
    snapshot: primary.snapshot,
    snapshot_revision: primary.snapshot_revision,
  });
  integrations.flatMap(provider => provider.indexes || []).forEach(index => {
    if (index.snapshot) stores.push({
      name: index.name,
      snapshot: index.snapshot,
      snapshot_revision: index.snapshot_revision,
    });
  });
  if (!stores.length) return primary;
  const segments = stores.flatMap(store => (store.snapshot.segments || []).map(segment => ({
    ...segment,
    index_name: store.name,
    segment_id: `${store.name}/${segment.segment_id}`
  })));
  return {
    ...primary,
    snapshot_revision: Math.max(...stores.map(store => Number(store.snapshot_revision || 0))),
    snapshot: {
      layout: 'project stores',
      store_count: stores.length,
      segment_count: segments.length,
      global_document_count: segments.reduce((sum, segment) => sum + Number(segment.document_count || 0), 0),
      segments
    }
  };
}

// Sequential = one hue, monotonic lightness. On a dark ground "more" reads as
// brighter, so the ramp runs dim -> bright as document load rises.
const LOAD_RAMP = ['#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef'];
const loadColor = fraction => LOAD_RAMP[Math.min(LOAD_RAMP.length - 1, Math.max(0, Math.round(fraction * (LOAD_RAMP.length - 1))))];

function segmentStats(segments) {
  const counts = segments.map(segment => Number(segment.document_count || 0));
  const total = counts.reduce((sum, count) => sum + count, 0);
  const average = counts.length ? total / counts.length : 0;
  const smallest = counts.length ? Math.min(...counts) : 0;
  const largest = counts.length ? Math.max(...counts) : 0;
  return { total, average, smallest, largest, spread: average ? (largest - smallest) / average : 0 };
}

function renderIndexBrief(index) {
  const target = document.querySelector('#index-brief');
  const snapshot = index?.snapshot;
  if (!snapshot) {
    target.innerHTML = '<div class="empty-state">No published segment snapshot yet.</div>';
    return;
  }
  const segments = snapshot.segments || [];
  const { total, largest, spread } = segmentStats(segments);
  const bars = segments.slice(0, 6).map((segment, position) => {
    const count = Number(segment.document_count || 0);
    const fraction = largest ? count / largest : 0;
    return `<div class="brief-bar"><span>${String(position + 1).padStart(2, '0')}</span><i style="width:${Math.max(3, fraction * 100)}%;background:${loadColor(fraction)}"></i><b>${fmt(count)}</b></div>`;
  }).join('');
  target.innerHTML = `
    <div class="brief-figures">
      <div><span>Segments</span><strong>${fmt(snapshot.segment_count)}</strong></div>
      <div><span>Documents</span><strong>${fmt(total)}</strong></div>
      <div><span>Revision</span><strong>${fmt(index.snapshot_revision)}</strong></div>
      <div><span>Balance</span><strong>${spread < 0.2 ? 'Even' : 'Uneven'}</strong></div>
    </div>
    <div>
      <p class="eyebrow">Document load · ${esc(snapshot.layout)}</p>
      <div class="brief-bars">${bars || '<div class="empty-state">No segments published.</div>'}</div>
      ${segments.length > 6 ? `<p class="panel__note" style="text-align:left;margin-top:6px">+${fmt(segments.length - 6)} more segments</p>` : ''}
    </div>`;
}

// Hub-and-spoke network diagram. Segments fan out on one or two arcs to the
// right of the router, so the trunk from the gateway stays clear on the left.
// Node radius and spoke weight both encode document load.
function topologyDiagram(snapshot, segments, largest) {
  const width = 360;
  const height = 340;
  const gateway = { x: 46, y: height / 2 };
  const hub = { x: 168, y: height / 2 };
  // A 118° fan reads as "routes out to the right" without wrapping back around
  // the hub and colliding with its caption.
  const arc = (118 * Math.PI) / 180;
  const twoRings = segments.length > 12;
  const count = segments.length;

  const nodes = segments.map((segment, index) => {
    const documents = Number(segment.document_count || 0);
    const fraction = largest ? documents / largest : 0;
    const position = count === 1 ? 0.5 : index / (count - 1);
    const angle = -arc / 2 + position * arc;
    const radius = twoRings ? (index % 2 ? 156 : 104) : 132;
    return {
      index, segment, documents, fraction, angle,
      x: hub.x + Math.cos(angle) * radius,
      y: hub.y + Math.sin(angle) * radius,
      r: 7.5 + fraction * 7,
      color: loadColor(fraction),
      // Brighter ramp steps carry dark type; dim steps carry light type.
      ink: fraction > 0.45 ? '#08111f' : '#dbe6f7'
    };
  });

  const spokes = nodes.map(node => {
    // Bow each spoke slightly so overlapping runs stay tellable apart.
    const midX = (hub.x + node.x) / 2 + Math.cos(node.angle) * 5;
    const midY = (hub.y + node.y) / 2 + Math.sin(node.angle) * 5 - 8;
    return `<path class="tspoke" id="tspoke-${node.index}" d="M${hub.x} ${hub.y} Q${midX.toFixed(1)} ${midY.toFixed(1)} ${node.x.toFixed(1)} ${node.y.toFixed(1)}" stroke-width="${(0.8 + node.fraction * 2).toFixed(2)}" style="--spoke:${node.color}"/>`;
  }).join('');

  const marks = nodes.map(node => `<g class="tnode" data-index="${node.index}" transform="translate(${node.x.toFixed(1)} ${node.y.toFixed(1)})"><title>Segment ${String(node.index + 1).padStart(2, '0')} · ${esc(node.segment.segment_id)} · ${fmt(node.documents)} documents</title><circle class="tnode__halo" r="${(node.r + 5).toFixed(1)}" style="--node:${node.color}"/><circle class="tnode__dot" r="${node.r.toFixed(1)}" style="--node:${node.color}"/><text class="tnode__label" dy="0.32em" fill="${node.ink}">${String(node.index + 1).padStart(2, '0')}</text></g>`).join('');

  // SMIL packets are emitted only when motion is welcome.
  const packets = stillness.matches ? '' : [0, 1.3, 2.6].map(delay =>
    `<circle class="tpacket" r="2.6"><animateMotion dur="3.9s" begin="${delay}s" repeatCount="indefinite" path="M${gateway.x + 30} ${gateway.y} L${hub.x - 27} ${hub.y}"/></circle>`).join('');

  const guides = (twoRings ? [104, 156] : [132]).map(radius =>
    `<path class="tguide" d="M${(hub.x + Math.cos(-arc / 2) * radius).toFixed(1)} ${(hub.y + Math.sin(-arc / 2) * radius).toFixed(1)} A${radius} ${radius} 0 0 1 ${(hub.x + Math.cos(arc / 2) * radius).toFixed(1)} ${(hub.y + Math.sin(arc / 2) * radius).toFixed(1)}"/>`).join('');

  return `<svg class="topology-svg" viewBox="0 0 ${width} ${height}" role="img" preserveAspectRatio="xMidYMid meet" aria-label="Query gateway feeds a segment router that fans out to ${fmt(count)} published segments, sized by document load">
    <g class="tguides">${guides}</g>
    <line class="ttrunk" x1="${gateway.x + 30}" y1="${gateway.y}" x2="${hub.x - 27}" y2="${hub.y}"/>
    ${packets}
    <g class="tspokes">${spokes}</g>
    <g class="tgateway" transform="translate(${gateway.x} ${gateway.y})">
      <rect x="-30" y="-24" width="60" height="48" rx="11"/>
      <path class="tglyph" d="M-9 -6 h18 M-9 0 h18 M-9 6 h12"/>
      <text class="tlabel" y="38">Query gateway</text>
      <text class="tsub" y="50">HTTP search</text>
    </g>
    <g class="thub" transform="translate(${hub.x} ${hub.y})">
      <circle class="thub__ring" r="25"/>
      <circle class="thub__core" r="14"/>
      <text class="tlabel" y="-36">Segment router</text>
      <text class="tsub" y="-25">${esc(snapshot.layout)} layout</text>
    </g>
    <g class="tmarks">${marks}</g>
  </svg>`;
}

function renderIndexTopology(index) {
  const target = document.querySelector('#index-topology');
  const snapshot = index?.snapshot;
  if (!snapshot) {
    target.innerHTML = '<section class="panel"><div class="empty-state tall">The index has no published segment snapshot yet.</div></section>';
    return;
  }
  const segments = snapshot.segments || [];
  if (!segments.length) {
    target.innerHTML = '<section class="panel"><div class="empty-state tall">No segments have been published.</div></section>';
    return;
  }
  const { total, average, smallest, largest, spread } = segmentStats(segments);
  const cards = segments.map((segment, position) => `
    <article class="segment-card" data-index="${position}">
      <div class="segment-heading">
        <div><strong>Segment ${String(position + 1).padStart(2, '0')}</strong><small>${esc(segment.index_name ? `${segment.index_name} · ${segment.segment_id.split('/').slice(1).join('/')}` : segment.segment_id)}</small></div>
        <span class="badge active">ready</span>
      </div>
      <div class="segment-count"><strong>${fmt(segment.document_count)}</strong><span>documents</span></div>
      <div class="segment-profile">
        <span>Terms <b>${fmt(segment.profile_term_count)}</b></span>
        <span>Entities <b>${fmt(segment.profile_entity_count)}</b></span>
        <span>Topics <b>${fmt(segment.profile_topic_count)}</b></span>
        <span>Local memory <b>${fmt(segment.profile_local_memory_count)}</b></span>
      </div>
    </article>`).join('');

  target.innerHTML = `
    <section class="panel">
      <div class="panel__head">
        <div><p class="eyebrow">Query routing</p><h2>Segment topology</h2></div>
        <span class="panel__note">${esc(snapshot.layout)} · revision ${fmt(index.snapshot_revision)}</span>
      </div>
      <div class="topology-stage">
        ${topologyDiagram(snapshot, segments, largest)}
        <div class="topology-readout">
          <div class="topology-readout__head"><p class="eyebrow">Hovered segment</p><strong id="topology-readout-title">All segments</strong></div>
          <p id="topology-readout-body" class="muted">${fmt(total)} documents across ${fmt(segments.length)} published segments.</p>
          <div class="load-legend">
            <span>Document load</span>
            <div class="load-ramp">${LOAD_RAMP.map(step => `<i style="background:${step}"></i>`).join('')}</div>
            <div class="load-ends"><small>low</small><small>high</small></div>
          </div>
          <div class="distribution-stats">
            <div><span>Smallest</span><strong>${fmt(smallest)}</strong></div>
            <div><span>Average</span><strong>${fmt(Math.round(average))}</strong></div>
            <div><span>Largest</span><strong>${fmt(largest)}</strong></div>
            <div><span>Spread</span><strong>${pct(spread)}</strong></div>
          </div>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel__head">
        <div><p class="eyebrow">Detail</p><h2>All segments</h2></div>
        <span class="panel__note">${fmt(segments.length)} segments · ${fmt(total)} documents</span>
      </div>
      <div class="segment-grid">${cards}</div>
    </section>`;

  wireTopology(target, segments, total);
}

// Hovering a node lifts its spoke and fills the readout; the matching detail
// card highlights too, so the diagram and the list stay tied together.
function wireTopology(target, segments, total) {
  const title = target.querySelector('#topology-readout-title');
  const body = target.querySelector('#topology-readout-body');
  const clear = () => {
    target.querySelectorAll('.tspoke.lit').forEach(spoke => spoke.classList.remove('lit'));
    target.querySelectorAll('.segment-card.lit').forEach(card => card.classList.remove('lit'));
    title.textContent = 'All segments';
    body.textContent = `${fmt(total)} documents across ${fmt(segments.length)} published segments.`;
  };
  target.querySelectorAll('.tnode').forEach(node => {
    const position = Number(node.dataset.index);
    node.addEventListener('mouseenter', () => {
      clear();
      target.querySelector(`#tspoke-${position}`)?.classList.add('lit');
      target.querySelector(`.segment-card[data-index="${position}"]`)?.classList.add('lit');
      const segment = segments[position];
      title.textContent = `Segment ${String(position + 1).padStart(2, '0')}`;
      body.textContent = `${fmt(segment.document_count)} documents · ${fmt(segment.profile_term_count)} terms · ${fmt(segment.profile_entity_count)} entities`;
    });
  });
  target.querySelector('.topology-svg')?.addEventListener('mouseleave', clear);
}

/* ── providers ─────────────────────────────────────────────────── */

function renderProviders(items, projectIndex = projectIndexView(state.status?.index, items)) {
  const visible = items
    .filter(item => item.events_total > 0 || (item.indexes || []).length > 0)
    .sort((left, right) => Number(right.last_seen_ms || 0) - Number(left.last_seen_ms || 0) || left.provider.localeCompare(right.provider));

  const rail = document.querySelector('#rail-providers');
  if (!visible.length) {
    state.provider = null;
    rail.innerHTML = '<p class="rail__empty">No provider telemetry yet.</p>';
    document.querySelector('#provider-view').innerHTML = '<section class="panel"><div class="empty-state tall">Providers appear here once an integration creates an index or sends lifecycle telemetry.</div></section>';
    renderIndexMetrics(items, projectIndex);
    return;
  }
  if (!state.provider || !visible.some(item => item.provider === state.provider)) state.provider = visible[0].provider;

  rail.innerHTML = visible.map(item => {
    const tone = item.state === 'active' ? 'active' : item.state === 'idle' ? 'idle' : '';
    return `<button type="button" class="railprovider ${item.provider === state.provider ? 'selected' : ''}" data-provider="${esc(item.provider)}" title="${esc(item.compiled ? item.state.replace('_', ' ') : 'not compiled')}"><span class="dot ${tone}"></span><span class="railprovider__name">${esc(item.provider)}</span><span class="railprovider__count">${fmt(item.events_total)}</span></button>`;
  }).join('');
  rail.querySelectorAll('.railprovider').forEach(button => button.addEventListener('click', () => {
    state.provider = button.dataset.provider;
    state.tool = null;
    renderProviders(items, projectIndexView(state.status?.index, items));
    showView('providers');
  }));

  renderIndexMetrics(items, projectIndex);
  renderProvider(visible.find(item => item.provider === state.provider));
}

function renderIndexMetrics(providers, projectIndex) {
  const indexes = providers.flatMap(provider => provider.indexes || []);
  const fallback = state.status?.index || {};
  const documents = Number(projectIndex?.snapshot?.global_document_count ?? fallback.source_document_count ?? 0);
  const records = Number(fallback.record_count || 0)
    + indexes.reduce((sum, index) => sum + Number(index.record_count || 0), 0);
  const storeCount = Number(projectIndex?.snapshot?.store_count || (indexes.length ? indexes.length : 1));
  const scope = projectIndex?.snapshot
    ? `${fmt(storeCount)} project ${storeCount === 1 ? 'store' : 'stores'} · ${fmt(projectIndex.snapshot.segment_count)} segments`
    : 'primary query index';
  tween(document.querySelector('#documents'), [documents], ([value]) => fmt(Math.round(value)));
  tween(document.querySelector('#records'), [records], ([value]) => fmt(Math.round(value)));
  document.querySelector('#documents-scope').textContent = scope;
  document.querySelector('#records-scope').textContent = scope;
}

function renderFeed(providers) {
  const events = providers
    .flatMap(provider => (provider.events || []).map(event => ({ ...event, provider: provider.provider })))
    .sort((left, right) => right.timestamp_ms - left.timestamp_ms)
    .slice(0, 12);
  document.querySelector('#feed-note').textContent = events.length ? `${events.length} most recent events` : '';
  document.querySelector('#overview-feed').innerHTML = events.length
    ? events.map(event => `
      <div class="activity-item">
        <i class="activity-dot"></i>
        <div class="event-detail">
          <div class="tool-heading">
            <span class="feed-provider">${esc(event.provider)}</span>
            <strong>${esc(event.tool_name ? toolGroup(event.tool_name) : event.event || 'Lifecycle event')}</strong>
            ${event.tool_name ? `<span>${esc(toolDetail(event))}</span>` : ''}
          </div>
          <small>${esc(event.session_key || 'unknown session')} · ${esc(event.category || 'lifecycle')}</small>
        </div>
        <strong class="topbar__stamp">${clock(event.timestamp_ms)}</strong>
      </div>`).join('')
    : '<div class="empty-state">No provider events captured yet.</div>';
}

function sessionsFor(events) {
  const groups = new Map();
  events.forEach(event => { if (!groups.has(event.session_key)) groups.set(event.session_key, []); groups.get(event.session_key).push(event); });
  return [...groups]
    .map(([session_key, list]) => {
      list.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
      return { session_key, events: list, last_seen_ms: list.at(-1).timestamp_ms, last_event: list.at(-1).event };
    })
    .sort((a, b) => b.last_seen_ms - a.last_seen_ms);
}

function cycleAnalytics(events) {
  const isPrompt = event => event.prompt_preview || /^(UserPromptSubmit|UserPromptExpansion|BeforeAgent)$/i.test(event.event);
  const isStop = event => event.stop_response_preview || /^(Stop|AfterAgent|SessionEnd)$/i.test(event.event);
  const prompts = events.filter(isPrompt);
  const toolCalls = events.filter(event => event.tool_name);
  const responses = events.filter(event => event.tool_response_preview);
  const usageEvents = events.filter(event => [event.input_tokens, event.output_tokens, event.total_tokens].some(Number.isFinite));
  const sumTokens = field => usageEvents.reduce((sum, event) => sum + Number(event[field] || 0), 0);
  const durations = [];
  sessionsFor(events).forEach(session => session.events.forEach((event, index) => {
    if (!isPrompt(event)) return;
    const stop = session.events.slice(index + 1).find(isStop);
    if (stop) durations.push(Math.max(0, stop.timestamp_ms - event.timestamp_ms));
  }));
  return {
    prompts: prompts.length,
    toolCalls: toolCalls.length,
    responses: responses.length,
    stops: durations.length,
    completionRate: prompts.length ? durations.length / prompts.length : 0,
    averageTools: prompts.length ? toolCalls.length / prompts.length : 0,
    averageDuration: durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : 0,
    tokenUsage: tokenAccounting(usageEvents, prompts.length)
  };
}

// The three input counters are DISJOINT: `input_tokens` is the uncached
// remainder only, with cached context reported separately as cache writes and
// cache reads. Real context sent = all three summed. The provider's own
// `total_tokens` ignores both cache fields, so it is recomputed here rather
// than trusted — reading it directly under-reports a cached turn enormously.
function tokenAccounting(usageEvents, promptCount) {
  const sum = field => usageEvents.reduce((total, event) => total + Number(event[field] || 0), 0);
  const fresh = sum('input_tokens');
  const cacheWrite = sum('cache_creation_input_tokens');
  const cacheRead = sum('cache_read_input_tokens');
  const output = sum('output_tokens');
  const contextIn = fresh + cacheWrite + cacheRead;
  const total = contextIn + output;
  return {
    available: usageEvents.length > 0,
    fresh, cacheWrite, cacheRead, output, contextIn, total,
    cacheHitRate: contextIn ? cacheRead / contextIn : 0,
    averageTotal: promptCount ? total / promptCount : 0,
    reportedTotal: sum('total_tokens')
  };
}

function delegationAnalytics(events) {
  const starts = events.filter(event => event.event === 'SubagentStart');
  const stops = events.filter(event => event.event === 'SubagentStop');
  const usageByAgent = new Map();
  events.filter(event => event.event === 'TurnUsage' && (event.agent_id || event.agent_type)).forEach(event => {
    const key = event.agent_id || event.agent_type;
    const usage = usageByAgent.get(key) || { fresh: 0, cacheWrite: 0, cacheRead: 0, output: 0 };
    usage.fresh += Number(event.input_tokens || 0);
    usage.cacheWrite += Number(event.cache_creation_input_tokens || 0);
    usage.cacheRead += Number(event.cache_read_input_tokens || 0);
    usage.output += Number(event.output_tokens || 0);
    usageByAgent.set(key, usage);
  });
  const usedStops = new Set();
  const runs = starts.map(start => {
    const stopIndex = stops.findIndex((stop, index) => !usedStops.has(index)
      && (start.agent_id && stop.agent_id ? start.agent_id === stop.agent_id : stop.timestamp_ms >= start.timestamp_ms));
    const stop = stopIndex >= 0 ? stops[stopIndex] : null;
    if (stop) usedStops.add(stopIndex);
    return { start, stop, duration: stop ? Math.max(0, stop.timestamp_ms - start.timestamp_ms) : null };
  });
  const known = new Set(runs.map(run => run.start.agent_id || run.start.agent_type).filter(Boolean));
  usageByAgent.forEach((usage, key) => {
    if (!known.has(key)) runs.push({ start: { agent_id: key }, stop: null, duration: null, usageOnly: true });
  });
  const completed = runs.filter(run => run.stop);
  return {
    starts: starts.length,
    completed: completed.length,
    active: runs.filter(run => !run.stop).length,
    averageDuration: completed.length ? completed.reduce((sum, run) => sum + run.duration, 0) / completed.length : 0,
    usageByAgent,
    runs: runs.sort((a, b) => (b.start.timestamp_ms || 0) - (a.start.timestamp_ms || 0)).slice(0, 8)
  };
}

function renderCyclePanel(stats) {
  const step = (label, value, note) => `<div class="flow-step"><em>${value}</em><strong>${label}</strong><small>${note}</small></div>`;
  return `
    <section class="panel">
      <div class="panel__head">
        <div><p class="eyebrow">Agent run cycle</p><h2>Execution flow</h2></div>
        <span class="panel__note">Prompt to final response</span>
      </div>
      <div class="flow">
        ${step('Prompts', fmt(stats.prompts), 'submitted')}
        ${step('Tool work', fmt(stats.toolCalls), 'invocations')}
        ${step('Responses', fmt(stats.responses), 'returned')}
        ${step('Stops', fmt(stats.stops), 'completed')}
      </div>
      <div class="figure-row">
        <div><span>Completion rate</span><strong>${pct(stats.completionRate)}</strong></div>
        <div><span>Avg. tools / prompt</span><strong>${stats.averageTools.toFixed(1)}</strong></div>
        <div><span>Avg. cycle time</span><strong>${seconds(stats.averageDuration)}</strong></div>
      </div>
    </section>`;
}

function renderTokenPanel(stats) {
  const usage = stats.tokenUsage;
  const head = `
    <div class="panel__head">
      <div><p class="eyebrow">Token usage</p><h2>Context accounting</h2></div>
      <span class="panel__note">Cache-aware</span>
    </div>`;
  if (!usage.available) {
    return `<section class="panel">${head}<div class="empty-state">No provider token usage has been reported yet.</div></section>`;
  }
  const value = (label, amount, note) => `<div><span>${label}</span><strong>${amount}</strong><small>${note}</small></div>`;
  const parts = [
    { key: 'fresh', label: 'Fresh input', amount: usage.fresh },
    { key: 'write', label: 'Cache write', amount: usage.cacheWrite },
    { key: 'read', label: 'Cache read', amount: usage.cacheRead }
  ].filter(part => part.amount > 0);
  // Part-to-whole across the context window: a stacked bar, with a 2px surface
  // gap between fills and every segment directly labelled in the legend.
  const composition = usage.contextIn ? `
    <div class="token-bar" role="img" aria-label="Context composition: ${parts.map(part => `${part.label} ${fmt(part.amount)} tokens`).join(', ')}">
      ${parts.map(part => `<i class="token-bar__part token-bar__part--${part.key}" style="flex:${part.amount}"></i>`).join('')}
    </div>
    <div class="token-key">
      ${parts.map(part => `<span><i class="token-bar__part--${part.key}"></i>${part.label} <b>${pct(part.amount / usage.contextIn)}</b></span>`).join('')}
    </div>` : '';
  return `
    <section class="panel">${head}
      <div class="figure-row" style="margin-top:0;padding-top:0;border-top:0">
        ${value('Context in', fmt(usage.contextIn), 'fresh + cached')}
        ${value('Output', fmt(usage.output), 'response generated')}
        ${value('Total', fmt(usage.total), 'context + output')}
        ${value('Cache hit rate', pct(usage.cacheHitRate), 'context served from cache')}
      </div>
      ${composition}
      <p class="token-note">Avg. ${fmt(Math.round(usage.averageTotal))} tokens / prompt. Provider-reported total is ${fmt(usage.reportedTotal)} — it counts neither cache reads nor cache writes.</p>
    </section>`;
}

function renderDelegationPanel(events) {
  const stats = delegationAnalytics(events);
  if (!stats.starts && !stats.completed && !stats.usageByAgent.size) return '';
  const rows = stats.runs.map(run => {
    const label = run.start.agent_type || run.start.agent_id || 'Subagent';
    const id = run.start.agent_type ? run.start.agent_id || '' : '';
    const status = run.stop ? 'completed' : 'running';
    const usage = stats.usageByAgent.get(run.start.agent_id || run.start.agent_type);
    const usageTotal = usage ? usage.fresh + usage.cacheWrite + usage.cacheRead + usage.output : 0;
    return `
      <div class="delegation-row">
        <div><strong>${esc(label)}</strong><small>${esc(id || 'delegated task')} · ${clock(run.start.timestamp_ms)}${usage ? ` · ${fmt(usageTotal)} tokens` : ''}</small></div>
        <span class="delegation-duration ${status}">${run.duration == null ? 'in progress' : seconds(run.duration)}</span>
        <span class="badge ${run.stop ? 'active' : 'idle'}">${status}</span>
      </div>`;
  }).join('');
  return `
    <section class="panel">
      <div class="panel__head">
        <div><p class="eyebrow">Delegation</p><h2>Subagent activity</h2></div>
        <span class="panel__note">Lifecycle hooks</span>
      </div>
      <div class="figure-row" style="margin-top:0;padding-top:0;border-top:0">
        <div><span>Started</span><strong>${fmt(stats.starts)}</strong></div>
        <div><span>Completed</span><strong>${fmt(stats.completed)}</strong></div>
        <div><span>Active</span><strong>${fmt(stats.active)}</strong></div>
        <div><span>Avg. duration</span><strong>${seconds(stats.averageDuration)}</strong></div>
      </div>
      <div>${rows}</div>
    </section>`;
}

function renderStores(indexes) {
  const groups = ['mcp', 'memory'].map(role => {
    const roleIndexes = indexes.filter(index => index.role === role);
    if (!roleIndexes.length) return '';
    const cards = roleIndexes.map(index => `
      <article class="store-card">
        <div class="segment-heading">
          <div><strong>${role === 'mcp' ? 'Legacy query index' : 'Captured memories'}</strong><small>${esc(index.name)}</small></div>
          <span class="badge ${index.dirty ? 'idle' : 'active'}">${index.dirty ? 'dirty' : 'ready'}</span>
        </div>
        <div class="store-counts">
          <span><b>${fmt(index.source_document_count)}</b> documents</span>
          <span><b>${fmt(index.record_count)}</b> records</span>
          <span><b>${fmt(index.snapshot?.segment_count || 0)}</b> segments</span>
        </div>
        <small>Revision ${fmt(index.snapshot_revision)}</small>
      </article>`).join('');
    return `
      <div class="store-group">
        <div class="store-group-heading">
          <strong>${role === 'mcp' ? 'Legacy MCP store' : 'Captured memory store'}</strong>
          <span>${roleIndexes.length} ${roleIndexes.length === 1 ? 'index' : 'indexes'}</span>
        </div>
        <div class="store-grid">${cards}</div>
      </div>`;
  }).join('');
  return groups || '<div class="empty-state">No provider memory directories with metadata.json were found.</div>';
}

function renderProvider(provider) {
  if (!provider) return;
  const events = (provider.events || []).slice().sort((a, b) => b.timestamp_ms - a.timestamp_ms);
  const toolEvents = events.filter(event => event.tool_name);
  const sessions = sessionsFor(events);
  const indexes = provider.indexes || [];
  const current = sessions[0];
  const live = current && Date.now() - current.last_seen_ms < 60_000;

  const counts = toolEvents.reduce((result, event) => {
    const group = toolGroup(event.tool_name);
    result[group] = (result[group] || 0) + 1;
    return result;
  }, {});
  const tools = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (state.tool && !counts[state.tool]) state.tool = null;

  const detailsByGroup = toolEvents.reduce((result, event) => {
    const group = toolGroup(event.tool_name);
    result[group] ||= {};
    const detail = toolDetail(event);
    result[group][detail] = (result[group][detail] || 0) + 1;
    return result;
  }, {});
  const calls = toolEvents.filter(event => !state.tool || toolGroup(event.tool_name) === state.tool).slice(0, 30);
  const details = Object.entries(detailsByGroup[state.tool] || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const stats = cycleAnalytics(events);

  document.querySelector('#provider-view').innerHTML = `
    <div class="provider-head">
      <div>
        <div class="provider-head__id">
          <h2>${esc(provider.provider)}</h2>
          <span class="badge ${esc(provider.state)}">${provider.compiled ? esc(provider.state.replace('_', ' ')) : 'not compiled'}</span>
        </div>
        <p>${esc(provider.note)}</p>
      </div>
      <div class="provider-head__last"><span>Last activity</span><strong>${clock(provider.last_seen_ms)}</strong></div>
    </div>

    <div class="row row--halves">${renderCyclePanel(stats)}${renderTokenPanel(stats)}</div>
    ${renderDelegationPanel(events)}

    <div class="stat-row">
      <article class="stat"><span>Documents</span><strong>${fmt(indexes.reduce((sum, index) => sum + Number(index.source_document_count || 0), 0))}</strong><small>${fmt(indexes.length)} private memory stores</small></article>
      <article class="stat"><span>Records</span><strong>${fmt(indexes.reduce((sum, index) => sum + Number(index.record_count || 0), 0))}</strong><small>across private memory stores</small></article>
      <article class="stat"><span>Sessions</span><strong>${fmt(sessions.length)}</strong><small>${fmt(provider.sessions_active)} currently active</small></article>
      <article class="stat"><span>Tool calls</span><strong>${fmt(toolEvents.length)}</strong><small>${fmt(tools.length)} categories</small></article>
    </div>

    <div class="row row--narrow-first">
      <section class="panel">
        <div class="sub-head">
          <div><p class="eyebrow">Session history</p><h2>Sessions</h2></div>
          <span class="panel__note">${sessions.length ? `${sessions.length} observed` : 'None'}</span>
        </div>
        <div class="scroll-list">${renderSessions(sessions)}</div>
      </section>
      <section class="panel">
        <div class="sub-head">
          <div><p class="eyebrow">Live session</p><h2>${live ? 'Current activity' : 'Latest activity'}</h2></div>
          <span class="live-indicator ${live ? 'on' : ''}"><i></i>${live ? 'LIVE' : 'IDLE'}</span>
        </div>
        ${renderLive(current)}
      </section>
    </div>

    <div class="row row--halves">
      <section class="panel">
        <div class="sub-head">
          <div><p class="eyebrow">Private memory</p><h2>Memory stores</h2></div>
          <span class="panel__note">${indexes.length ? `${indexes.length} discovered` : 'None found'}</span>
        </div>
        ${renderStores(indexes)}
      </section>
      <section class="panel">
        <div class="sub-head">
          <div><p class="eyebrow">Tool categories</p><h2>Frequently used tools</h2></div>
          <span class="panel__note">Select for details</span>
        </div>
        <div class="tool-grid">${tools.length ? tools.map(([name, count]) => {
          const top = Object.entries(detailsByGroup[name] || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3);
          return `<button type="button" class="tool-card ${state.tool === name ? 'selected' : ''}" data-tool="${esc(name)}">
            <div class="tool-card-title"><strong>${esc(name)}</strong><span>${fmt(count)}</span></div>
            <div class="tool-meter"><i style="width:${Math.max(6, count / tools[0][1] * 100)}%"></i></div>
            ${top.length ? `<div class="tool-breakdown">${top.map(([detail, detailCount]) => `<div><span>${esc(detail)}</span><strong>${fmt(detailCount)}</strong></div>`).join('')}</div>` : ''}
          </button>`;
        }).join('') : '<div class="empty-state">No tool calls captured yet.</div>'}</div>
        ${state.tool ? `<div class="detail-breakdown"><div class="breakdown-heading"><span>${esc(state.tool)} details</span><span>${details.length} distinct</span></div>${details.map(([name, count]) => `<div class="breakdown-row"><span>${esc(name)}</span><strong>${fmt(count)}</strong></div>`).join('')}</div>` : ''}
      </section>
    </div>

    <section class="panel">
      <div class="sub-head">
        <div><p class="eyebrow">Call history</p><h2>${state.tool ? esc(state.tool) : 'All tool categories'}</h2></div>
        <span class="panel__note">${calls.length} most recent</span>
      </div>
      <div>${calls.length ? calls.map(event => renderToolCall(event, 'history')).join('') : '<div class="empty-state">No tool calls match this filter.</div>'}</div>
    </section>`;

  document.querySelectorAll('.tool-card').forEach(button => button.addEventListener('click', () => {
    state.tool = state.tool === button.dataset.tool ? null : button.dataset.tool;
    renderProvider(provider);
  }));
  document.querySelectorAll('[data-flip-card]').forEach(card => {
    const flip = () => card.classList.toggle('flipped');
    card.addEventListener('click', flip);
    card.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); flip(); }
    });
  });
}

function renderSessions(sessions) {
  if (!sessions.length) return '<div class="empty-state">No provider sessions observed yet.</div>';
  return sessions.slice(0, 12).map((session, index) => `
    <div class="session-item ${index === 0 ? 'latest' : ''}">
      <div><span class="session-key">${esc(session.session_key)}</span><small>${fmt(session.events.length)} events · ${esc(session.last_event)}</small></div>
      <strong>${clock(session.last_seen_ms)}</strong>
    </div>`).join('');
}

function renderLive(session) {
  if (!session) return '<div class="empty-state tall">Start a provider session to see live activity here.</div>';
  const events = session.events.slice().reverse().slice(0, 8);
  return `
    <div class="live-summary">
      <div><span class="session-key">${esc(session.session_key)}</span><small>${fmt(session.events.length)} events in this session</small></div>
      <strong>${clock(session.last_seen_ms)}</strong>
    </div>
    <div class="live-list">${events.map(event => renderEventDetail(event, 'live')).join('')}</div>`;
}

/* ── command palette ───────────────────────────────────────────── */

const palette = document.querySelector('#palette');
const paletteInput = document.querySelector('#palette-input');
const paletteResults = document.querySelector('#palette-results');
let paletteItems = [];
let paletteActive = 0;

function paletteCommands() {
  const commands = Object.entries(VIEWS).map(([view, meta]) => ({
    label: meta.title, hint: 'View', run: () => showView(view)
  }));
  (state.status?.integrations || [])
    .filter(item => item.events_total > 0 || (item.indexes || []).length > 0)
    .forEach(item => commands.push({
      label: item.provider,
      hint: `Provider · ${fmt(item.events_total)} events`,
      run: () => { state.provider = item.provider; state.tool = null; renderProviders(state.status.integrations || []); showView('providers'); }
    }));
  commands.push({ label: 'Refresh now', hint: 'Action', run: refresh });
  return commands;
}

function renderPalette() {
  const query = paletteInput.value.trim().toLowerCase();
  paletteItems = paletteCommands().filter(item => !query || item.label.toLowerCase().includes(query));
  paletteActive = Math.min(paletteActive, Math.max(0, paletteItems.length - 1));
  paletteResults.innerHTML = paletteItems.length
    ? paletteItems.map((item, index) => `<button type="button" role="option" aria-selected="${index === paletteActive}" class="palette__item ${index === paletteActive ? 'active' : ''}" data-index="${index}"><span>${esc(item.label)}</span><small>${esc(item.hint)}</small></button>`).join('')
    : '<div class="empty-state">Nothing matches.</div>';
  paletteResults.querySelectorAll('.palette__item').forEach(button => {
    button.addEventListener('mousemove', () => { paletteActive = Number(button.dataset.index); renderPalette(); });
    button.addEventListener('click', () => runPalette(Number(button.dataset.index)));
  });
}

function openPalette() {
  palette.hidden = false;
  paletteInput.value = '';
  paletteActive = 0;
  renderPalette();
  paletteInput.focus();
}

function closePalette() { palette.hidden = true; }

function runPalette(index) {
  const item = paletteItems[index];
  if (!item) return;
  closePalette();
  item.run();
}

paletteInput.addEventListener('input', () => { paletteActive = 0; renderPalette(); });
palette.addEventListener('click', event => { if (event.target === palette) closePalette(); });
window.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    palette.hidden ? openPalette() : closePalette();
    return;
  }
  if (palette.hidden) return;
  if (event.key === 'Escape') { event.preventDefault(); closePalette(); }
  if (event.key === 'ArrowDown') { event.preventDefault(); paletteActive = (paletteActive + 1) % (paletteItems.length || 1); renderPalette(); }
  if (event.key === 'ArrowUp') { event.preventDefault(); paletteActive = (paletteActive - 1 + paletteItems.length) % (paletteItems.length || 1); renderPalette(); }
  if (event.key === 'Enter') { event.preventDefault(); runPalette(paletteActive); }
});

/* ── boot ──────────────────────────────────────────────────────── */

showView(location.hash.slice(1) || 'overview');
refresh();
setInterval(refresh, 5000);
window.addEventListener('resize', drawSparks);
