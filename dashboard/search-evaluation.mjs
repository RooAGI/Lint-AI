const escapeHtml = value => String(value).replace(/[&<>\'"]/g, character => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  "'": '&#39;',
  '"': '&quot;'
}[character]));

const formatCount = value => Number(value || 0).toLocaleString('en-US');

function renderMetric(metric) {
  return metric && metric.value != null
    ? `${Number(metric.value).toFixed(2)} <small>n=${formatCount(metric.n || 0)}</small>`
    : 'Not enough judged searches';
}

function renderLabelCounts(counts) {
  return Object.entries(counts || {})
    .map(([label, count]) => `${label.replaceAll('_', ' ')} ${formatCount(count)}`)
    .join(' · ');
}

export function renderSearchEvaluationHtml(evaluation = {}) {
  const smallSample = evaluation.sample_status !== 'usable';
  const overallLabels = renderLabelCounts(evaluation.overall_labels);
  const relevanceLabels = renderLabelCounts(evaluation.relevance_labels);
  return `<section class="review-evaluation__panel">
    <div class="review-evaluation__head"><div><p class="eyebrow">Offline ranking evaluation</p><h3>Feedback summary</h3></div>
      <span class="review-label ${smallSample ? 'review-label--pending' : 'review-label--relevant'}">${smallSample ? 'Small sample' : '30+ judged searches'}</span></div>
    <p class="muted">${formatCount(evaluation.reviewed_searches || 0)} reviewed searches · ${formatCount(evaluation.complete_judged_searches || 0)} complete ranked judgments · ${formatCount(evaluation.judged_documents || 0)} document judgments</p>
    <div class="review-evaluation__metrics">
      <div><span>NDCG@5</span><strong>${renderMetric(evaluation.ndcg_at_5)}</strong></div>
      <div><span>NDCG@10</span><strong>${renderMetric(evaluation.ndcg_at_10)}</strong></div>
      <div><span>MRR</span><strong>${renderMetric(evaluation.mrr)}</strong></div>
    </div>
    <p class="muted">Overall labels: ${escapeHtml(overallLabels || 'none')}<br>Document relevance: ${escapeHtml(relevanceLabels || 'none')}</p>
    <p class="review-evaluation__note">${smallSample
      ? 'Metrics describe the currently reviewed searches. The sample is too small to assess ranking reliably; feedback does not train or change live ranking.'
      : 'This is an offline evaluation of recorded ranked results. Feedback does not train or change live ranking.'}</p>
  </section>`;
}
