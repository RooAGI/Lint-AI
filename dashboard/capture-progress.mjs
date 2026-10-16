const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const bytes = value => `${(Math.max(0, Number(value) || 0) / 1_000_000).toFixed(2)} MB`;

export function renderCaptureProgressHtml(captures) {
  if (!Array.isArray(captures)) return '<p>Capture status unavailable.</p>';
  if (!captures.length) return '<p>No transcript captures have been submitted.</p>';
  return captures.map(capture => {
    const lastProgress = capture.last_progress_ms ? new Date(capture.last_progress_ms).toLocaleString() : 'No completed batch yet';
    return `<article class="search-review">
      <div><strong>${escapeHtml(capture.provider)} · ${escapeHtml(capture.session_id)}</strong> <span>${escapeHtml(capture.status)}</span></div>
      <p>Captured ${bytes(capture.captured_bytes)} / ${bytes(capture.target_bytes)} · Remaining ${bytes(capture.remaining_bytes)}</p>
      <p>Last progress: ${escapeHtml(lastProgress)}</p>
      ${capture.last_error ? `<p class="error">${escapeHtml(capture.last_error)}</p>` : ''}
    </article>`;
  }).join('');
}
