import assert from 'node:assert/strict';
import test from 'node:test';
import { renderCaptureProgressHtml } from './capture-progress.mjs';
test('shows pending bytes without claiming capture completed', () => {
  const html = renderCaptureProgressHtml([{provider:'codex',session_id:'session-1',status:'pending',captured_bytes:146_000_000,target_bytes:294_000_000,remaining_bytes:148_000_000}]);
  assert.match(html,/Captured 146\.00 MB \/ 294\.00 MB · Remaining 148\.00 MB/);
  assert.match(html,/pending/);
  assert.match(html,/No completed batch yet/);
});
test('escapes session and error strings', () => {
  const html=renderCaptureProgressHtml([{provider:'claude',session_id:'<script>',status:'retrying',last_error:'<img onerror="bad">'}]);
  assert.doesNotMatch(html,/<script>|<img/);
  assert.match(html,/&lt;img onerror=&quot;bad&quot;&gt;/);
});
test('distinguishes empty and unavailable capture status', () => {
  assert.match(renderCaptureProgressHtml([]),/No transcript captures/);
  assert.match(renderCaptureProgressHtml(null),/unavailable/);
});
