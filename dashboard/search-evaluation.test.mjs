import assert from 'node:assert/strict';
import test from 'node:test';
import { renderSearchEvaluationHtml } from './search-evaluation.mjs';

test('renders metric values with their judged-search sample counts', () => {
  const html = renderSearchEvaluationHtml({
    reviewed_searches: 12,
    complete_judged_searches: 10,
    judged_documents: 47,
    sample_status: 'small',
    overall_labels: { helpful: 8, partially_helpful: 3, not_helpful: 1 },
    relevance_labels: { relevant: 25, partially_relevant: 12, not_relevant: 10 },
    ndcg_at_5: { value: 0.87654, n: 10 },
    ndcg_at_10: { value: 0.91234, n: 10 },
    mrr: { value: 0.76543, n: 10 }
  });

  assert.match(html, /12 reviewed searches · 10 complete ranked judgments · 47 document judgments/);
  assert.match(html, /NDCG@5<\/span><strong>0\.88 <small>n=10<\/small>/);
  assert.match(html, /NDCG@10<\/span><strong>0\.91 <small>n=10<\/small>/);
  assert.match(html, /MRR<\/span><strong>0\.77 <small>n=10<\/small>/);
  assert.match(html, /Small sample/);
  assert.match(html, /partially helpful 3/);
  assert.match(html, /partially relevant 12/);
});

test('renders an empty evaluation without fabricating metric scores', () => {
  const html = renderSearchEvaluationHtml({});

  assert.match(html, /0 reviewed searches · 0 complete ranked judgments · 0 document judgments/);
  assert.equal((html.match(/Not enough judged searches/g) || []).length, 3);
  assert.match(html, /Small sample/);
  assert.match(html, /feedback does not train or change live ranking/);
});

test('escapes feedback label text before inserting it into dashboard HTML', () => {
  const html = renderSearchEvaluationHtml({
    overall_labels: { '<script>alert(1)</script>': 1 },
    relevance_labels: { 'relevant&trusted': 1 }
  });

  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt; 1/);
  assert.match(html, /relevant&amp;trusted 1/);
});
