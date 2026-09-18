function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[character]));
}

function eventTime(timestamp) {
  return timestamp ? new Date(timestamp).toLocaleTimeString() : '—';
}

function argumentObject(event) {
  if (!event.tool_arguments) return {};
  try { return JSON.parse(event.tool_arguments); } catch { return {}; }
}

export function toolGroup(name) {
  const value = String(name || 'Unknown tool');
  if (/^mcp__/i.test(value)) return 'MCP';
  if (/^(apply_patch|patch|edit|write_file|str_replace)/i.test(value)) return 'Patch';
  if (/^(bash|shell|sh|zsh|exec|terminal|run_command|rg|grep|find|ls|cat|sed|pwd|git|cargo|npm|node|curl|mkdir|cp|mv|rm)$/i.test(value)) return 'Bash';
  return value;
}

function patchText(event) {
  const args = argumentObject(event);
  return String(args.patch || args.input || args.content || args.command || '');
}

function symbolsFor(file, text) {
  const extension = file.split('.').pop()?.toLowerCase();
  const patterns = {
    js: /(?:async\s+)?function\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>|class\s+([\w$]+)/g,
    jsx: /(?:async\s+)?function\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>|class\s+([\w$]+)/g,
    ts: /(?:async\s+)?function\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>|class\s+([\w$]+)/g,
    tsx: /(?:async\s+)?function\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>|class\s+([\w$]+)/g,
    py: /(?:async\s+)?def\s+([\w$]+)|class\s+([\w$]+)/g,
    rs: /(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([\w$]+)|(?:struct|enum)\s+([\w$]+)/g,
    go: /func\s+(?:\([^)]*\)\s*)?([\w$]+)|type\s+([\w$]+)/g
  };
  const pattern = patterns[extension];
  if (!pattern) return [];
  return [...text.matchAll(pattern)].map(match => match.slice(1).find(Boolean)).filter(Boolean);
}

function patchDetails(event) {
  const text = patchText(event);
  if (!text) return [];
  const files = [...text.matchAll(/\*\*\* (Update|Add|Delete) File: ([^\n]+)/g)];
  if (!files.length) return [];
  return files.map((match, index) => {
    const file = match[2].trim();
    const body = text.slice(match.index, files[index + 1]?.index ?? text.length);
    const symbols = [...new Set(symbolsFor(file, body))].slice(0, 3);
    const hunks = [...body.matchAll(/@@ -([\d]+)(?:,([\d]+))? \+([\d]+)(?:,([\d]+))?/g)];
    const additions = body.split('\n').filter(line => line.startsWith('+') && !line.startsWith('+++')).length;
    const removals = body.split('\n').filter(line => line.startsWith('-') && !line.startsWith('---')).length;
    const lineRange = hunks.length ? `lines ${hunks[0][3]}${hunks[0][4] ? `–${Number(hunks[0][3]) + Number(hunks[0][4]) - 1}` : ''}` : null;
    const summary = symbols.length ? symbols.join(', ') : lineRange || 'file contents';
    const change = match[1].toLowerCase() === 'add' ? 'added' : match[1].toLowerCase() === 'delete' ? 'deleted' : `+${additions}/−${removals}`;
    return `${file} · ${summary} · ${change}`;
  });
}

export function toolDetail(event) {
  const name = String(event.tool_name || 'Unknown tool');
  const args = argumentObject(event);
  if (/^mcp__/i.test(name)) {
    const parts = name.split('__').filter(Boolean);
    return parts.length > 2 ? `${parts[1]} · ${parts.slice(2).join('__')}` : parts.slice(1).join('__');
  }
  if (toolGroup(name) === 'Patch') {
    const details = patchDetails(event);
    return details.length ? details.join(' | ') : name;
  }
  if (toolGroup(name) === 'Bash') {
    const command = String(args.command || args.cmd || args.script || args.input || '');
    return command.trim().split(/\s+/)[0] || name;
  }
  return name;
}

function preview(event, label, field) {
  if (!event[field]) return '';
  return `<div class="event-field"><span>${label}</span><pre>${escapeHtml(event[field])}</pre></div>`;
}

function toolSummary(event) {
  const group = toolGroup(event.tool_name);
  if (group === 'Patch') return patchDetails(event).join('\n') || 'Patch content';
  if (group === 'Bash') {
    const args = argumentObject(event);
    return String(args.command || args.cmd || args.script || args.input || toolDetail(event)).trim();
  }
  return toolDetail(event);
}

export function renderToolCall(event, variant = 'history') {
  const response = preview(event, 'Tool response', 'tool_response_preview');
  const group = toolGroup(event.tool_name);
  const detail = toolDetail(event);
  const summaryLabel = group === 'Bash' ? 'Parsed command' : group === 'Patch' ? 'Parsed changes' : 'Parsed operation';
  const parsed = `<div class="tool-face tool-face-parsed"><span>${summaryLabel}</span><pre>${escapeHtml(toolSummary(event))}</pre><small>Click to view raw arguments</small></div>`;
  const raw = `<div class="tool-face tool-face-raw"><span>Raw arguments</span><pre>${escapeHtml(event.tool_arguments || '{}')}</pre><small>Click to return to parsed view</small></div>`;
  const card = `<div class="tool-flip" role="button" tabindex="0" aria-label="Toggle parsed and raw tool call" data-flip-card><div class="tool-flip-inner">${parsed}${raw}</div></div>${response}`;
  if (variant === 'live') return `<div class="activity-item tool-call-item"><i class="activity-dot"></i><div class="event-detail"><div class="tool-heading"><strong>${escapeHtml(group)}</strong><span>${escapeHtml(detail)}</span><em>${escapeHtml(event.event)}</em></div><small>${escapeHtml(event.category || 'tool')} · ${eventTime(event.timestamp_ms)}</small>${card}</div></div>`;
  return `<div class="call-row tool-call-item"><div class="call-main"><div><strong>${escapeHtml(group)}</strong><span class="tool-detail">${escapeHtml(detail)}</span><span class="event-chip">${escapeHtml(event.event)}</span></div><small>${escapeHtml(event.session_key || 'unknown')} · ${eventTime(event.timestamp_ms)}</small></div><div class="event-detail">${card}</div></div>`;
}

export function renderEventDetail(event, variant = 'live') {
  if (event.retrieved_memory_details) return renderMemoryRetrieval(event, variant);
  if (event.tool_name) return renderToolCall(event, variant === 'live' ? 'live' : 'history');
  const detail = [
    preview(event, 'Prompt', 'prompt_preview'),
    preview(event, 'Tool response', 'tool_response_preview'),
    preview(event, 'Final response', 'stop_response_preview')
  ].join('');
  return `<div class="activity-item"><i class="activity-dot"></i><div class="event-detail"><strong>${escapeHtml(event.event || 'Lifecycle event')}</strong><small>${escapeHtml(event.category || 'lifecycle')} · ${eventTime(event.timestamp_ms)}</small><div class="event-previews">${detail}</div></div></div>`;
}

function renderMemoryRetrieval(event, variant) {
  let memories = [];
  try { memories = JSON.parse(event.retrieved_memory_details) || []; } catch { memories = []; }
  const rows = memories.map((memory, index) => `<article class="retrieval-memory"><div><span class="retrieval-rank">#${index + 1}</span><strong>${escapeHtml(memory.type || 'memory')}</strong><small>${escapeHtml(memory.source || 'unknown source')}</small></div><span class="retrieval-score">${Number(memory.score || 0).toFixed(2)}</span><pre>${escapeHtml(memory.excerpt || '')}</pre></article>`).join('');
  const body = `<div class="retrieval-meta"><span>Query</span><strong>${escapeHtml(event.retrieval_query_preview || '—')}</strong><span>${event.retrieved_memory_count || memories.length} injected · ${event.retrieval_latency_ms || 0} ms</span></div><div class="retrieval-list">${rows || '<div class="empty-state">No memories were selected.</div>'}</div>`;
  return `<div class="${variant === 'live' ? 'activity-item' : 'call-row'} retrieval-item"><i class="activity-dot"></i><div class="event-detail"><div class="tool-heading"><strong>Retrieved memory</strong><span>${escapeHtml(event.event)}</span><em>${escapeHtml(event.category || 'retrieval')}</em></div><small>${eventTime(event.timestamp_ms)}</small>${body}</div></div>`;
}
