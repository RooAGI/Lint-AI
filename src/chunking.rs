use crate::ids::stable_chunk_id;
use crate::index::SectionChunk;
use crate::tier1::{RankedTerm, Tier1Entity};
use regex::Regex;

pub fn chunk_document_sections(content: &str, doc_id: &str) -> Vec<SectionChunk> {
    let heading_re = Regex::new(r"(?m)^#{1,6}\s+(.*)$").expect("valid heading regex");
    let mut chunks = Vec::new();
    let mut last_start = 0usize;
    let mut current_heading = "(document)".to_string();

    for cap in heading_re.captures_iter(content) {
        let m = match cap.get(0) {
            Some(v) => v,
            None => continue,
        };
        if m.start() > last_start {
            let body = content[last_start..m.start()].trim();
            if !body.is_empty() {
                let start_line = content[..last_start].lines().count().max(1);
                let end_line = content[..m.start()].lines().count().max(start_line);
                let chunk_id =
                    stable_chunk_id(doc_id, &current_heading, body, start_line, end_line);
                chunks.push(SectionChunk {
                    chunk_id,
                    heading: current_heading.clone(),
                    content: body.to_string(),
                    start_line,
                    end_line,
                    timestamp: None,
                    key_entities: Vec::new(),
                    important_terms: Vec::new(),
                });
            }
        }
        current_heading = cap
            .get(1)
            .map(|v| v.as_str().trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "(section)".to_string());
        last_start = m.end();
    }

    let tail = content[last_start..].trim();
    if !tail.is_empty() {
        let start_line = content[..last_start].lines().count().max(1);
        let end_line = content.lines().count().max(start_line);
        let chunk_id = stable_chunk_id(doc_id, &current_heading, tail, start_line, end_line);
        chunks.push(SectionChunk {
            chunk_id,
            heading: current_heading,
            content: tail.to_string(),
            start_line,
            end_line,
            timestamp: None,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }

    if chunks.is_empty() {
        let chunk_id = stable_chunk_id(
            doc_id,
            "(document)",
            content,
            1,
            content.lines().count().max(1),
        );
        chunks.push(SectionChunk {
            chunk_id,
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line: 1,
            end_line: content.lines().count().max(1),
            timestamp: None,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }
    chunks
}

pub fn chunk_document_lines(
    content: &str,
    doc_id: &str,
    lines_per_chunk: usize,
    overlap: usize,
) -> Vec<SectionChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![SectionChunk {
            chunk_id: format!("{}::0", doc_id),
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line: 1,
            end_line: 1,
            timestamp: None,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        }];
    }
    let size = lines_per_chunk.clamp(1, 500);
    let ov = overlap.min(size.saturating_sub(1));
    let step = (size - ov).max(1);

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + size).min(lines.len());
        let body = lines[start..end].join("\n").trim().to_string();
        if !body.is_empty() {
            let heading = format!("lines {}-{}", start + 1, end);
            let chunk_id = stable_chunk_id(doc_id, &heading, &body, start + 1, end);
            chunks.push(SectionChunk {
                chunk_id,
                heading,
                content: body,
                start_line: start + 1,
                end_line: end,
                timestamp: None,
                key_entities: Vec::new(),
                important_terms: Vec::new(),
            });
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
    if chunks.is_empty() {
        let chunk_id = stable_chunk_id(doc_id, "(document)", content, 1, lines.len().max(1));
        chunks.push(SectionChunk {
            chunk_id,
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line: 1,
            end_line: lines.len().max(1),
            timestamp: None,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }
    chunks
}

fn estimate_tokens(text: &str) -> usize {
    // CJK-aware estimate: the old split_whitespace()*1.3 counted a whole
    // Chinese paragraph as one "word" (~1 token), grossly undercounting.
    // Latin words keep the historical 1.3 factor; each Han/Hangul char
    // counts ~0.6 tokens (empirically 1-2 chars per token); other chars
    // (digits, punctuation) count 0.3.
    let mut latin_words = 0usize;
    let mut cjk = 0usize;
    let mut other = 0usize;
    let mut in_latin_word = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            in_latin_word = false;
        } else if crate::lang::is_han(ch) || crate::lang::is_hangul(ch) {
            cjk += 1;
            in_latin_word = false;
        } else if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            if !in_latin_word {
                latin_words += 1;
                in_latin_word = true;
            }
        } else {
            other += 1;
            in_latin_word = false;
        }
    }
    ((latin_words as f32) * 1.3 + (cjk as f32) * 0.6 + (other as f32) * 0.3).ceil() as usize
}

pub fn chunk_document_hybrid(
    content: &str,
    doc_id: &str,
    lines_per_chunk: usize,
    overlap: usize,
    target_tokens: usize,
    max_tokens: usize,
) -> Vec<SectionChunk> {
    let base = chunk_document_lines(content, doc_id, lines_per_chunk, overlap);
    let mut out = Vec::new();
    let safe_target = target_tokens.clamp(100, max_tokens.max(100));
    let safe_max = max_tokens.max(safe_target);

    for chunk in base {
        if estimate_tokens(&chunk.content) <= safe_max {
            let chunk_id = stable_chunk_id(
                doc_id,
                &chunk.heading,
                &chunk.content,
                chunk.start_line,
                chunk.end_line,
            );
            out.push(SectionChunk {
                chunk_id,
                heading: chunk.heading,
                content: chunk.content,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                timestamp: None,
                key_entities: Vec::new(),
                important_terms: Vec::new(),
            });
            continue;
        }

        let lines: Vec<&str> = chunk.content.lines().collect();
        let mut start = 0usize;
        while start < lines.len() {
            let mut end = start + 1;
            let mut best_end = end;
            while end <= lines.len() {
                let candidate = lines[start..end].join("\n");
                let toks = estimate_tokens(&candidate);
                if toks <= safe_target {
                    best_end = end;
                    end += 1;
                    continue;
                }
                if toks <= safe_max && best_end == start + 1 {
                    best_end = end;
                }
                break;
            }
            if best_end <= start {
                best_end = (start + 1).min(lines.len());
            }
            let body = lines[start..best_end].join("\n").trim().to_string();
            if !body.is_empty() {
                let heading = format!(
                    "{} (part {}-{})",
                    chunk.heading,
                    chunk.start_line.saturating_add(start),
                    chunk.start_line.saturating_add(best_end).saturating_sub(1)
                );
                let start_line = chunk.start_line.saturating_add(start);
                let end_line = chunk.start_line.saturating_add(best_end).saturating_sub(1);
                let chunk_id = stable_chunk_id(doc_id, &heading, &body, start_line, end_line);
                out.push(SectionChunk {
                    chunk_id,
                    heading,
                    content: body,
                    start_line,
                    end_line,
                    timestamp: None,
                    key_entities: Vec::new(),
                    important_terms: Vec::new(),
                });
            }
            if best_end == lines.len() {
                break;
            }
            start = best_end;
        }
    }

    if out.is_empty() {
        return chunk_document_lines(content, doc_id, lines_per_chunk, overlap);
    }
    out
}

pub fn enrich_section_chunks(
    mut chunks: Vec<SectionChunk>,
    key_entities: &[Tier1Entity],
    important_terms: &[RankedTerm],
) -> Vec<SectionChunk> {
    for chunk in &mut chunks {
        let chunk_l = chunk.content.to_lowercase();
        for ent in key_entities {
            let term = ent.text.trim();
            if term.len() < 2 {
                continue;
            }
            if chunk_l.contains(&term.to_lowercase()) {
                chunk.key_entities.push(term.to_string());
            }
        }
        for term in important_terms {
            let t = term.term.trim();
            if t.len() < 2 {
                continue;
            }
            if chunk_l.contains(&t.to_lowercase()) {
                chunk.important_terms.push(t.to_string());
            }
        }
        chunk.key_entities.sort();
        chunk.key_entities.dedup();
        chunk.important_terms.sort();
        chunk.important_terms.dedup();
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_estimate_unchanged() {
        // Historical behavior: words * 1.3.
        assert_eq!(estimate_tokens("hello world"), 3);
        assert_eq!(estimate_tokens("one two three four"), 6);
    }

    #[test]
    fn chinese_estimate_counts_characters() {
        // 8 Han chars -> ~5 tokens, not 2 (the old split_whitespace count).
        let est = estimate_tokens("我毕业于清华大学");
        assert!(
            (4..=8).contains(&est),
            "expected a realistic CJK estimate, got {est}"
        );
        // Mixed text: Latin words plus CJK chars.
        let mixed = estimate_tokens("hello 清华大学 world");
        assert!(
            mixed >= 5,
            "mixed text should count both scripts, got {mixed}"
        );
    }
}
