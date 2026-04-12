use crate::cli::{
    ChunkStrategy, GraphExportFormat, GraphLevel, LlmChunkStrategy, Tier1NerProvider,
    Tier1TermRankerKind,
};
use crate::config::{load_config, normalize_list, Config};
use crate::filters::is_noise_concept;
use crate::graph::{normalize_concept, Graph, Tier0Record};
use crate::index::{DocRecord, MemoryIndex, Provenance, SectionChunk};
use crate::report::Report;
use crate::rules::cross_refs::check_cross_refs;
use crate::rules::orphan_pages::check_orphans;
use crate::tier1::{
    CValueStyleTermRanker, HeuristicKeyEntityRanker, ImportantTermRanker, KeyEntityRanker,
    RakeStyleTermRanker, SpacyKeyEntityRanker, TextRankStyleTermRanker, Tier1DocEntities,
    Tier1DocInput, Tier1DocTerms, YakeStyleTermRanker,
};
use aho_corasick::AhoCorasick;
use anyhow::Result;
use comrak::{
    nodes::{AstNode, NodeValue},
    parse_document, Arena, ComrakOptions,
};
use deunicode::deunicode;
use inflector::Inflector;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use unicode_normalization::UnicodeNormalization;
use regex::Regex;
use walkdir::WalkDir;

fn surface_forms(raw: &str) -> Vec<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return vec![];
    }
    let mut forms: HashSet<String> = HashSet::new();
    let lowered = deunicode(&raw.nfc().collect::<String>().to_lowercase()).to_lowercase();
    forms.insert(lowered.clone());
    let spaced = lowered.replace('_', " ").replace('-', " ");
    forms.insert(spaced.clone());
    forms.insert(lowered.replace('_', "").replace('-', ""));
    forms.insert(spaced.to_plural());
    forms.insert(spaced.to_singular());
    forms.into_iter().filter(|s| !s.is_empty()).collect()
}

fn build_matcher(
    graph: &Graph,
    cfg: &Config,
) -> (Option<AhoCorasick>, Vec<String>, HashMap<String, String>) {
    let mut concept_raw: HashMap<String, String> = HashMap::new();
    for page in &graph.pages {
        if is_noise_concept(&page.concept, cfg) {
            continue;
        }
        if let Some(prefix) = cfg.scope_prefix.as_ref() {
            let rel = page.rel_path.to_lowercase();
            if !rel.starts_with(&prefix.to_lowercase()) {
                continue;
            }
        }
        concept_raw
            .entry(page.concept.clone())
            .or_insert_with(|| page.raw_concept.clone());
    }

    let mut forms = Vec::new();
    let mut form_to_concept: HashMap<String, String> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();

    for (concept, raw) in &concept_raw {
        for form in surface_forms(raw) {
            if let Some(existing) = form_to_concept.get(&form) {
                if existing != concept {
                    ambiguous.insert(form.clone());
                }
                continue;
            }
            form_to_concept.insert(form.clone(), concept.clone());
            forms.push(form);
        }
    }

    forms.retain(|form| !ambiguous.contains(form));
    if forms.is_empty() {
        return (None, forms, form_to_concept);
    }
    let ac = AhoCorasick::new(&forms).ok();
    (ac, forms, form_to_concept)
}

fn heading_text<'a>(node: &'a AstNode<'a>) -> String {
    let mut text = String::new();
    for child in node.children() {
        if let NodeValue::Text(ref t) = child.data.borrow().value {
            text.push_str(t);
        }
    }
    text.trim().to_string()
}

fn collect_section_concepts(
    content: &str,
    ac: &AhoCorasick,
    forms: &[String],
    form_to_concept: &HashMap<String, String>,
    cfg: &Config,
) -> Vec<(String, HashSet<String>)> {
    let arena = Arena::new();
    let ast = parse_document(&arena, content, &ComrakOptions::default());
    let mut sections: Vec<(String, HashSet<String>)> = Vec::new();
    let mut section_index: HashMap<String, usize> = HashMap::new();
    let mut current = "(unscoped)".to_string();

    let idx = sections.len();
    sections.push((current.clone(), HashSet::new()));
    section_index.insert(current.clone(), idx);

    fn walk<'a>(
        node: &'a AstNode<'a>,
        in_code: bool,
        current: &mut String,
        sections: &mut Vec<(String, HashSet<String>)>,
        section_index: &mut HashMap<String, usize>,
        ac: &AhoCorasick,
        forms: &[String],
        form_to_concept: &HashMap<String, String>,
        cfg: &Config,
    ) {
        let value = &node.data.borrow().value;
        let now_in_code = in_code || matches!(value, NodeValue::Code(_) | NodeValue::CodeBlock(_));

        if let NodeValue::Heading(_) = value {
            let text = heading_text(node);
            if !text.is_empty() {
                *current = text;
            } else {
                *current = "(untitled section)".to_string();
            }
            if !section_index.contains_key(current) {
                let idx = sections.len();
                sections.push((current.clone(), HashSet::new()));
                section_index.insert(current.clone(), idx);
            }
            return;
        }

        if !now_in_code {
            if let NodeValue::Text(ref t) = value {
                let content_lower = normalize_concept(t);
                let mut found: HashSet<String> = HashSet::new();
                for mat in ac.find_iter(&content_lower) {
                    let start = mat.start();
                    let end = mat.end();
                    let left_ok = if start == 0 {
                        true
                    } else {
                        content_lower[..start]
                            .chars()
                            .next_back()
                            .map(|c| !c.is_ascii_alphanumeric())
                            .unwrap_or(true)
                    };
                    let right_ok = if end >= content_lower.len() {
                        true
                    } else {
                        content_lower[end..]
                            .chars()
                            .next()
                            .map(|c| !c.is_ascii_alphanumeric())
                            .unwrap_or(true)
                    };
                    if !(left_ok && right_ok) {
                        continue;
                    }
                    let form = &forms[mat.pattern()];
                    if let Some(concept) = form_to_concept.get(form) {
                        if !is_noise_concept(concept, cfg) {
                            found.insert(concept.clone());
                        }
                    }
                }
                if !found.is_empty() {
                    let idx = *section_index
                        .get(current)
                        .unwrap_or_else(|| section_index.get("(unscoped)").unwrap());
                    let entry = &mut sections[idx].1;
                    for concept in found {
                        entry.insert(concept);
                    }
                }
            }
        }

        for child in node.children() {
            walk(
                child,
                now_in_code,
                current,
                sections,
                section_index,
                ac,
                forms,
                form_to_concept,
                cfg,
            );
        }
    }

    walk(
        ast,
        false,
        &mut current,
        &mut sections,
        &mut section_index,
        ac,
        forms,
        form_to_concept,
        cfg,
    );

    sections
}

fn debug_phrase_matches(
    content: &str,
    ac: &AhoCorasick,
    forms: &[String],
    form_to_concept: &HashMap<String, String>,
    cfg: &Config,
) -> Vec<(String, String, usize, usize)> {
    let arena = Arena::new();
    let ast = parse_document(&arena, content, &ComrakOptions::default());
    let mut out = Vec::new();

    fn walk<'a>(
        node: &'a AstNode<'a>,
        in_code: bool,
        out: &mut Vec<(String, String, usize, usize)>,
        ac: &AhoCorasick,
        forms: &[String],
        form_to_concept: &HashMap<String, String>,
        cfg: &Config,
    ) {
        let value = &node.data.borrow().value;
        let now_in_code = in_code || matches!(value, NodeValue::Code(_) | NodeValue::CodeBlock(_));

        if !now_in_code {
            if let NodeValue::Text(ref t) = value {
                let normalized = normalize_concept(t);
                for mat in ac.find_iter(&normalized) {
                    let start = mat.start();
                    let end = mat.end();
                    let left_ok = if start == 0 {
                        true
                    } else {
                        normalized[..start]
                            .chars()
                            .next_back()
                            .map(|c| !c.is_ascii_alphanumeric())
                            .unwrap_or(true)
                    };
                    let right_ok = if end >= normalized.len() {
                        true
                    } else {
                        normalized[end..]
                            .chars()
                            .next()
                            .map(|c| !c.is_ascii_alphanumeric())
                            .unwrap_or(true)
                    };
                    if !(left_ok && right_ok) {
                        continue;
                    }
                    let form = &forms[mat.pattern()];
                    if let Some(concept) = form_to_concept.get(form) {
                        if is_noise_concept(concept, cfg) {
                            continue;
                        }
                        out.push((t.to_string(), concept.clone(), start, end));
                    }
                }
            }
        }

        for child in node.children() {
            walk(child, now_in_code, out, ac, forms, form_to_concept, cfg);
        }
    }

    walk(ast, false, &mut out, ac, forms, form_to_concept, cfg);
    out
}

/// Normalize a section heading to a stable bucket name.
pub fn normalize_heading(name: &str) -> String {
    let lower = name.trim().to_lowercase();
    if lower.is_empty() || lower == "(unscoped)" || lower == "(untitled section)" {
        return "unscoped".to_string();
    }
    if lower.contains("related") {
        return "related".to_string();
    }
    if lower.contains("troubleshoot") {
        return "troubleshooting".to_string();
    }
    if lower.contains("setup") || lower.contains("quickstart") || lower.contains("quick start") {
        return "setup".to_string();
    }
    if lower.contains("config") {
        return "configuration".to_string();
    }
    if lower.contains("overview") || lower.contains("what it is") || lower.contains("history") {
        return "overview".to_string();
    }
    if lower.contains("security") || lower.contains("access control") || lower.contains("auth") {
        return "security".to_string();
    }
    if lower.contains("routing") || lower.contains("session") {
        return "routing".to_string();
    }
    lower
}

fn show_concepts_by_section(graph: &Graph, cfg: &Config) {
    let (ac, forms, form_to_concept) = build_matcher(graph, cfg);
    let ac = match ac {
        Some(ac) => ac,
        None => return,
    };
    let ignore_sections = normalize_list(&cfg.ignore_sections);
    let mut aggregated: BTreeMap<String, HashMap<String, usize>> = BTreeMap::new();

    for page in &graph.pages {
        let sections = collect_section_concepts(&page.content, &ac, &forms, &form_to_concept, cfg);
        for (heading, concepts) in sections {
            if concepts.is_empty() {
                continue;
            }
            let key = normalize_heading(&heading);
            if ignore_sections.contains(&key) {
                continue;
            }
            let entry = aggregated.entry(key).or_default();
            for concept in concepts {
                *entry.entry(concept).or_insert(0) += 1;
            }
        }
    }

    for (section, counts) in aggregated {
        let mut list: Vec<(String, usize)> = counts.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        println!("Section: {}", section);
        for (concept, count) in list {
            println!("- {} ({})", concept, count);
        }
    }
}

fn common_dir_prefix(paths: &[String]) -> Option<String> {
    if paths.is_empty() {
        return None;
    }
    let parts: Vec<Vec<&str>> = paths.iter().map(|p| p.split('/').collect()).collect();
    let mut prefix: Vec<&str> = Vec::new();
    'outer: for idx in 0..parts[0].len().saturating_sub(1) {
        let candidate = parts[0][idx];
        for path_parts in &parts {
            if path_parts.get(idx).copied() != Some(candidate) {
                break 'outer;
            }
        }
        prefix.push(candidate);
    }
    if prefix.is_empty() {
        None
    } else {
        Some(format!("{}/", prefix.join("/")))
    }
}

/// Generate the `--analyze` output for tests and snapshotting.
pub fn analyze_for_tests(graph: &Graph, cfg: &Config) -> String {
    let (ac, forms, form_to_concept) = build_matcher(graph, cfg);
    let ac = match ac {
        Some(ac) => ac,
        None => return "Suggested config:\n{\n  \"stopwords\": [],\n  \"ignore_sections\": [\"unscoped\", \"related\"],\n  \"ignore_crossref_sections\": [\"unscoped\", \"related\"],\n  \"ignore_paths\": [],\n  \"allowlist_concepts\": []\n}\n\nStats:\npages: 0\n".to_string(),
    };
    let mut concept_pages: HashMap<String, usize> = HashMap::new();
    let mut section_counts: HashMap<String, usize> = HashMap::new();
    let mut page_count = 0usize;

    for page in &graph.pages {
        page_count += 1;
        let sections = collect_section_concepts(&page.content, &ac, &forms, &form_to_concept, cfg);
        let mut page_concepts: HashSet<String> = HashSet::new();
        for (heading, concepts) in sections {
            if !heading.trim().is_empty() {
                let key = normalize_heading(&heading);
                *section_counts.entry(key).or_insert(0) += 1;
            }
            for concept in concepts {
                page_concepts.insert(concept);
            }
        }
        for concept in page_concepts {
            *concept_pages.entry(concept).or_insert(0) += 1;
        }
    }

    let mut concept_list: Vec<(String, usize)> = concept_pages.into_iter().collect();
    concept_list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut suggested_stopwords = Vec::new();
    for (concept, count) in &concept_list {
        if page_count > 0 && (*count as f64 / page_count as f64) >= 0.4 {
            suggested_stopwords.push(concept.clone());
        }
    }

    let mut section_list: Vec<(String, usize)> = section_counts.into_iter().collect();
    section_list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut suggested_ignore_sections = Vec::new();
    for (section, count) in &section_list {
        if page_count > 0 && (*count as f64 / page_count as f64) >= 0.3 {
            if section == "related" || section == "unscoped" {
                suggested_ignore_sections.push(section.clone());
            }
        }
    }

    let rel_paths: Vec<String> = graph.pages.iter().map(|p| p.rel_path.clone()).collect();
    let scope_prefix = common_dir_prefix(&rel_paths);

    let mut out = String::new();
    out.push_str("Suggested config:\n");
    out.push_str("{\n");
    out.push_str(&format!("  \"stopwords\": {:?},\n", suggested_stopwords));
    out.push_str(&format!(
        "  \"ignore_sections\": {:?},\n",
        suggested_ignore_sections
    ));
    out.push_str(&format!(
        "  \"ignore_crossref_sections\": {:?},\n",
        suggested_ignore_sections
    ));
    out.push_str("  \"ignore_paths\": [],\n");
    if let Some(prefix) = scope_prefix {
        out.push_str("  \"allowlist_concepts\": [],\n");
        out.push_str(&format!("  \"scope_prefix\": \"{}\"\n", prefix));
    } else {
        out.push_str("  \"allowlist_concepts\": []\n");
    }
    out.push_str("}\n\n");
    out.push_str("Stats:\n");
    out.push_str(&format!("pages: {}\n", page_count));
    out.push_str("top concepts:\n");
    for (concept, count) in concept_list.iter().take(15) {
        out.push_str(&format!("- {} ({})\n", concept, count));
    }
    out.push_str("top sections:\n");
    for (section, count) in section_list.iter().take(10) {
        out.push_str(&format!("- {} ({})\n", section, count));
    }
    out
}

fn analyze_corpus(graph: &Graph, cfg: &Config) {
    let out = analyze_for_tests(graph, cfg);
    print!("{}", out);
}

fn show_tier1_entities(
    graph: &Graph,
    provider: &Tier1NerProvider,
    spacy_model: &str,
) -> Result<()> {
    let docs: Vec<Tier1DocInput> = graph
        .pages
        .iter()
        .map(|p| Tier1DocInput {
            id: p.rel_path.clone(),
            source: p.rel_path.clone(),
            content: p.content.clone(),
            concept: p.raw_concept.clone(),
            headings: p.headings.clone(),
        })
        .collect();
    let heuristic = HeuristicKeyEntityRanker;
    let mut heuristic_by_doc = heuristic.rank_docs(&docs)?;
    let mut by_doc = match provider {
        Tier1NerProvider::Heuristic => heuristic_by_doc.clone(),
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: spacy_model.to_string(),
                script_path: "scripts/spacy_ner.py".to_string(),
            };
            match spacy.rank_docs(&docs) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic_by_doc.clone()
                }
            }
        }
    };

    let mut docs_out = Vec::new();
    for doc in docs {
        let key_entities = by_doc
            .remove(&doc.id)
            .or_else(|| heuristic_by_doc.remove(&doc.id))
            .unwrap_or_default();
        docs_out.push(Tier1DocEntities {
            id: doc.id,
            source: doc.source,
            key_entities,
        });
    }
    println!("{}", serde_json::to_string_pretty(&docs_out)?);
    Ok(())
}

fn show_tier1_terms(graph: &Graph, ranker_kind: &Tier1TermRankerKind) -> Result<()> {
    let docs: Vec<Tier1DocInput> = graph
        .pages
        .iter()
        .map(|p| Tier1DocInput {
            id: p.rel_path.clone(),
            source: p.rel_path.clone(),
            content: p.content.clone(),
            concept: p.raw_concept.clone(),
            headings: p.headings.clone(),
        })
        .collect();

    let ranker: Box<dyn ImportantTermRanker> = match ranker_kind {
        Tier1TermRankerKind::Yake => Box::new(YakeStyleTermRanker),
        Tier1TermRankerKind::Rake => Box::new(RakeStyleTermRanker),
        Tier1TermRankerKind::Cvalue => Box::new(CValueStyleTermRanker),
        Tier1TermRankerKind::Textrank => Box::new(TextRankStyleTermRanker),
    };

    let mut out = Vec::new();
    for doc in docs {
        let important_terms = ranker.rank_terms(&doc);
        out.push(Tier1DocTerms {
            id: doc.id,
            source: doc.source,
            important_terms,
        });
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn select_term_ranker(ranker_kind: &Tier1TermRankerKind) -> Box<dyn ImportantTermRanker> {
    match ranker_kind {
        Tier1TermRankerKind::Yake => Box::new(YakeStyleTermRanker),
        Tier1TermRankerKind::Rake => Box::new(RakeStyleTermRanker),
        Tier1TermRankerKind::Cvalue => Box::new(CValueStyleTermRanker),
        Tier1TermRankerKind::Textrank => Box::new(TextRankStyleTermRanker),
    }
}

fn chunk_document_sections(content: &str, doc_id: &str) -> Vec<SectionChunk> {
    let heading_re = Regex::new(r"(?m)^#{1,6}\s+(.*)$").expect("valid heading regex");
    let mut chunks = Vec::new();
    let mut last_start = 0usize;
    let mut current_heading = "(document)".to_string();
    let mut idx = 0usize;

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
                chunks.push(SectionChunk {
                    chunk_id: format!("{}::{}", doc_id, idx),
                    heading: current_heading.clone(),
                    content: body.to_string(),
                    start_line,
                    end_line,
                    key_entities: Vec::new(),
                    important_terms: Vec::new(),
                });
                idx += 1;
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
        chunks.push(SectionChunk {
            chunk_id: format!("{}::{}", doc_id, idx),
            heading: current_heading,
            content: tail.to_string(),
            start_line,
            end_line,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }

    if chunks.is_empty() {
        chunks.push(SectionChunk {
            chunk_id: format!("{}::0", doc_id),
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line: 1,
            end_line: content.lines().count().max(1),
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }
    chunks
}

fn chunk_document_lines(
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
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        }];
    }
    let size = lines_per_chunk.clamp(1, 500);
    let ov = overlap.min(size.saturating_sub(1));
    let step = (size - ov).max(1);

    let mut chunks = Vec::new();
    let mut idx = 0usize;
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + size).min(lines.len());
        let body = lines[start..end].join("\n").trim().to_string();
        if !body.is_empty() {
            chunks.push(SectionChunk {
                chunk_id: format!("{}::{}", doc_id, idx),
                heading: format!("lines {}-{}", start + 1, end),
                content: body,
                start_line: start + 1,
                end_line: end,
                key_entities: Vec::new(),
                important_terms: Vec::new(),
            });
            idx += 1;
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
    if chunks.is_empty() {
        chunks.push(SectionChunk {
            chunk_id: format!("{}::0", doc_id),
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line: 1,
            end_line: lines.len().max(1),
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        });
    }
    chunks
}

fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    ((words as f32) * 1.3).ceil() as usize
}

fn chunk_document_hybrid(
    content: &str,
    doc_id: &str,
    lines_per_chunk: usize,
    overlap: usize,
    target_tokens: usize,
    max_tokens: usize,
) -> Vec<SectionChunk> {
    let base = chunk_document_lines(content, doc_id, lines_per_chunk, overlap);
    let mut out = Vec::new();
    let mut idx = 0usize;
    let safe_target = target_tokens.clamp(100, max_tokens.max(100));
    let safe_max = max_tokens.max(safe_target);

    for chunk in base {
        if estimate_tokens(&chunk.content) <= safe_max {
            out.push(SectionChunk {
                chunk_id: format!("{}::{}", doc_id, idx),
                heading: chunk.heading,
                content: chunk.content,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                key_entities: Vec::new(),
                important_terms: Vec::new(),
            });
            idx += 1;
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
                out.push(SectionChunk {
                    chunk_id: format!("{}::{}", doc_id, idx),
                    heading: format!("{} (part {})", chunk.heading, idx),
                    content: body,
                    start_line: chunk.start_line.saturating_add(start),
                    end_line: chunk.start_line.saturating_add(best_end).saturating_sub(1),
                    key_entities: Vec::new(),
                    important_terms: Vec::new(),
                });
                idx += 1;
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

fn enrich_section_chunks(
    mut chunks: Vec<SectionChunk>,
    key_entities: &[crate::tier1::Tier1Entity],
    important_terms: &[crate::tier1::RankedTerm],
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

fn build_memory_index(
    graph: &Graph,
    provider: &Tier1NerProvider,
    spacy_model: &str,
    ranker_kind: &Tier1TermRankerKind,
    chunk_strategy: &ChunkStrategy,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
    lexical_dir: Option<&Path>,
) -> Result<MemoryIndex> {
    let docs: Vec<Tier1DocInput> = graph
        .pages
        .iter()
        .map(|p| Tier1DocInput {
            id: p.rel_path.clone(),
            source: p.rel_path.clone(),
            content: p.content.clone(),
            concept: p.raw_concept.clone(),
            headings: p.headings.clone(),
        })
        .collect();

    let heuristic = HeuristicKeyEntityRanker;
    let entities_by_doc = match provider {
        Tier1NerProvider::Heuristic => heuristic.rank_docs(&docs)?,
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: spacy_model.to_string(),
                script_path: "scripts/spacy_ner.py".to_string(),
            };
            match spacy.rank_docs(&docs) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic.rank_docs(&docs).unwrap_or_default()
                }
            }
        }
    };

    let term_ranker = select_term_ranker(ranker_kind);
    let term_ranker_name = term_ranker.name().to_string();
    let ner_provider_name = match provider {
        Tier1NerProvider::Heuristic => "heuristic".to_string(),
        Tier1NerProvider::Spacy => format!("spacy:{}", spacy_model),
    };
    let tier0_by_source: HashMap<String, &Tier0Record> = graph
        .tier0_records
        .iter()
        .map(|r| (r.source.clone(), r))
        .collect();
    let concept_to_rel: HashMap<String, String> = graph
        .pages
        .iter()
        .map(|p| (p.concept.clone(), p.rel_path.clone()))
        .collect();

    let mut records = Vec::new();
    for doc in docs {
        let key_entities = entities_by_doc.get(&doc.id).cloned().unwrap_or_default();
        let important_terms = term_ranker.rank_terms(&doc);
        let t0 = tier0_by_source.get(&doc.source).copied();
        let probable_topic = if let Some(first_heading) = doc.headings.first() {
            Some(first_heading.clone())
        } else {
            important_terms.first().map(|t| t.term.clone())
        };
        let joined_headings = doc.headings.join(" ").to_lowercase();
        let content_l = doc.content.to_lowercase();
        let doc_type_guess = if joined_headings.contains("incident")
            || content_l.contains("postmortem")
        {
            Some("incident".to_string())
        } else if joined_headings.contains("runbook") || content_l.contains("playbook") {
            Some("runbook".to_string())
        } else if joined_headings.contains("changelog") || content_l.contains("release notes") {
            Some("changelog".to_string())
        } else if joined_headings.contains("reference") {
            Some("reference".to_string())
        } else if joined_headings.contains("tutorial") || joined_headings.contains("quick start") {
            Some("tutorial".to_string())
        } else if joined_headings.contains("decision") || content_l.contains("adr") {
            Some("decision".to_string())
        } else {
            None
        };
        let base_chunks = match chunk_strategy {
            ChunkStrategy::Heading => chunk_document_sections(&doc.content, &doc.id),
            ChunkStrategy::Line => {
                chunk_document_lines(&doc.content, &doc.id, chunk_lines, chunk_overlap)
            }
            ChunkStrategy::Hybrid => chunk_document_hybrid(
                &doc.content,
                &doc.id,
                chunk_lines,
                chunk_overlap,
                chunk_target_tokens,
                chunk_max_tokens,
            ),
        };
        let section_chunks = enrich_section_chunks(base_chunks, &key_entities, &important_terms);
        let doc_links = graph
            .pages
            .iter()
            .find(|p| p.rel_path == doc.id)
            .map(|p| {
                p.links
                    .iter()
                    .filter_map(|c| concept_to_rel.get(c).cloned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        records.push(DocRecord {
            doc_id: doc.id.clone(),
            source: doc.source.clone(),
            content: doc.content.clone(),
            timestamp: t0.and_then(|r| r.timestamp.clone()),
            doc_length: t0.map(|r| r.doc_length).unwrap_or(doc.content.len()),
            author_agent: t0.and_then(|r| r.author_agent.clone()),
            probable_topic,
            doc_type_guess,
            headings: doc.headings.clone(),
            doc_links,
            key_entities,
            important_terms,
            section_chunks,
            embedding: None,
            top_claims: Vec::new(),
            provenance: Provenance {
                source: doc.source.clone(),
                timestamp: t0.and_then(|r| r.timestamp.clone()),
                ner_provider: ner_provider_name.clone(),
                term_ranker: term_ranker_name.clone(),
                index_version: "v1-memory-hybrid".to_string(),
            },
        });
    }
    Ok(MemoryIndex::from_records_with_lexical_dir(records, lexical_dir))
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedQueryIndex {
    version: String,
    path: String,
    ner_provider: String,
    term_ranker: String,
    spacy_model: String,
    chunk_strategy: String,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
    corpus_fingerprint: String,
    records: Vec<DocRecord>,
}

const QUERY_CACHE_VERSION: &str = "v1-query-cache";
const DEFAULT_QUERY_TOP_K: usize = 5;
const LLM_CONTEXT_CANDIDATE_TOP_K: usize = 20;
const LLM_CONTEXT_DUPLICATE_DOC_PENALTY: f32 = 0.35;
const MAX_RESULT_COUNT: usize = 50;

#[derive(Debug, Clone)]
struct CacheSettings<'a> {
    root_path: &'a str,
    ner_provider: &'a Tier1NerProvider,
    term_ranker: &'a Tier1TermRankerKind,
    spacy_model: &'a str,
    chunk_strategy: &'a ChunkStrategy,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
}

fn query_cache_key(s: &CacheSettings<'_>) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.root_path.hash(&mut hasher);
    format!("{:?}", s.ner_provider).to_lowercase().hash(&mut hasher);
    format!("{:?}", s.term_ranker).to_lowercase().hash(&mut hasher);
    s.spacy_model.hash(&mut hasher);
    format!("{:?}", s.chunk_strategy).to_lowercase().hash(&mut hasher);
    s.chunk_lines.hash(&mut hasher);
    s.chunk_overlap.hash(&mut hasher);
    s.chunk_target_tokens.hash(&mut hasher);
    s.chunk_max_tokens.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn query_cache_file(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.json", query_cache_key(s)))
}

fn query_cache_lexical_dir(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.tantivy", query_cache_key(s)))
}

fn query_cache_core_file(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.core.bin", query_cache_key(s)))
}

fn fingerprint_base_path(root_path: &str) -> PathBuf {
    let root = Path::new(root_path);
    if root.is_file() {
        return root.parent().unwrap_or(root).to_path_buf();
    }
    let docs = root.join("docs");
    if docs.is_dir() {
        docs
    } else {
        root.to_path_buf()
    }
}

fn compute_corpus_fingerprint(
    root_path: &str,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
) -> String {
    let root = Path::new(root_path);
    let base = fingerprint_base_path(root_path);
    let single_file = if root.is_file() {
        Some(root.to_path_buf())
    } else {
        None
    };
    let rel_root = if root.is_file() {
        base.clone()
    } else {
        root.to_path_buf()
    };

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    root_path.hash(&mut hasher);
    let mut files_seen = 0usize;
    let mut total_bytes = 0usize;
    let mut items: Vec<(String, u64, u64, u64)> = Vec::new();

    for entry in WalkDir::new(base).max_depth(max_depth) {
        let Ok(entry) = entry else {
            continue;
        };
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(ref only) = single_file {
            if entry.path() != only {
                continue;
            }
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if ext != "md" {
            continue;
        }
        files_seen += 1;
        if files_seen > max_files {
            break;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        total_bytes = total_bytes.saturating_add(metadata.len() as usize);
        if total_bytes > max_total_bytes {
            break;
        }
        let rel = entry
            .path()
            .strip_prefix(&rel_root)
            .unwrap_or(entry.path())
            .display()
            .to_string();
        let mut mtime_secs = 0u64;
        if let Ok(modified) = metadata.modified() {
            if let Ok(dur) = modified.duration_since(UNIX_EPOCH) {
                mtime_secs = dur.as_secs();
            }
        }
        let content_hash = match fs::read(entry.path()) {
            Ok(bytes) => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                bytes.hash(&mut h);
                h.finish()
            }
            Err(_) => 0,
        };
        items.push((rel, metadata.len(), mtime_secs, content_hash));
    }

    items.sort_by(|a, b| a.0.cmp(&b.0));
    for (rel, len, mtime_secs, content_hash) in items {
        rel.hash(&mut hasher);
        len.hash(&mut hasher);
        mtime_secs.hash(&mut hasher);
        content_hash.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn load_cached_query_index(s: &CacheSettings<'_>, corpus_fingerprint: &str) -> Option<MemoryIndex> {
    let cache_file = query_cache_file(s);
    let data = fs::read_to_string(&cache_file).ok()?;
    let cached: CachedQueryIndex = serde_json::from_str(&data).ok()?;
    if cached.version != QUERY_CACHE_VERSION
        || cached.path != s.root_path
        || cached.ner_provider != format!("{:?}", s.ner_provider).to_lowercase()
        || cached.term_ranker != format!("{:?}", s.term_ranker).to_lowercase()
        || cached.spacy_model != s.spacy_model
        || cached.chunk_strategy != format!("{:?}", s.chunk_strategy).to_lowercase()
        || cached.chunk_lines != s.chunk_lines
        || cached.chunk_overlap != s.chunk_overlap
        || cached.chunk_target_tokens != s.chunk_target_tokens
        || cached.chunk_max_tokens != s.chunk_max_tokens
        || cached.corpus_fingerprint != corpus_fingerprint
    {
        return None;
    }
    let lexical_dir = query_cache_lexical_dir(s);
    let core_file = query_cache_core_file(s);
    match MemoryIndex::load_with_binary_core(cached.records.clone(), &core_file, Some(&lexical_dir)) {
        Ok(index) => return Some(index),
        Err(err) => {
            eprintln!("warning: binary core cache load failed (falling back): {}", err);
        }
    }
    Some(MemoryIndex::from_records_with_lexical_dir(
        cached.records,
        Some(&lexical_dir),
    ))
}

fn save_cached_query_index(
    s: &CacheSettings<'_>,
    corpus_fingerprint: &str,
    index: &MemoryIndex,
) -> Result<()> {
    let cache_file = query_cache_file(s);
    if let Some(parent) = cache_file.parent() {
        fs::create_dir_all(parent)?;
    }
    let records = index.docs.values().cloned().collect::<Vec<_>>();
    let payload = CachedQueryIndex {
        version: QUERY_CACHE_VERSION.to_string(),
        path: s.root_path.to_string(),
        ner_provider: format!("{:?}", s.ner_provider).to_lowercase(),
        term_ranker: format!("{:?}", s.term_ranker).to_lowercase(),
        spacy_model: s.spacy_model.to_string(),
        chunk_strategy: format!("{:?}", s.chunk_strategy).to_lowercase(),
        chunk_lines: s.chunk_lines,
        chunk_overlap: s.chunk_overlap,
        chunk_target_tokens: s.chunk_target_tokens,
        chunk_max_tokens: s.chunk_max_tokens,
        corpus_fingerprint: corpus_fingerprint.to_string(),
        records,
    };
    fs::write(cache_file, serde_json::to_string(&payload)?)?;
    let core_file = query_cache_core_file(s);
    index.save_binary_core(&core_file)?;
    Ok(())
}

#[derive(Serialize)]
struct Tier0IndexFile {
    tier: String,
    generated_at_unix: u64,
    path: String,
    document_count: usize,
    documents_by_id: BTreeMap<String, Tier0Record>,
}

#[derive(Serialize)]
struct QueryOutput {
    query: String,
    elapsed_ms: u128,
    result_count: usize,
    results: Vec<crate::index::SearchResult>,
}

#[derive(Clone, Serialize)]
struct LlmContextChunk {
    doc_id: String,
    source: String,
    chunk_id: String,
    heading: String,
    start_line: usize,
    end_line: usize,
    matched_entities: Vec<String>,
    matched_terms: Vec<String>,
    score: f32,
    score_breakdown: ChunkScoreBreakdown,
    text: String,
}

#[derive(Clone, Serialize, Default)]
struct ChunkScoreBreakdown {
    entity_overlap: f32,
    term_overlap: f32,
    heading_overlap: f32,
    chunk_link_score: f32,
    entity_graph_score: f32,
}

#[derive(Serialize)]
struct LlmContextOutput {
    mode: String,
    query: String,
    generated_at_unix: u64,
    corpus_path: String,
    elapsed_ms: u128,
    chunk_result_count: usize,
    top_chunks: Vec<LlmContextChunk>,
    prompt_policy: String,
}

#[derive(Clone, Copy)]
struct ChunkRankingWeights {
    entity_overlap: f32,
    term_overlap: f32,
    heading_overlap: f32,
}

impl Default for ChunkRankingWeights {
    fn default() -> Self {
        Self {
            entity_overlap: 2.0,
            term_overlap: 1.0,
            heading_overlap: 0.4,
        }
    }
}

fn truncate_for_llm(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>()
}

fn score_chunk_for_query(
    chunk: &SectionChunk,
    matched_entities_l: &[String],
    matched_terms_l: &[String],
    weights: ChunkRankingWeights,
    link_bonus: f32,
    entity_graph_bonus: f32,
) -> (f32, ChunkScoreBreakdown) {
    let hay = format!("{}\n{}", chunk.heading, chunk.content).to_lowercase();
    let heading_l = chunk.heading.to_lowercase();
    let entity_hits = matched_entities_l
        .iter()
        .filter(|e| !e.is_empty() && hay.contains(e.as_str()))
        .count() as f32;
    let term_hits = matched_terms_l
        .iter()
        .filter(|t| !t.is_empty() && hay.contains(t.as_str()))
        .count() as f32;
    let heading_hits = matched_terms_l
        .iter()
        .chain(matched_entities_l.iter())
        .filter(|x| !x.is_empty() && heading_l.contains(x.as_str()))
        .count() as f32;
    let breakdown = ChunkScoreBreakdown {
        entity_overlap: entity_hits * weights.entity_overlap,
        term_overlap: term_hits * weights.term_overlap,
        heading_overlap: heading_hits * weights.heading_overlap,
        chunk_link_score: link_bonus,
        entity_graph_score: entity_graph_bonus,
    };
    let total = breakdown.entity_overlap + breakdown.term_overlap + breakdown.heading_overlap;
    (
        total + breakdown.chunk_link_score + breakdown.entity_graph_score,
        breakdown,
    )
}

fn build_llm_context_output(
    index: &MemoryIndex,
    query: &str,
    corpus_path: &str,
    elapsed_ms: u128,
    chunk_candidates: &[crate::index::SearchResult],
    strategy: &LlmChunkStrategy,
    result_count: usize,
) -> LlmContextOutput {
    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut top_chunks = Vec::new();
    let weights = ChunkRankingWeights::default();
    let mut chunk_candidates_scored: Vec<(String, f32, LlmContextChunk)> = Vec::new();
    let anchor_doc_ids: HashSet<String> = chunk_candidates
        .iter()
        .take(5)
        .map(|r| r.doc_id.clone())
        .collect();
    let mut anchor_entities: HashSet<String> = HashSet::new();
    for r in chunk_candidates.iter().take(5) {
        if let Some(doc) = index.docs.get(&r.doc_id) {
            for e in &doc.key_entities {
                let k = normalize_concept(&e.text);
                if !k.is_empty() {
                    anchor_entities.insert(k);
                }
            }
        }
    }

    for result in chunk_candidates {
        let Some(doc) = index.docs.get(&result.doc_id) else {
            continue;
        };
        let matched_terms_l = result
            .matched_terms
            .iter()
            .map(|t| t.to_lowercase())
            .collect::<Vec<_>>();
        let matched_entities_l = result
            .matched_entities
            .iter()
            .map(|e| e.to_lowercase())
            .collect::<Vec<_>>();

        let mut chunk_scores = Vec::new();
        for chunk in &doc.section_chunks {
            let link_hits = doc
                .doc_links
                .iter()
                .filter(|d| anchor_doc_ids.contains(d.as_str()))
                .count();
            let link_bonus = (0.15 * link_hits as f32).min(0.6);
            let entity_overlap = chunk
                .key_entities
                .iter()
                .map(|e| normalize_concept(e))
                .filter(|e| !e.is_empty() && anchor_entities.contains(e))
                .count();
            let entity_graph_bonus = (0.10 * entity_overlap as f32).min(0.6);
            let (score, breakdown) = score_chunk_for_query(
                chunk,
                &matched_entities_l,
                &matched_terms_l,
                weights,
                link_bonus,
                entity_graph_bonus,
            );
            chunk_scores.push((score, breakdown, chunk));
        }
        chunk_scores.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (score, breakdown, chunk) in chunk_scores {
            if score <= 0.0 {
                continue;
            }
            let combined_score = score + 0.25 * result.score.max(0.0);
            chunk_candidates_scored.push((
                result.doc_id.clone(),
                combined_score,
                LlmContextChunk {
                doc_id: doc.doc_id.clone(),
                source: doc.source.clone(),
                chunk_id: chunk.chunk_id.clone(),
                heading: chunk.heading.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                matched_entities: result.matched_entities.clone(),
                matched_terms: result.matched_terms.clone(),
                score: combined_score,
                score_breakdown: breakdown,
                text: truncate_for_llm(&chunk.content, 1200),
                },
            ));
        }
    }

    match strategy {
        LlmChunkStrategy::All => {
            chunk_candidates_scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            top_chunks.extend(
                chunk_candidates_scored
                    .into_iter()
                    .take(result_count)
                    .map(|(_, _, chunk)| chunk),
            );
        }
        LlmChunkStrategy::ByDoc => {
            let mut best_by_doc: HashMap<String, (f32, LlmContextChunk)> = HashMap::new();
            let mut leftovers: Vec<(String, f32, LlmContextChunk)> = Vec::new();
            for (doc_id, combined, chunk) in chunk_candidates_scored {
                match best_by_doc.get(&doc_id) {
                    Some((best, _)) if *best >= combined => leftovers.push((doc_id, combined, chunk)),
                    Some((_, prev)) => {
                        leftovers.push((doc_id.clone(), *best_by_doc.get(&doc_id).map(|x| &x.0).unwrap_or(&combined), prev.clone()));
                        best_by_doc.insert(doc_id, (combined, chunk));
                    }
                    None => {
                        best_by_doc.insert(doc_id, (combined, chunk));
                    }
                }
            }
            let mut selected_docs: HashSet<String> = HashSet::new();
            for result in chunk_candidates {
                if top_chunks.len() >= result_count {
                    break;
                }
                if let Some((_, chunk)) = best_by_doc.remove(&result.doc_id) {
                    selected_docs.insert(result.doc_id.clone());
                    top_chunks.push(chunk);
                }
            }
            leftovers.sort_by(|a, b| {
                let a_s = if selected_docs.contains(&a.0) {
                    a.1 - LLM_CONTEXT_DUPLICATE_DOC_PENALTY
                } else {
                    a.1
                };
                let b_s = if selected_docs.contains(&b.0) {
                    b.1 - LLM_CONTEXT_DUPLICATE_DOC_PENALTY
                } else {
                    b.1
                };
                b_s.partial_cmp(&a_s).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (doc_id, _, chunk) in leftovers {
                if top_chunks.len() >= result_count {
                    break;
                }
                selected_docs.insert(doc_id);
                top_chunks.push(chunk);
            }
        }
    }
    top_chunks.truncate(result_count);

    LlmContextOutput {
        mode: "llm_context".to_string(),
        query: query.to_string(),
        generated_at_unix,
        corpus_path: corpus_path.to_string(),
        elapsed_ms,
        chunk_result_count: top_chunks.len(),
        top_chunks,
        prompt_policy:
            "Use only provided evidence. Every claim must cite source+chunk_id+line range; no citation means unknown."
                .to_string(),
    }
}

fn write_tier0_index(path: &str, records: &[Tier0Record], scanned_path: &str) -> Result<String> {
    let mut output_path = Path::new(path).to_path_buf();
    if output_path.extension().is_none() {
        output_path.set_extension("json");
    }
    let mut documents_by_id: BTreeMap<String, Tier0Record> = BTreeMap::new();
    for record in records {
        documents_by_id.insert(record.id.clone(), record.clone());
    }
    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let payload = Tier0IndexFile {
        tier: "tier0".to_string(),
        generated_at_unix,
        path: scanned_path.to_string(),
        document_count: documents_by_id.len(),
        documents_by_id,
    };
    let json = serde_json::to_string_pretty(&payload)?;
    safe_write_text_file(&output_path.display().to_string(), &json)?;
    Ok(output_path.display().to_string())
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\"', "\\\"")
}

fn js_string_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        // Prevent HTML parser from terminating inline <script> blocks.
        .replace("</", "<\\/")
}

fn safe_write_text_file(out_path: &str, content: &str) -> Result<()> {
    let path = Path::new(out_path);
    if path.is_dir() {
        anyhow::bail!("refusing to write: output path is a directory");
    }
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to write: output path is a symlink");
        }
    }
    if let Some(parent) = path.parent() {
        // Reject symlink components in the existing parent chain.
        let mut cur = if parent.is_absolute() {
            PathBuf::from("/")
        } else {
            std::env::current_dir()?
        };
        for comp in parent.components() {
            use std::path::Component;
            match comp {
                Component::RootDir | Component::CurDir => continue,
                Component::ParentDir => anyhow::bail!("refusing to write: parent traversal is not allowed"),
                Component::Normal(seg) => {
                    cur.push(seg);
                    if let Ok(meta) = fs::symlink_metadata(&cur) {
                        if meta.file_type().is_symlink() {
                            anyhow::bail!(
                                "refusing to write: parent path component is a symlink ({})",
                                cur.display()
                            );
                        }
                    }
                }
                Component::Prefix(_) => {}
            }
        }
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, content)?;
    Ok(())
}

fn export_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=box, style=rounded];".to_string());

    let mut concept_to_rel: HashMap<String, String> = HashMap::new();
    for page in &graph.pages {
        concept_to_rel.insert(page.concept.clone(), page.rel_path.clone());
    }

    for page in &graph.pages {
        let node_id = format!("n_{}", dot_escape(&page.concept.replace(' ', "_")));
        let label = format!("{}\\n({})", dot_escape(&page.rel_path), dot_escape(&page.concept));
        lines.push(format!("  \"{}\" [label=\"{}\"];", node_id, label));
    }

    for page in &graph.pages {
        let from_id = format!("n_{}", dot_escape(&page.concept.replace(' ', "_")));
        for link in &page.links {
            if concept_to_rel.contains_key(link) {
                let to_id = format!("n_{}", dot_escape(&link.replace(' ', "_")));
                lines.push(format!("  \"{}\" -> \"{}\";", from_id, to_id));
            }
        }
    }

    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

fn export_chunk_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_chunk_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=box, style=rounded];".to_string());
    for chunk in &graph.chunks {
        let node_id = dot_escape(&chunk.chunk_id);
        let label = format!(
            "{}\\n{}:{}-{}",
            dot_escape(&chunk.doc_rel_path),
            dot_escape(&chunk.heading),
            chunk.start_line,
            chunk.end_line
        );
        lines.push(format!("  \"{}\" [label=\"{}\"];", node_id, label));
    }
    for edge in graph.chunk_graph.raw_edges() {
        let from = &graph.chunk_graph[edge.source()];
        let to = &graph.chunk_graph[edge.target()];
        lines.push(format!("  \"{}\" -> \"{}\";", dot_escape(from), dot_escape(to)));
    }
    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

fn export_entity_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_entity_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=ellipse, style=filled, fillcolor=\"#e8f1ff\"];".to_string());
    for ent in graph.entity_index.keys() {
        let id = dot_escape(ent);
        lines.push(format!("  \"{}\" [label=\"{}\"];", id, id));
    }
    for edge in graph.entity_graph.raw_edges() {
        let from = &graph.entity_graph[edge.source()];
        let to = &graph.entity_graph[edge.target()];
        let label = match edge.weight {
            crate::graph::EntityEdgeKind::CoOccurs => "co_occurs",
            crate::graph::EntityEdgeKind::DocLink => "doc_link",
        };
        lines.push(format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];",
            dot_escape(from),
            dot_escape(to),
            label
        ));
    }
    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct GraphJsonNode {
    id: String,
    concept: String,
    source: String,
}

#[derive(Serialize)]
struct GraphJsonEdge {
    source: String,
    target: String,
}

#[derive(Serialize)]
struct GraphJsonExport {
    nodes: Vec<GraphJsonNode>,
    edges: Vec<GraphJsonEdge>,
}

fn build_graph_json_export(graph: &Graph) -> GraphJsonExport {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut concept_exists: HashSet<String> = HashSet::new();

    for page in &graph.pages {
        concept_exists.insert(page.concept.clone());
        nodes.push(GraphJsonNode {
            id: page.concept.clone(),
            concept: page.concept.clone(),
            source: page.rel_path.clone(),
        });
    }

    for page in &graph.pages {
        for link in &page.links {
            if concept_exists.contains(link) {
                edges.push(GraphJsonEdge {
                    source: page.concept.clone(),
                    target: link.clone(),
                });
            }
        }
    }

    GraphJsonExport { nodes, edges }
}

fn export_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct ChunkGraphJsonNode {
    id: String,
    chunk_id: String,
    doc_id: String,
    heading: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Serialize)]
struct ChunkGraphJsonEdge {
    source: String,
    target: String,
    kind: String,
}

#[derive(Serialize)]
struct ChunkGraphJsonExport {
    nodes: Vec<ChunkGraphJsonNode>,
    edges: Vec<ChunkGraphJsonEdge>,
}

fn build_chunk_graph_json_export(graph: &Graph) -> ChunkGraphJsonExport {
    let nodes = graph
        .chunks
        .iter()
        .map(|c| ChunkGraphJsonNode {
            id: c.chunk_id.clone(),
            chunk_id: c.chunk_id.clone(),
            doc_id: c.doc_rel_path.clone(),
            heading: c.heading.clone(),
            start_line: c.start_line,
            end_line: c.end_line,
        })
        .collect::<Vec<_>>();
    let edges = graph
        .chunk_graph
        .raw_edges()
        .iter()
        .map(|e| {
            let kind = match e.weight {
                crate::graph::ChunkEdgeKind::Next => "next",
                crate::graph::ChunkEdgeKind::DocLink => "doc_link",
            };
            ChunkGraphJsonEdge {
                source: graph.chunk_graph[e.source()].clone(),
                target: graph.chunk_graph[e.target()].clone(),
                kind: kind.to_string(),
            }
        })
        .collect::<Vec<_>>();
    ChunkGraphJsonExport { nodes, edges }
}

fn export_chunk_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_chunk_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct EntityGraphJsonNode {
    id: String,
    label: String,
}

#[derive(Serialize)]
struct EntityGraphJsonEdge {
    source: String,
    target: String,
    kind: String,
}

#[derive(Serialize)]
struct EntityGraphJsonExport {
    nodes: Vec<EntityGraphJsonNode>,
    edges: Vec<EntityGraphJsonEdge>,
}

fn build_entity_graph_json_export(graph: &Graph) -> EntityGraphJsonExport {
    let nodes = graph
        .entity_index
        .keys()
        .map(|e| EntityGraphJsonNode {
            id: e.clone(),
            label: e.clone(),
        })
        .collect::<Vec<_>>();
    let edges = graph
        .entity_graph
        .raw_edges()
        .iter()
        .map(|e| {
            let kind = match e.weight {
                crate::graph::EntityEdgeKind::CoOccurs => "co_occurs",
                crate::graph::EntityEdgeKind::DocLink => "doc_link",
            };
            EntityGraphJsonEdge {
                source: graph.entity_graph[e.source()].clone(),
                target: graph.entity_graph[e.target()].clone(),
                kind: kind.to_string(),
            }
        })
        .collect::<Vec<_>>();
    EntityGraphJsonExport { nodes, edges }
}

fn export_entity_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_entity_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

fn export_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\", source: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.concept),
                js_string_escape(&n.source)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let html = format!(
        r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Lint-AI Graph</title>
  <script src="./cytoscape.min.js"></script>
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; }}
    #cy {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="cy"></div>
  <script>
    const elements = {{
      nodes: [{nodes_js}],
      edges: [{edges_js}]
    }};
    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements,
      layout: {{ name: 'cose', animate: false }},
      style: [
        {{
          selector: 'node',
          style: {{
            'label': 'data(label)',
            'font-size': 10,
            'background-color': '#1f77b4',
            'color': '#111'
          }}
        }},
        {{
          selector: 'edge',
          style: {{
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'line-color': '#888',
            'target-arrow-color': '#888',
            'width': 1
          }}
        }}
      ]
    }});
  </script>
</body>
</html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

fn export_chunk_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_chunk_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\", source: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.heading),
                js_string_escape(&n.doc_id)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\", kind: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target),
                js_string_escape(&e.kind)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let html = format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8" /><title>Lint-AI Chunk Graph</title>
<script src="./cytoscape.min.js"></script>
<style>html, body {{ width: 100%; height: 100%; margin: 0; }} #cy {{ width:100%; height:100%; }}</style>
</head><body><div id="cy"></div><script>
const elements = {{ nodes:[{nodes_js}], edges:[{edges_js}] }};
cytoscape({{
  container: document.getElementById('cy'),
  elements,
  layout: {{ name: 'cose', animate: false }},
  style: [
    {{ selector: 'node', style: {{ 'label': 'data(label)', 'font-size': 9, 'background-color': '#2a9d8f' }} }},
    {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#777', 'target-arrow-color': '#777', 'width': 1 }} }}
  ]
}});
</script></body></html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

fn export_entity_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_entity_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.label)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\", kind: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target),
                js_string_escape(&e.kind)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let html = format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8" /><title>Lint-AI Entity Graph</title>
<script src="./cytoscape.min.js"></script>
<style>html, body {{ width:100%; height:100%; margin:0; }} #cy {{ width:100%; height:100%; }}</style>
</head><body><div id="cy"></div><script>
const elements = {{ nodes:[{nodes_js}], edges:[{edges_js}] }};
cytoscape({{
  container: document.getElementById('cy'),
  elements,
  layout: {{ name: 'cose', animate: false }},
  style: [
    {{ selector: 'node', style: {{ 'label': 'data(label)', 'font-size': 10, 'background-color': '#457b9d' }} }},
    {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#666', 'target-arrow-color': '#666', 'width': 1 }} }}
  ]
}});
</script></body></html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct ChunkGraphStats {
    chunk_nodes: usize,
    chunk_edges: usize,
    next_edges: usize,
    doc_link_edges: usize,
    docs_with_chunks: usize,
    avg_chunks_per_doc: f32,
}

fn chunk_graph_stats(graph: &Graph) -> ChunkGraphStats {
    let mut next_edges = 0usize;
    let mut doc_link_edges = 0usize;
    for e in graph.chunk_graph.raw_edges() {
        match e.weight {
            crate::graph::ChunkEdgeKind::Next => next_edges += 1,
            crate::graph::ChunkEdgeKind::DocLink => doc_link_edges += 1,
        }
    }
    let docs_with_chunks = graph.doc_to_chunks.len();
    let avg_chunks_per_doc = if docs_with_chunks == 0 {
        0.0
    } else {
        graph.chunks.len() as f32 / docs_with_chunks as f32
    };
    ChunkGraphStats {
        chunk_nodes: graph.chunks.len(),
        chunk_edges: graph.chunk_graph.edge_count(),
        next_edges,
        doc_link_edges,
        docs_with_chunks,
        avg_chunks_per_doc,
    }
}

#[derive(Serialize)]
struct OntologyNode {
    id: String,
    node_type: String,
    label: String,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct OntologyEdge {
    source: String,
    target: String,
    edge_type: String,
    weight: f32,
    confidence: f32,
    provenance: serde_json::Value,
}

#[derive(Serialize)]
struct OntologyExport {
    version: String,
    generated_at_unix: u64,
    corpus_path: String,
    node_count: usize,
    edge_count: usize,
    nodes: Vec<OntologyNode>,
    edges: Vec<OntologyEdge>,
}

fn canonical_entity_id(text: &str) -> String {
    let base = normalize_concept(text);
    let key = if base.trim().is_empty() {
        "unknown".to_string()
    } else {
        base.replace(' ', "_")
    };
    format!("entity:{}", key)
}

fn export_ontology_json(index: &MemoryIndex, out_path: &str, corpus_path: &str) -> Result<String> {
    let mut nodes: Vec<OntologyNode> = Vec::new();
    let mut edges: Vec<OntologyEdge> = Vec::new();
    let mut node_ids: HashSet<String> = HashSet::new();
    let mut entity_aliases: HashMap<String, HashSet<String>> = HashMap::new();
    let mut cooccur_counts: HashMap<(String, String), usize> = HashMap::new();

    for doc in index.docs.values() {
        let doc_node_id = format!("doc:{}", doc.doc_id);
        if node_ids.insert(doc_node_id.clone()) {
            nodes.push(OntologyNode {
                id: doc_node_id.clone(),
                node_type: "doc".to_string(),
                label: doc.source.clone(),
                metadata: serde_json::json!({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "timestamp": doc.timestamp,
                    "probable_topic": doc.probable_topic,
                    "doc_type_guess": doc.doc_type_guess,
                }),
            });
        }

        for chunk in &doc.section_chunks {
            let chunk_node_id = format!("chunk:{}", chunk.chunk_id);
            if node_ids.insert(chunk_node_id.clone()) {
                nodes.push(OntologyNode {
                    id: chunk_node_id.clone(),
                    node_type: "chunk".to_string(),
                    label: chunk.heading.clone(),
                    metadata: serde_json::json!({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": doc.doc_id,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }),
                });
            }
            edges.push(OntologyEdge {
                source: chunk_node_id.clone(),
                target: doc_node_id.clone(),
                edge_type: "belongs_to".to_string(),
                weight: 1.0,
                confidence: 1.0,
                provenance: serde_json::json!({"source":"chunk-structure"}),
            });

            let mut canonical_in_chunk: Vec<String> = Vec::new();
            for ent in &chunk.key_entities {
                let eid = canonical_entity_id(ent);
                entity_aliases
                    .entry(eid.clone())
                    .or_default()
                    .insert(ent.clone());
                canonical_in_chunk.push(eid.clone());
                if node_ids.insert(eid.clone()) {
                    nodes.push(OntologyNode {
                        id: eid.clone(),
                        node_type: "entity".to_string(),
                        label: ent.clone(),
                        metadata: serde_json::json!({}),
                    });
                }
                edges.push(OntologyEdge {
                    source: chunk_node_id.clone(),
                    target: eid,
                    edge_type: "mentions".to_string(),
                    weight: 1.0,
                    confidence: 0.8,
                    provenance: serde_json::json!({"source":"tier1_chunk_entities"}),
                });
            }
            canonical_in_chunk.sort();
            canonical_in_chunk.dedup();
            for i in 0..canonical_in_chunk.len() {
                for j in (i + 1)..canonical_in_chunk.len() {
                    let a = canonical_in_chunk[i].clone();
                    let b = canonical_in_chunk[j].clone();
                    let key = if a <= b { (a, b) } else { (b, a) };
                    *cooccur_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    for (eid, aliases) in entity_aliases {
        if let Some(node) = nodes.iter_mut().find(|n| n.id == eid) {
            let mut alias_list = aliases.into_iter().collect::<Vec<_>>();
            alias_list.sort();
            node.metadata = serde_json::json!({ "aliases": alias_list });
        }
    }

    for ((a, b), count) in cooccur_counts {
        edges.push(OntologyEdge {
            source: a,
            target: b,
            edge_type: "co_occurs".to_string(),
            weight: count as f32,
            confidence: 0.6,
            provenance: serde_json::json!({"source":"chunk_cooccurrence","count":count}),
        });
    }

    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let payload = OntologyExport {
        version: "v0.1-seed".to_string(),
        generated_at_unix,
        corpus_path: corpus_path.to_string(),
        node_count: nodes.len(),
        edge_count: edges.len(),
        nodes,
        edges,
    };
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

/// Run the lint pipeline using CLI arguments.
///
/// This is the main entry point used by the CLI wrapper in `main.rs`.
pub fn run(args: crate::cli::Args) -> Result<()> {
    let cfg = load_config(
        args.config.as_deref(),
        &args.path,
        args.strict_config,
        args.max_config_bytes,
    )
    .map_err(|err| anyhow::anyhow!(err))?;
    if args.query.is_some() || args.llm_context.is_some() {
        let cache_settings = CacheSettings {
            root_path: &args.path,
            ner_provider: &args.tier1_ner_provider,
            term_ranker: &args.tier1_term_ranker,
            spacy_model: &args.spacy_model,
            chunk_strategy: &args.chunk_strategy,
            chunk_lines: args.chunk_lines,
            chunk_overlap: args.chunk_overlap,
            chunk_target_tokens: args.chunk_target_tokens,
            chunk_max_tokens: args.chunk_max_tokens,
        };
        let corpus_fingerprint = compute_corpus_fingerprint(
            &args.path,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        );
        let lexical_dir = query_cache_lexical_dir(&cache_settings);
        let index = if let Some(cached) = load_cached_query_index(&cache_settings, &corpus_fingerprint) {
            cached
        } else {
            let mut graph = Graph::build(
                &args.path,
                args.max_bytes,
                args.max_files,
                args.max_depth,
                args.max_total_bytes,
            )?;
            if !cfg.ignore_paths.is_empty() {
                let ignore = normalize_list(&cfg.ignore_paths);
                graph.pages.retain(|p| {
                    let rel = p.rel_path.to_lowercase();
                    !ignore.iter().any(|pat| rel.contains(pat))
                });
                let retained: HashSet<String> =
                    graph.pages.iter().map(|p| p.rel_path.clone()).collect();
                graph.tier0_records.retain(|r| retained.contains(&r.source));
            }
            let built = build_memory_index(
                &graph,
                &args.tier1_ner_provider,
                &args.spacy_model,
                &args.tier1_term_ranker,
                &args.chunk_strategy,
                args.chunk_lines,
                args.chunk_overlap,
                args.chunk_target_tokens,
                args.chunk_max_tokens,
                Some(&lexical_dir),
            )?;
            if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &built) {
                eprintln!("warning: unable to persist query cache: {}", err);
            }
            built
        };
        let query_value = args
            .llm_context
            .as_deref()
            .or(args.query.as_deref());
        if let Some(query) = query_value {
            let started = Instant::now();
            if args.llm_context.is_some() {
                let requested = args.result_count.clamp(1, MAX_RESULT_COUNT);
                let candidate_top_k = requested.max(LLM_CONTEXT_CANDIDATE_TOP_K);
                let candidate_results = index.query(query, candidate_top_k);
                let elapsed_ms = started.elapsed().as_millis();
                let payload = build_llm_context_output(
                    &index,
                    query,
                    &args.path,
                    elapsed_ms,
                    &candidate_results,
                    &args.llm_chunk_strategy,
                    requested,
                );
                if args.simplified {
                    let top_text = payload
                        .top_chunks
                        .iter()
                        .map(|c| c.text.clone())
                        .collect::<Vec<_>>();
                    let mut value = serde_json::to_value(&payload)?;
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("top_chunks".to_string(), serde_json::json!(top_text));
                    }
                    println!("{}", serde_json::to_string_pretty(&value)?);
                } else {
                    println!("{}", serde_json::to_string_pretty(&payload)?);
                }
            } else {
                let results = index.query(query, DEFAULT_QUERY_TOP_K);
                let elapsed_ms = started.elapsed().as_millis();
                let payload = QueryOutput {
                    query: query.to_string(),
                    elapsed_ms,
                    result_count: results.len(),
                    results,
                };
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
        return Ok(());
    }

    let mut graph = Graph::build(
        &args.path,
        args.max_bytes,
        args.max_files,
        args.max_depth,
        args.max_total_bytes,
    )?;
    if !cfg.ignore_paths.is_empty() {
        let ignore = normalize_list(&cfg.ignore_paths);
        graph.pages.retain(|p| {
            let rel = p.rel_path.to_lowercase();
            !ignore.iter().any(|pat| rel.contains(pat))
        });
        let retained: HashSet<String> = graph.pages.iter().map(|p| p.rel_path.clone()).collect();
        graph.tier0_records.retain(|r| retained.contains(&r.source));
    }
    if args.show_tier0 {
        println!("{}", serde_json::to_string_pretty(&graph.tier0_records)?);
        return Ok(());
    }
    if args.show_tier1_entities {
        show_tier1_entities(&graph, &args.tier1_ner_provider, &args.spacy_model)?;
        return Ok(());
    }
    if args.show_tier1_terms {
        show_tier1_terms(&graph, &args.tier1_term_ranker)?;
        return Ok(());
    }
    if args.index || args.index_redacted {
        let cache_settings = CacheSettings {
            root_path: &args.path,
            ner_provider: &args.tier1_ner_provider,
            term_ranker: &args.tier1_term_ranker,
            spacy_model: &args.spacy_model,
            chunk_strategy: &args.chunk_strategy,
            chunk_lines: args.chunk_lines,
            chunk_overlap: args.chunk_overlap,
            chunk_target_tokens: args.chunk_target_tokens,
            chunk_max_tokens: args.chunk_max_tokens,
        };
        let corpus_fingerprint = compute_corpus_fingerprint(
            &args.path,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        );
        let lexical_dir = query_cache_lexical_dir(&cache_settings);
        let index = build_memory_index(
            &graph,
            &args.tier1_ner_provider,
            &args.spacy_model,
            &args.tier1_term_ranker,
            &args.chunk_strategy,
            args.chunk_lines,
            args.chunk_overlap,
            args.chunk_target_tokens,
            args.chunk_max_tokens,
            Some(&lexical_dir),
        )?;
        if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &index) {
            eprintln!("warning: unable to persist query cache: {}", err);
        }
        if args.index_redacted {
            println!("{}", serde_json::to_string_pretty(&index.redacted_for_export())?);
        } else {
            println!("{}", serde_json::to_string_pretty(&index)?);
        }
        return Ok(());
    }
    if let Some(out_path) = args.tier0_index_out.as_deref() {
        let written_path = write_tier0_index(out_path, &graph.tier0_records, &args.path)?;
        println!(
            "Wrote Tier 0 index ({} documents) to {}",
            graph.tier0_records.len(),
            written_path
        );
        return Ok(());
    }
    if args.show_concepts {
        show_concepts_by_section(&graph, &cfg);
        return Ok(());
    }
    if args.analyze {
        analyze_corpus(&graph, &cfg);
        return Ok(());
    }
    if args.debug_matches {
        let (ac, forms, form_to_concept) = build_matcher(&graph, &cfg);
        let ac = match ac {
            Some(ac) => ac,
            None => return Ok(()),
        };
        for page in &graph.pages {
            println!("{}", page.rel_path);
            let matches = debug_phrase_matches(&page.content, &ac, &forms, &form_to_concept, &cfg);
            for (line, concept, start, end) in matches {
                println!("match [{}..{}]: {} -> {}", start, end, concept, line.trim());
            }
        }
        return Ok(());
    }
    if args.show_headings {
        for page in &graph.pages {
            println!("{}", page.rel_path);
            for heading in &page.headings {
                println!("- {}", heading);
            }
        }
        return Ok(());
    }
    if args.show_chunk_graph_stats {
        println!("{}", serde_json::to_string_pretty(&chunk_graph_stats(&graph))?);
        return Ok(());
    }
    if let Some(format) = args.export_graph.as_ref() {
        let written = match (&args.graph_level, format) {
            (GraphLevel::Doc, GraphExportFormat::Dot) => export_graph_dot(&graph, &args.graph_out)?,
            (GraphLevel::Doc, GraphExportFormat::Json) => {
                export_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Doc, GraphExportFormat::CytoscapeHtml) => {
                export_graph_cytoscape_html(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::Dot) => {
                export_chunk_graph_dot(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::Json) => {
                export_chunk_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::CytoscapeHtml) => {
                export_chunk_graph_cytoscape_html(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::Dot) => {
                export_entity_graph_dot(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::Json) => {
                export_entity_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::CytoscapeHtml) => {
                export_entity_graph_cytoscape_html(&graph, &args.graph_out)?
            }
        };
        println!(
            "Wrote graph export ({:?}, level={:?}) with {} pages to {}",
            format,
            args.graph_level,
            graph.pages.len(),
            written
        );
        return Ok(());
    }
    if args.export_ontology {
        let index = build_memory_index(
            &graph,
            &args.tier1_ner_provider,
            &args.spacy_model,
            &args.tier1_term_ranker,
            &args.chunk_strategy,
            args.chunk_lines,
            args.chunk_overlap,
            args.chunk_target_tokens,
            args.chunk_max_tokens,
            None,
        )?;
        let written = export_ontology_json(&index, &args.ontology_out, &args.path)?;
        println!(
            "Wrote ontology export with {} docs to {}",
            index.docs.len(),
            written
        );
        return Ok(());
    }
    let mut report = Report::new();

    check_orphans(&graph, &mut report);
    check_cross_refs(&graph, &mut report, &cfg);

    report.print();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_heading_rules() {
        assert_eq!(normalize_heading("Related"), "related");
        assert_eq!(normalize_heading("Quick start"), "setup");
        assert_eq!(normalize_heading("Security notes"), "security");
        assert_eq!(normalize_heading("Routing & Sessions"), "routing");
        assert_eq!(normalize_heading(""), "unscoped");
    }

    #[test]
    fn common_prefix() {
        let paths = vec![
            "docs/channels/a.md".to_string(),
            "docs/channels/b.md".to_string(),
        ];
        assert_eq!(common_dir_prefix(&paths).as_deref(), Some("docs/channels/"));
        assert_eq!(common_dir_prefix(&[]), None);
    }
}
