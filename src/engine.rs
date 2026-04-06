use crate::config::{load_config, normalize_list, Config};
use crate::graph::{normalize_concept, Graph};
use crate::report::Report;
use crate::rules::orphan_pages::check_orphans;
use crate::rules::cross_refs::check_cross_refs;
use aho_corasick::AhoCorasick;
use anyhow::Result;
use comrak::{nodes::{AstNode, NodeValue}, parse_document, Arena, ComrakOptions};
use deunicode::deunicode;
use inflector::Inflector;
use std::collections::{BTreeMap, HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

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

fn is_stopword(concept: &str) -> bool {
    const STOP: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if", "in", "is",
        "it", "its", "of", "on", "or", "the", "to", "via", "with",
        "api", "app", "apps", "auth", "build", "config", "data", "doc", "docs", "feature",
        "features", "file", "files", "guide", "help", "id", "index", "info", "issue", "issues",
        "key", "keys", "log", "logs", "model", "models", "page", "pages", "role", "roles",
        "run", "runs", "service", "services", "setup", "status", "system", "test", "tests",
        "tool", "tools", "user", "users", "web", "cli", "sdk", "repo", "project",
    ];
    STOP.contains(&concept)
}

fn is_noise_concept(concept: &str, cfg: &Config) -> bool {
    if concept.len() < 3 {
        return true;
    }
    if concept.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if is_stopword(concept) {
        return true;
    }
    let extra = normalize_list(&cfg.stopwords);
    extra.contains(&concept.to_string())
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
        let now_in_code = in_code
            || matches!(value, NodeValue::Code(_) | NodeValue::CodeBlock(_));

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
        let now_in_code = in_code
            || matches!(value, NodeValue::Code(_) | NodeValue::CodeBlock(_));

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
    let parts: Vec<Vec<&str>> = paths
        .iter()
        .map(|p| p.split('/').collect())
        .collect();
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
    out.push_str(&format!("  \"ignore_sections\": {:?},\n", suggested_ignore_sections));
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

pub fn run(args: crate::cli::Args) -> Result<()> {
    let cfg = load_config(
        args.config.as_deref(),
        &args.path,
        args.strict_config,
        args.max_config_bytes,
    )
    .map_err(|err| anyhow::anyhow!(err))?;
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
