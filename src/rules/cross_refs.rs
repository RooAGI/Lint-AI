use crate::config::{normalize_list, Config};
use crate::filters::is_noise_concept;
use crate::graph::{normalize_concept, Graph};
use crate::report::Report;
use aho_corasick::AhoCorasick;
use deunicode::deunicode;
use inflector::Inflector;
use petgraph::algo::kosaraju_scc;
use std::collections::{HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn has_word_boundaries(text: &str, start: usize, end: usize) -> bool {
    let left_ok = if start == 0 {
        true
    } else {
        text[..start]
            .chars()
            .next_back()
            .map(|c| !is_word_char(c))
            .unwrap_or(true)
    };

    let right_ok = if end >= text.len() {
        true
    } else {
        text[end..]
            .chars()
            .next()
            .map(|c| !is_word_char(c))
            .unwrap_or(true)
    };

    left_ok && right_ok
}

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

/// Check for missing cross-references based on concept mentions.
pub fn check_cross_refs(graph: &Graph, report: &mut Report, cfg: &Config) {
    let sccs = kosaraju_scc(&graph.graph);
    let mut component: HashMap<petgraph::graph::NodeIndex, usize> = HashMap::new();
    for (idx, nodes) in sccs.iter().enumerate() {
        for node in nodes {
            component.insert(*node, idx);
        }
    }
    let mut avg_rank = 0.0;
    if graph.graph.node_count() > 0 {
        avg_rank = graph.graph.edge_count() as f64 / graph.graph.node_count() as f64;
    }
    let allowlist = normalize_list(&cfg.allowlist_concepts);
    let ignore_sections = normalize_list(&cfg.ignore_crossref_sections);
    let scope_prefix = cfg.scope_prefix.as_ref().map(|s| s.to_lowercase());

    let mut concept_raw: HashMap<String, String> = HashMap::new();
    for page in &graph.pages {
        if is_noise_concept(&page.concept, cfg) {
            continue;
        }
        if let Some(prefix) = scope_prefix.as_ref() {
            let rel = page.rel_path.to_lowercase();
            if !rel.starts_with(prefix) {
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
        return;
    }
    let ac = AhoCorasick::new(&forms).unwrap();
    for page in &graph.pages {
        if let Some(prefix) = scope_prefix.as_ref() {
            let rel = page.rel_path.to_lowercase();
            if !rel.starts_with(prefix) {
                continue;
            }
        }

        let linked: HashSet<String> = page.links.clone();
        let page_concept = page.concept.clone();

        let arena = comrak::Arena::new();
        let ast = comrak::parse_document(&arena, &page.content, &comrak::ComrakOptions::default());

        let mut current_heading = String::from("unscoped");
        let mut found: HashSet<String> = HashSet::new();

        fn walk<'a>(
            node: &'a comrak::nodes::AstNode<'a>,
            in_code: bool,
            current_heading: &mut String,
            found: &mut HashSet<String>,
            ac: &AhoCorasick,
            forms: &[String],
            form_to_concept: &HashMap<String, String>,
            cfg: &Config,
            ignore_sections: &[String],
            allowlist: &[String],
        ) {
            let value = &node.data.borrow().value;
            let now_in_code = in_code
                || matches!(
                    value,
                    comrak::nodes::NodeValue::Code(_) | comrak::nodes::NodeValue::CodeBlock(_)
                );

            if let comrak::nodes::NodeValue::Heading(_) = value {
                let mut text = String::new();
                for child in node.children() {
                    if let comrak::nodes::NodeValue::Text(ref t) = child.data.borrow().value {
                        text.push_str(t);
                    }
                }
                let label = text.trim().to_lowercase();
                if !label.is_empty() {
                    *current_heading = label;
                } else {
                    *current_heading = "unscoped".to_string();
                }
                return;
            }

            if !now_in_code {
                if let comrak::nodes::NodeValue::Text(ref t) = value {
                    let heading_key = crate::engine::normalize_heading(current_heading);
                    if ignore_sections.contains(&heading_key) {
                        // Skip matches in ignored sections.
                    } else {
                        let normalized = normalize_concept(t);
                        for mat in ac.find_iter(&normalized) {
                            let start = mat.start();
                            let end = mat.end();
                            if !has_word_boundaries(&normalized, start, end) {
                                continue;
                            }
                            let form = &forms[mat.pattern()];
                            if let Some(concept) = form_to_concept.get(form) {
                                if is_noise_concept(concept, cfg) {
                                    continue;
                                }
                                if !allowlist.is_empty() && !allowlist.contains(concept) {
                                    continue;
                                }
                                found.insert(concept.clone());
                            }
                        }
                    }
                }
            }

            for child in node.children() {
                walk(
                    child,
                    now_in_code,
                    current_heading,
                    found,
                    ac,
                    forms,
                    form_to_concept,
                    cfg,
                    ignore_sections,
                    allowlist,
                );
            }
        }

        walk(
            ast,
            false,
            &mut current_heading,
            &mut found,
            &ac,
            &forms,
            &form_to_concept,
            cfg,
            &ignore_sections,
            &allowlist,
        );

        for concept in found {
            if concept == page_concept {
                continue;
            }
            if !linked.contains(&concept) {
                let mut severity = "low";
                if let Some(idx) = graph.index.get(&concept) {
                    let degree = graph.graph.edges(*idx).count() as f64;
                    if degree >= avg_rank {
                        severity = "high";
                    }
                    if let Some(src_idx) = graph.index.get(&page.concept) {
                        if component.get(src_idx) != component.get(idx) {
                            severity = "high";
                        }
                    }
                }
                report.add(format!(
                    "Missing cross-ref in {} -> [[{}]] ({})",
                    page.rel_path, concept, severity
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_forms_variants() {
        let forms = surface_forms("Group-Messages");
        let set: HashSet<String> = forms.into_iter().collect();
        assert!(set.contains("group messages"));
        assert!(set.contains("groupmessages"));
        assert!(set.contains("group messages"));
    }
}
