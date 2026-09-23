use deunicode::deunicode;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    pub original_terms: Vec<String>,
    pub expanded_terms: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Entry {
    term: String,
    related: Vec<Related>,
}

#[derive(Debug, Clone, Deserialize)]
struct Related {
    term: String,
    relation: String,
    confidence: f32,
}

#[derive(Debug, Default)]
struct LexicalStore {
    by_term: HashMap<String, Vec<Related>>,
}

static WORDNET_DATA: &[u8] = include_bytes!("../data/lexical/wordnet_subset.json.gz");
static CONCEPTNET_DATA: &[u8] = include_bytes!("../data/lexical/conceptnet_subset.json.gz");
static STORE: OnceLock<Option<LexicalStore>> = OnceLock::new();
static STEMMER: OnceLock<Stemmer> = OnceLock::new();
static NORMALIZE_RE: OnceLock<Regex> = OnceLock::new();
const MAX_EXPANSIONS_PER_TERM: usize = 3;
const CONCEPTNET_MIN_CONFIDENCE: f32 = 0.82;

pub fn expand_query_terms(input_terms: &[String]) -> ExpandedQuery {
    let original_terms = input_terms
        .iter()
        .map(|t| normalize_for_index(t))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>();

    let Some(store) = STORE.get_or_init(load_store).as_ref() else {
        return ExpandedQuery {
            original_terms,
            expanded_terms: Vec::new(),
        };
    };

    let original_set: HashSet<String> = original_terms.iter().cloned().collect();
    let mut expanded = Vec::new();

    for term in &original_terms {
        let mut count = 0usize;
        if let Some(related) = store.by_term.get(term) {
            for rel in related {
                if count >= MAX_EXPANSIONS_PER_TERM {
                    break;
                }
                let candidate = normalize_for_index(&rel.term);
                if candidate.is_empty()
                    || original_set.contains(&candidate)
                    || expanded.contains(&candidate)
                {
                    continue;
                }
                expanded.push(candidate);
                count += 1;
            }
        }
    }

    ExpandedQuery {
        original_terms,
        expanded_terms: expanded,
    }
}

fn load_store() -> Option<LexicalStore> {
    let mut store = LexicalStore::default();
    let wn: Vec<Entry> = serde_json::from_slice(&decompress_gzip(WORDNET_DATA)?).ok()?;
    let cn: Vec<Entry> = serde_json::from_slice(&decompress_gzip(CONCEPTNET_DATA)?).ok()?;

    for e in wn {
        let key = normalize_for_index(&e.term);
        if key.is_empty() {
            continue;
        }
        for rel in e.related {
            if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo") {
                continue;
            }
            add_related(&mut store, &key, rel);
        }
    }

    for e in cn {
        let key = normalize_for_index(&e.term);
        if key.is_empty() {
            continue;
        }
        for rel in e.related {
            if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo" | "RelatedTo") {
                continue;
            }
            if rel.relation == "RelatedTo" && rel.confidence < CONCEPTNET_MIN_CONFIDENCE {
                continue;
            }
            add_related(&mut store, &key, rel);
        }
    }

    for rels in store.by_term.values_mut() {
        rels.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Some(store)
}

/// The lexical files are stored gzip-compressed so the embedded vocabulary
/// does not blow the crates.io package size limit; decompress at load.
fn decompress_gzip(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut decoder = libflate::gzip::Decoder::new(data).ok()?;
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok()?;
    Some(out)
}

fn add_related(store: &mut LexicalStore, key: &str, rel: Related) {
    let related_key = normalize_for_index(&rel.term);
    store
        .by_term
        .entry(key.to_string())
        .or_default()
        .push(rel.clone());

    if related_key.is_empty() || related_key == key {
        return;
    }
    if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo") {
        return;
    }
    store.by_term.entry(related_key).or_default().push(Related {
        term: key.to_string(),
        relation: rel.relation,
        confidence: rel.confidence,
    });
}

pub fn normalize_for_index(input: &str) -> String {
    let lowered = deunicode(input).to_lowercase();
    let token_re =
        NORMALIZE_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9]{1,}").expect("valid regex"));
    let stemmer = STEMMER.get_or_init(|| Stemmer::create(Algorithm::English));
    token_re
        .find_iter(&lowered)
        .map(|m| stemmer.stem(m.as_str()).to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize_for_index("Virtual-Machine!"), "virtual machin");
        assert_eq!(normalize_for_index("Café"), "cafe");
        assert_eq!(normalize_for_index("running"), "run");
    }

    #[test]
    fn expansion_caps_and_dedups() {
        let terms = vec!["install".to_string(), "setup".to_string()];
        let out = expand_query_terms(&terms);
        assert!(out.expanded_terms.len() <= terms.len() * MAX_EXPANSIONS_PER_TERM);
        for t in &out.expanded_terms {
            assert!(!out.original_terms.contains(t));
        }
    }

    #[test]
    fn install_expands_from_generated_subset() {
        let out = expand_query_terms(&["install".to_string()]);
        assert!(!out.expanded_terms.is_empty());
        assert!(out.expanded_terms.iter().all(|t| !t.is_empty()));
    }

    #[test]
    fn symmetric_relations_expand_back_to_source_terms() {
        let out = expand_query_terms(&["occupation".to_string()]);
        assert!(
            out.expanded_terms.iter().any(|t| t == "job"),
            "expected job expansion, got {:?}",
            out.expanded_terms
        );
    }

    #[test]
    fn expanded_lexical_subset_covers_common_search_terms() {
        let out = expand_query_terms(&["job".to_string(), "bug".to_string()]);
        assert!(!out.expanded_terms.is_empty());
        for term in &out.expanded_terms {
            assert!(!out.original_terms.contains(term));
        }
    }

    #[test]
    fn certificate_expands_to_degree_via_conceptnet() {
        // Regression test for a LoCoMo miss: the question asked about a
        // "certificate" while the dialogue turn said "degree". The bridge is
        // associative (ConceptNet RelatedTo), not a WordNet synonym.
        let out = expand_query_terms(&["certificate".to_string()]);
        let degree = normalize_for_index("degree");
        assert!(
            out.expanded_terms.iter().any(|t| t == &degree),
            "expected degree expansion, got {:?}",
            out.expanded_terms
        );
    }
}
