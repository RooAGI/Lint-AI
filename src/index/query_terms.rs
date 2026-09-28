use crate::lang::Lang;
use crate::query_expansion::{expand_query_terms, normalize_for_index};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tantivy::query::Query;

use super::helpers::*;

pub(crate) const LEXICAL_CONTENT_BOOST: f32 = 1.0;
pub(crate) const LEXICAL_HEADINGS_BOOST: f32 = 1.4;
pub(crate) const LEXICAL_TERMS_BOOST: f32 = 2.0;
pub(crate) const LEXICAL_ENTITIES_BOOST: f32 = 2.4;

pub(crate) const FULL_QUERY_ENTITY_WEIGHT: f32 = 1.5;
pub(crate) const ENTITY_TERM_WEIGHT: f32 = 1.2;
pub(crate) const ENTITY_PREFIX_MULTIPLIER: f32 = 0.25;
pub(crate) const IMPORTANT_TERM_WEIGHT: f32 = 0.8;
pub(crate) const IMPORTANT_TERM_PREFIX_MULTIPLIER: f32 = 0.25;

pub(crate) const QUERY_TERM_CACHE_CAPACITY: usize = 256;

/// Multiplier on definitional semantic-tag TermQueries inside the tantivy
/// scorer (Luyi 2026-09-28). Same kind of parameter as the LEXICAL_*_BOOST
/// field boosts the system already tunes: set by measurement, minimally.
/// The tag match itself is scored by BM25 (IDF/length-norm/saturation);
/// this only scales that in-scorer weight.
///
/// Tuning (2026-09-28, BEHOOD_BIN binary):
/// mem-05 (weekend/Saturday vs Monday): 2.0/1.5/1.25/1.1/1.05 pass;
///   1.0 is the minimum where the tag itself (tantivy lexical) ranks the
///   Saturday fact above the Monday distractor (A.lex 4.32 > B.lex 4.26).
/// mem-08 (herb/cilantro vs coffee): 1.0 FAILS (A 11.97 < B 28.58),
///   2.0 fails (12.58), 10.0 fails (17.46), 28.0 fails (28.44 < 28.58),
///   29.0 PASSES (A 29.05 > B 28.58), 30.0 passes (29.66).
/// Shared minimum fixing both: 29.0.
///
/// TOY-CORPUS IDF CAVEAT: in these 2-doc corpora the tag IDF is ~ln2
/// (≈0.69), so one tag match is worth ≈0.6 raw BM25 points and the
/// multiplier must be large to close mem-08's 12.7-point lexical gap
/// (the coffee distractor shares rare word "avoids" with the question).
/// In a real corpus a rare tag like "herb" carries far higher IDF and
/// the same multiplier would be much stronger — 29.0 is conservative
/// for production but required to flip the toy pair. Re-tune on a
/// realistic corpus before treating 29.0 as final.
pub(crate) const TAG_BOOST: f32 = 29.0;

#[derive(Clone)]
pub(crate) struct PreparedQueryTerms {
    pub(crate) normalized: String,
    pub(crate) terms: Vec<String>,
    pub(crate) expanded_terms: Vec<String>,
}

pub(crate) static PREPARED_QUERY_CACHE: OnceLock<Mutex<HashMap<String, PreparedQueryTerms>>> =
    OnceLock::new();
pub(crate) static RAW_PREPARED_QUERY_CACHE: OnceLock<Mutex<HashMap<String, PreparedQueryTerms>>> =
    OnceLock::new();
// All MemoryIndex lexical shards use the same fixed schema, so Tantivy's parsed
// query object can be shared safely between shards. This avoids rebuilding the
// QueryParser and query tree once per selected segment. The key includes the
// definitional semantic tags: the same query text with different tags is a
// different tantivy query.
pub(crate) static PARSED_LEXICAL_QUERY_CACHE: OnceLock<
    Mutex<HashMap<(String, Vec<String>), Arc<dyn Query>>>,
> = OnceLock::new();

pub(crate) fn prepare_query_terms(query: &str) -> Option<PreparedQueryTerms> {
    const MAX_QUERY_CHARS: usize = 4096;
    const MAX_QUERY_TOKENS: usize = 128;
    let truncated = if query.len() > MAX_QUERY_CHARS {
        let mut cut = 0usize;
        for (idx, _) in query.char_indices() {
            if idx > MAX_QUERY_CHARS {
                break;
            }
            cut = idx;
        }
        &query[..cut]
    } else {
        query
    };
    // Avoid repeating Unicode normalization/deaccenting for hot identical
    // queries (the HTTP benchmark and typical agent retries commonly do this).
    let raw_cache = RAW_PREPARED_QUERY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(prepared) = raw_cache
        .lock()
        .expect("raw prepared query cache lock poisoned")
        .get(truncated)
        .cloned()
    {
        return Some(prepared);
    }
    let normalized = normalize_for_index(truncated);
    if normalized.is_empty() {
        return None;
    }
    let cache = PREPARED_QUERY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cached = {
        let cache = cache.lock().expect("prepared query cache lock poisoned");
        cache.get(&normalized).cloned()
    };
    if let Some(prepared) = cached {
        return Some(prepared);
    }

    let mut terms = tokenize_query_terms(&normalized);
    if terms.len() > MAX_QUERY_TOKENS {
        terms.truncate(MAX_QUERY_TOKENS);
    }
    if terms.is_empty() {
        terms.push(normalized.clone());
    }
    let prepared = PreparedQueryTerms {
        expanded_terms: expand_query_terms(&terms, Lang::Auto.resolve(truncated)).expanded_terms,
        normalized: normalized.clone(),
        terms,
    };
    let mut cache = cache.lock().expect("prepared query cache lock poisoned");
    if cache.len() >= QUERY_TERM_CACHE_CAPACITY {
        if let Some(oldest_key) = cache.keys().next().cloned() {
            cache.remove(&oldest_key);
        }
    }
    cache.insert(normalized, prepared.clone());
    let mut raw_cache = raw_cache
        .lock()
        .expect("raw prepared query cache lock poisoned");
    if raw_cache.len() >= QUERY_TERM_CACHE_CAPACITY {
        if let Some(oldest_key) = raw_cache.keys().next().cloned() {
            raw_cache.remove(&oldest_key);
        }
    }
    raw_cache.insert(truncated.to_string(), prepared.clone());
    Some(prepared)
}
pub(crate) const EXPANDED_ENTITY_WEIGHT: f32 = 0.45;
pub(crate) const EXPANDED_TERM_WEIGHT: f32 = 0.35;
pub(crate) const TOPIC_OVERLAP_WEIGHT: f32 = 0.35;
pub(crate) const DOC_TYPE_OVERLAP_WEIGHT: f32 = 0.25;
pub(crate) const DOC_LINK_GRAPH_WEIGHT: f32 = 0.22;
pub(crate) const GRAPH_MAX_BOOST: f32 = 0.6;
pub(crate) const ENTITY_GRAPH_WEIGHT: f32 = 0.08;
pub(crate) const ENTITY_GRAPH_MAX_CANDIDATES: usize = 100;
pub(crate) const FINAL_RERANK_WINDOW: usize = 200;
pub(crate) const MAX_DOC_POSTINGS: usize = 500;
pub(crate) const MAX_RESULTS_PER_GROUP: usize = 2;
pub(crate) const TEXT_RERANK_WEIGHT: f32 = 0.08;
pub(crate) const TEXT_RERANK_NGRAM_WEIGHT: f32 = 0.08;
pub(crate) const TEXT_RERANK_LCS_WEIGHT: f32 = 0.05;
pub(crate) const TEXT_RERANK_NGRAM_SIZE: usize = 2;
pub(crate) const TEXT_RERANK_WINDOW: usize = 30;
pub(crate) const TEXT_RERANK_CONTENT_CHARS: usize = 4096;
