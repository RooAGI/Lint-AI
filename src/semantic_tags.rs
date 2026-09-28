//! Definitional semantic tags (Luyi 2026-09-28).
//!
//! Ruling: definitional knowledge (Saturday = weekend, cilantro = a herb)
//! is not a score bonus. It is a first-class match inside the retrieval
//! model. bekind judges each document at index time and each query at
//! query time; both sides emit the same closed-set tag vocabulary, and the
//! tags are indexed and searched as ordinary terms in the tantivy lexical
//! index — weighted by BM25 (IDF, length norm, saturation) like every
//! other term. There are no additive constants outside the scorer.
//!
//! Tags are purely additive (SHOULD clauses): a tag match can only raise a
//! document's score, never lower it, and a missing tag changes nothing.

use crate::behood_query::{analyze_scope_verdicts, ScopeVerdict};
use std::collections::HashMap;

/// Tag emitted when a scope verdict reports habitual/recurring content.
pub const HABITUAL_TAG: &str = "habitual";

/// Pure tag emission from one document's scope verdict. Unit-testable, no
/// I/O. Emits the canonical temporal words verbatim (bekind already emits
/// canonical lowercase from the closed 7-day set, so the query and index
/// sides share the exact token) plus [`HABITUAL_TAG`] when habitual.
pub fn doc_scope_tags(scope: &ScopeVerdict) -> Vec<String> {
    let mut tags: Vec<String> = scope.temporal_words.clone();
    if scope.habitual {
        tags.push(HABITUAL_TAG.to_string());
    }
    tags.sort();
    tags.dedup();
    tags
}

/// Batch tag computation over many document texts, in order. One batched
/// daemon round-trip for all inputs. Fail-open: any daemon failure yields
/// no tags and the index builds exactly as before.
pub fn batch_doc_scope_tags(contents: &[&str]) -> Vec<Vec<String>> {
    let verdicts = analyze_scope_verdicts(contents);
    let by_idx: HashMap<usize, &ScopeVerdict> = verdicts
        .iter()
        .filter_map(|v| {
            v.id.strip_prefix("s:")?
                .parse::<usize>()
                .ok()
                .map(|i| (i, v))
        })
        .collect();
    let empty = ScopeVerdict {
        id: String::new(),
        activity_phrase: String::new(),
        temporal_words: Vec::new(),
        habitual: false,
    };
    (0..contents.len())
        .map(|i| {
            let verdict = by_idx.get(&i).copied().unwrap_or(&empty);
            doc_scope_tags(verdict)
        })
        .collect()
}

/// Tags for the ORIGINAL user query: its canonical temporal words, plus
/// [`HABITUAL_TAG`] when the question itself is habitual. A non-habitual
/// question emits no habitual tag — no effect, never a penalty. This
/// preserves the old `!question.habitual || fact.habitual` semantics
/// additively: habitual questions match habitual facts through the shared
/// tag, everything else is untouched.
///
/// Fail-open: no daemon/binary or no temporal scope in the question yields
/// an empty vec, and the lexical query runs exactly as before.
pub fn query_scope_tags(query: &str) -> Vec<String> {
    let mut tags = Vec::new();
    if let Some(verdict) = analyze_scope_verdicts(&[query]).into_iter().next() {
        tags.extend(verdict.temporal_words.iter().cloned());
        if verdict.habitual {
            tags.push(HABITUAL_TAG.to_string());
        }
    }
    tags.sort();
    tags.dedup();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(temporal_words: &[&str], habitual: bool) -> ScopeVerdict {
        ScopeVerdict {
            id: "s:0".to_string(),
            activity_phrase: String::new(),
            temporal_words: temporal_words.iter().map(|s| s.to_string()).collect(),
            habitual,
        }
    }

    #[test]
    fn doc_tags_emit_temporal_words_and_habitual() {
        assert_eq!(
            doc_scope_tags(&verdict(&["weekend"], true)),
            vec!["habitual".to_string(), "weekend".to_string()]
        );
    }

    #[test]
    fn doc_tags_omit_habitual_when_not_habitual() {
        assert_eq!(
            doc_scope_tags(&verdict(&["weekday"], false)),
            vec!["weekday".to_string()]
        );
    }

    #[test]
    fn doc_tags_empty_verdict_yields_no_tags() {
        assert!(doc_scope_tags(&verdict(&[], false)).is_empty());
    }

    #[test]
    fn doc_tags_dedup_repeated_words() {
        assert_eq!(
            doc_scope_tags(&verdict(&["weekend", "weekend"], true)),
            vec!["habitual".to_string(), "weekend".to_string()]
        );
    }

    /// The `semantic_tags` index field uses tantivy's default TEXT analyzer;
    /// every tag token ("weekend", "weekday", "habitual", "herb") must survive
    /// it as a single lowercase token, otherwise the SHOULD TermQueries
    /// would silently match nothing.
    #[test]
    fn default_text_analyzer_preserves_tag_tokens() {
        use tantivy::collector::Count;
        use tantivy::query::TermQuery;
        use tantivy::schema::{IndexRecordOption, Schema, TEXT};
        use tantivy::{doc, Index, Term};

        let mut schema_builder = Schema::builder();
        let tags = schema_builder.add_text_field("semantic_tags", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer(15_000_000).expect("writer");
        writer
            .add_document(doc!(tags => "habitual weekend weekday herb"))
            .expect("index doc");
        writer.commit().expect("commit");
        let reader = index.reader().expect("reader");
        let searcher = reader.searcher();
        for token in ["habitual", "weekend", "weekday", "herb"] {
            let term = Term::from_field_text(tags, token);
            let query = TermQuery::new(term, IndexRecordOption::Basic);
            let count = searcher.search(&query, &Count).expect("search") as usize;
            assert_eq!(count, 1, "tag token '{token}' must survive the analyzer");
        }
    }
}
