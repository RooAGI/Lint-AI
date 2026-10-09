//! Definitional semantic tags (Luyi 2026-09-28; index-time-only since 2026-10-07).
//!
//! Ruling: definitional knowledge (Saturday = weekend, cilantro = a herb)
//! is not a score bonus. It is a first-class match inside the retrieval
//! model. bekind judges each document ONCE at index time; the canonical
//! closed-set tags (temporal words, "habitual", admitted kinds) are indexed
//! as ordinary terms in the tantivy `semantic_tags` field — the "synthetic
//! document". The query path does no per-query bekind mapping: a question
//! saying "weekend" matches a doc whose literal words say "Saturday"
//! through the doc's indexed canonical terms, scored by BM25 (IDF, length
//! norm, saturation) like every other term. There are no additive constants
//! outside the scorer, and no daemon round-trip on the query path.
//!
//! Tags are purely additive in the index: a doc without tags simply has an
//! empty synthetic field, and retrieval degrades to the pre-tag behavior.

use crate::behood_query::{
    analyze_kind_verdicts, analyze_scope_verdicts, KindVerdict, ScopeVerdict,
};
use std::collections::HashMap;

/// Tag emitted when a scope verdict reports habitual/recurring content.
pub const HABITUAL_TAG: &str = "habitual";

/// Admitted closed-set kind tags (Luyi 2026-09-28). Each new category needs
/// its own explicit admission here: bekind reporting a kind does NOT
/// automatically tag it.
pub const ADMITTED_KIND_TAGS: &[&str] = &["herb", "food", "restaurant", "dining", "eat", "cafe", "bar", "park", "gym", "exercise", "drink", "run"];

/// Venue lexicon (Luyi 2026-10-07): "where" questions point to venue kinds,/// Pure tag emission from one document's scope verdict. Unit-testable, no
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
        where_phrase: String::new(),
    };
    (0..contents.len())
        .map(|i| {
            let verdict = by_idx.get(&i).copied().unwrap_or(&empty);
            doc_scope_tags(verdict)
        })
        .collect()
}

/// Pure tag emission from one document's kind verdict: admitted kinds only.
/// Unit-testable, no I/O.
pub fn doc_kind_tags(verdict: &KindVerdict) -> Vec<String> {
    let mut tags: Vec<String> = verdict
        .kinds
        .iter()
        .map(|hit| hit.kind.to_lowercase())
        .filter(|kind| ADMITTED_KIND_TAGS.contains(&kind.as_str()))
        .collect();
    // Closed-set definitional kinds (Luyi 2026-10-09): beKIND's phrase
    // judgments like "gardening" for "planting", "doctor" for
    // "dermatologist". These bypass ADMITTED_KIND_TAGS (that's for entity
    // kinds); beKIND owns definitional knowledge.
    for hit in &verdict.kinds {
        for cs in &hit.closed_sets {
            let tag = cs.to_lowercase();
            if !tag.is_empty() {
                tags.push(tag);
            }
        }
    }
    tags.sort();
    tags.dedup();
    tags
}

/// One batched kind-verdict daemon call for all document contents; maps
/// `k:<index>` verdict ids back to input order. Fail-open: daemon failure
/// yields no tags for every document.
pub fn batch_doc_kind_tags(contents: &[&str]) -> Vec<Vec<String>> {
    let verdicts = analyze_kind_verdicts(contents);
    let mut by_index: HashMap<usize, &KindVerdict> = HashMap::new();
    for verdict in &verdicts {
        if let Some(index) = verdict
            .id
            .strip_prefix("k:")
            .and_then(|rest| rest.parse::<usize>().ok())
        {
            by_index.insert(index, verdict);
        }
    }
    (0..contents.len())
        .map(|i| {
            by_index
                .get(&i)
                .map(|verdict| doc_kind_tags(verdict))
                .unwrap_or_default()
        })
        .collect()
}

/// All definitional tags for document contents: scope tags + admitted kind
/// tags + activity categories, merged and deduplicated. One batched daemon
/// call per layer, plus the caller-owned food lexicon (bekind under-extracts
/// food entities).
pub fn batch_doc_semantic_tags(contents: &[&str]) -> Vec<Vec<String>> {
    let scope_tags = batch_doc_scope_tags(contents);
    let kind_tags = batch_doc_kind_tags(contents);
    let activity_tags = crate::behood_query::analyze_activity_categories(contents);
    scope_tags
        .into_iter()
        .zip(kind_tags)
        .zip(activity_tags)
        .map(|((mut scope, kind), activity)| {
            scope.extend(kind);
            scope.extend(activity);
            scope.sort();
            scope.dedup();
            scope
        })
        .collect()
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
            where_phrase: String::new(),
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

    fn kind_verdict(kinds: &[(&str, &str)]) -> KindVerdict {
        KindVerdict {
            id: "k:0".to_string(),
            kinds: kinds
                .iter()
                .map(|(text, kind)| crate::behood_query::KindHit {
                    text: text.to_string(),
                    kind: kind.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    #[test]
    fn doc_kind_tags_emit_admitted_herb_and_food_only() {
        assert_eq!(
            doc_kind_tags(&kind_verdict(&[("cilantro", "herb"), ("coffee", "food"), ("Jean", "person")])),
            vec!["food".to_string(), "herb".to_string()]
        );
    }

    #[test]
    fn doc_kind_tags_empty_without_admitted_kinds() {
        assert!(doc_kind_tags(&kind_verdict(&[("Jean", "person")])).is_empty());
        assert!(doc_kind_tags(&kind_verdict(&[])).is_empty());
    }

    #[test]
    fn doc_kind_tags_lowercase_and_dedup() {
        assert_eq!(
            doc_kind_tags(&kind_verdict(&[("Basil", "Herb"), ("cilantro", "herb")])),
            vec!["herb".to_string()]
        );
    }


    /// The `semantic_tags` index field uses tantivy's default TEXT analyzer;
    /// every tag token ("weekend", "weekday", "habitual", "herb") must survive
    /// it as a single lowercase token, otherwise the QueryParser would
    /// silently match nothing on the synthetic terms.
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
