use crate::ids::stable_chunk_id;
use crate::query_expansion::normalize_for_index;
use std::collections::HashMap;

use super::helpers::*;
use super::model::*;

pub(crate) fn build_semantic_doc_state(
    record: &DocRecord,
    claim_scoring: bool,
) -> SemanticDocState {
    let mut state = SemanticDocState {
        doc_id: record.doc_id.clone(),
        chunk_ids: Vec::new(),
        chunks: HashMap::new(),
        entity_to_docs: HashMap::new(),
        term_to_docs: HashMap::new(),
        claim_to_docs: HashMap::new(),
        topic: record
            .probable_topic
            .as_ref()
            .map(|topic| topic.to_lowercase()),
        doc_type: record
            .doc_type_guess
            .as_ref()
            .map(|doc_type| doc_type.to_lowercase()),
    };

    let chunks = if record.section_chunks.is_empty() {
        vec![SectionChunk {
            chunk_id: stable_chunk_id(
                &record.doc_id,
                record
                    .headings
                    .first()
                    .map(String::as_str)
                    .unwrap_or("(document)"),
                &record.content,
                1,
                record.content.lines().count().max(1),
            ),
            heading: record
                .headings
                .first()
                .cloned()
                .unwrap_or_else(|| "(document)".to_string()),
            content: record.content.clone(),
            start_line: 1,
            end_line: record.content.lines().count().max(1),
            timestamp: record.timestamp.clone(),
            key_entities: record
                .key_entities
                .iter()
                .map(|e| normalize_for_index(&e.text))
                .filter(|v| !v.is_empty())
                .collect(),
            important_terms: record
                .important_terms
                .iter()
                .map(|t| normalize_for_index(&t.term))
                .filter(|v| !v.is_empty())
                .collect(),
        }]
    } else {
        record.section_chunks.clone()
    };

    for chunk in &chunks {
        state.chunk_ids.push(chunk.chunk_id.clone());
        state.chunks.insert(
            chunk.chunk_id.clone(),
            SemanticChunkState {
                chunk_id: chunk.chunk_id.clone(),
                doc_id: record.doc_id.clone(),
                heading: chunk.heading.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                key_entities: chunk.key_entities.clone(),
                important_terms: chunk.important_terms.clone(),
            },
        );
        for term in &chunk.important_terms {
            let key = normalize_for_index(term);
            if key.is_empty() {
                continue;
            }
            state
                .term_to_docs
                .entry(key)
                .or_default()
                .push(TermPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.8,
                });
        }
        for token in tokenize_query_terms(&chunk.heading) {
            state
                .term_to_docs
                .entry(token)
                .or_default()
                .push(TermPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.4,
                });
        }
        for entity in &chunk.key_entities {
            let key = normalize_for_index(entity);
            if key.is_empty() {
                continue;
            }
            state
                .entity_to_docs
                .entry(key)
                .or_default()
                .push(EntityPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.9,
                });
        }
    }

    if claim_scoring {
        for claim in &record.top_claims {
            for token in claim_tokens(claim) {
                state
                    .claim_to_docs
                    .entry(token)
                    .or_default()
                    .push(TermPosting {
                        doc_id: record.doc_id.clone(),
                        score: claim.confidence.max(0.1),
                    });
            }
        }
    }

    for postings in state.entity_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for postings in state.term_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for postings in state.claim_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    state
}

impl SemanticAggregate {
    pub(crate) fn insert_doc_state(&mut self, state: &SemanticDocState) {
        self.doc_to_chunks
            .insert(state.doc_id.clone(), state.chunk_ids.clone());
        for chunk_id in &state.chunk_ids {
            self.chunk_to_doc
                .insert(chunk_id.clone(), state.doc_id.clone());
            if let Some(chunk) = state.chunks.get(chunk_id) {
                self.chunk_ranges
                    .insert(chunk_id.clone(), (chunk.start_line, chunk.end_line));
                for term in &chunk.important_terms {
                    let key = normalize_for_index(term);
                    if key.is_empty() {
                        continue;
                    }
                    self.term_to_chunks
                        .entry(key)
                        .or_default()
                        .push((chunk_id.clone(), 0.8));
                }
                for token in tokenize_query_terms(&chunk.heading) {
                    self.term_to_chunks
                        .entry(token)
                        .or_default()
                        .push((chunk_id.clone(), 0.4));
                }
                for entity in &chunk.key_entities {
                    let key = normalize_for_index(entity);
                    if key.is_empty() {
                        continue;
                    }
                    self.entity_to_chunks
                        .entry(key)
                        .or_default()
                        .push((chunk_id.clone(), 0.9));
                }
            }
        }
        for (key, postings) in &state.entity_to_docs {
            self.entity_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.iter().cloned());
            self.pending_entity_posting_keys.insert(key.clone());
        }
        for (key, postings) in &state.term_to_docs {
            self.term_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.iter().cloned());
            self.pending_term_posting_keys.insert(key.clone());
        }
        for (key, postings) in &state.claim_to_docs {
            self.claim_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.iter().cloned());
            self.pending_claim_posting_keys.insert(key.clone());
        }
        if let Some(topic) = state.topic.as_ref() {
            self.topic_to_docs
                .entry(topic.clone())
                .or_default()
                .push(state.doc_id.clone());
        }
        if let Some(doc_type) = state.doc_type.as_ref() {
            self.doc_type_to_docs
                .entry(doc_type.clone())
                .or_default()
                .push(state.doc_id.clone());
        }
    }

    pub(crate) fn sort_pending_postings(&mut self) {
        for key in self.pending_entity_posting_keys.drain() {
            if let Some(postings) = self.entity_to_docs.get_mut(&key) {
                postings.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        for key in self.pending_term_posting_keys.drain() {
            if let Some(postings) = self.term_to_docs.get_mut(&key) {
                postings.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        for key in self.pending_claim_posting_keys.drain() {
            if let Some(postings) = self.claim_to_docs.get_mut(&key) {
                postings.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    pub(crate) fn remove_doc(&mut self, doc_id: &str) {
        let Some(chunk_ids) = self.doc_to_chunks.remove(doc_id) else {
            return;
        };
        for chunk_id in chunk_ids {
            self.chunk_to_doc.remove(&chunk_id);
            self.chunk_ranges.remove(&chunk_id);
            for postings in self.term_to_chunks.values_mut() {
                postings.retain(|(id, _)| id != &chunk_id);
            }
            for postings in self.entity_to_chunks.values_mut() {
                postings.retain(|(id, _)| id != &chunk_id);
            }
        }
        self.term_to_chunks
            .retain(|_, postings| !postings.is_empty());
        self.entity_to_chunks
            .retain(|_, postings| !postings.is_empty());
        for postings in self.entity_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.entity_to_docs
            .retain(|_, postings| !postings.is_empty());
        for postings in self.term_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.term_to_docs.retain(|_, postings| !postings.is_empty());
        for postings in self.claim_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.claim_to_docs
            .retain(|_, postings| !postings.is_empty());
        for docs in self.topic_to_docs.values_mut() {
            docs.retain(|id| id != doc_id);
        }
        self.topic_to_docs.retain(|_, docs| !docs.is_empty());
        for docs in self.doc_type_to_docs.values_mut() {
            docs.retain(|id| id != doc_id);
        }
        self.doc_type_to_docs.retain(|_, docs| !docs.is_empty());
    }
}

#[cfg(test)]
mod aggregate_tests {
    use super::*;

    // Structural regression check: the hot refresh path calls remove_doc for
    // every changed document, including first-time inserts. Keep unknown IDs
    // out of the aggregate-wide posting scans. Replace this with direct work
    // instrumentation if removal gains a measurable public result.
    #[test]
    fn remove_doc_short_circuits_for_unindexed_documents() {
        let source = include_str!("semantic.rs");
        let start = source
            .find("pub(crate) fn remove_doc(&mut self, doc_id: &str)")
            .expect("remove_doc implementation exists");
        let body = &source[start..];
        let guard_pos = body
            .find("let Some(chunk_ids) = self.doc_to_chunks.remove(doc_id) else {")
            .expect("unindexed document guard exists");
        let return_pos = body[guard_pos..]
            .find("return;")
            .map(|offset| guard_pos + offset)
            .expect("unindexed document guard returns early");
        let scan_pos = body
            .find("self.entity_to_docs.values_mut()")
            .expect("aggregate posting scan exists");
        assert!(return_pos < scan_pos, "guard must precede aggregate scans");
    }

    #[test]
    fn pending_postings_are_sorted_before_snapshot_publication() {
        let mut aggregate = SemanticAggregate::default();
        let mut state = SemanticDocState {
            doc_id: "lower".to_string(),
            ..SemanticDocState::default()
        };
        state.entity_to_docs.insert(
            "entity".to_string(),
            vec![EntityPosting {
                doc_id: "lower".to_string(),
                score: 0.4,
            }],
        );
        aggregate.insert_doc_state(&state);
        let mut higher = SemanticDocState {
            doc_id: "higher".to_string(),
            ..SemanticDocState::default()
        };
        higher.entity_to_docs.insert(
            "entity".to_string(),
            vec![EntityPosting {
                doc_id: "higher".to_string(),
                score: 0.9,
            }],
        );
        aggregate.insert_doc_state(&higher);
        let postings = aggregate.entity_to_docs.get("entity").unwrap();
        assert_eq!(postings[0].doc_id, "lower");
        assert_eq!(postings[1].doc_id, "higher");
        let index = crate::index::MemoryIndex::from_records_with_semantic_aggregate(
            Vec::new(),
            aggregate,
            false,
            false,
            false,
        );
        let postings = index.entity_to_docs.get("entity").unwrap();
        assert_eq!(postings[0].doc_id, "higher");
        assert_eq!(postings[1].doc_id, "lower");
    }
}
