use super::LLM_CONTEXT_DUPLICATE_DOC_PENALTY;
use crate::cli::LlmChunkStrategy;
use crate::graph::{normalize_concept, Tier0Record};
use crate::index::{MemoryIndex, SectionChunk};
use crate::query_semantics::QueryAnalysis;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
pub(crate) struct Tier0IndexFile {
    pub(crate) tier: String,
    pub(crate) generated_at_unix: u64,
    pub(crate) path: String,
    pub(crate) document_count: usize,
    pub(crate) documents_by_id: BTreeMap<String, Tier0Record>,
}

#[derive(Serialize)]
pub(crate) struct QueryOutput {
    pub(crate) query: String,
    pub(crate) elapsed_ms: u128,
    pub(crate) result_count: usize,
    pub(crate) analysis: QueryAnalysis,
    pub(crate) results: Vec<crate::index::SearchResult>,
    pub(crate) aggregation: Option<crate::aggregation::AggregateOutput>,
}

#[derive(Clone, Serialize)]
pub(crate) struct LlmContextChunk {
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
    pub(crate) text: String,
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
pub(crate) struct LlmContextOutput {
    mode: String,
    query: String,
    generated_at_unix: u64,
    corpus_path: String,
    elapsed_ms: u128,
    chunk_result_count: usize,
    pub(crate) top_chunks: Vec<LlmContextChunk>,
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

pub(crate) fn build_llm_context_output(
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
        chunk_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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
            chunk_candidates_scored
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
                    Some((best, _)) if *best >= combined => {
                        leftovers.push((doc_id, combined, chunk))
                    }
                    Some((_, prev)) => {
                        leftovers.push((
                            doc_id.clone(),
                            *best_by_doc.get(&doc_id).map(|x| &x.0).unwrap_or(&combined),
                            prev.clone(),
                        ));
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
