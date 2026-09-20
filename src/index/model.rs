use crate::query_semantics::QueryRoutingIntent;
use crate::tier1::{RankedTerm, Tier1Entity};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tantivy::query::Bm25StatisticsProvider;
use tantivy::schema::Field;
use tantivy::{Index, IndexReader, Searcher, Term};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub timestamp: Option<String>,
    pub ner_provider: String,
    pub term_ranker: String,
    pub index_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionChunk {
    pub chunk_id: String,
    pub heading: String,
    pub content: String,
    #[serde(default)]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: usize,
    #[serde(default)]
    pub timestamp: Option<String>,
    pub key_entities: Vec<String>,
    pub important_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocRecord {
    pub doc_id: String,
    pub source: String,
    #[serde(skip_serializing)]
    pub content: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub author_agent: Option<String>,
    pub group_id: Option<String>,
    #[serde(default)]
    pub filters: std::collections::BTreeMap<String, String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub headings: Vec<String>,
    #[serde(default)]
    pub doc_links: Vec<String>,
    #[serde(default)]
    pub temporal_terms: Vec<String>,
    pub key_entities: Vec<Tier1Entity>,
    pub important_terms: Vec<RankedTerm>,
    pub section_chunks: Vec<SectionChunk>,
    pub embedding: Option<Vec<f32>>,
    pub top_claims: Vec<Claim>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct SemanticChunkState {
    pub chunk_id: String,
    pub doc_id: String,
    pub heading: String,
    pub start_line: usize,
    pub end_line: usize,
    pub key_entities: Vec<String>,
    pub important_terms: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticDocState {
    pub doc_id: String,
    pub chunk_ids: Vec<String>,
    pub chunks: HashMap<String, SemanticChunkState>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic: Option<String>,
    pub doc_type: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticAggregate {
    pub chunk_to_doc: HashMap<String, String>,
    pub doc_to_chunks: HashMap<String, Vec<String>>,
    pub chunk_ranges: HashMap<String, (usize, usize)>,
    pub term_to_chunks: HashMap<String, Vec<(String, f32)>>,
    pub entity_to_chunks: HashMap<String, Vec<(String, f32)>>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChunkMeta {
    pub(crate) doc_u32: u32,
    pub(crate) chunk_id: String,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
}

#[derive(Debug, Default, Clone)]
struct TrieNode {
    children: HashMap<char, usize>,
    value: Option<u32>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct LexiconTrie {
    nodes: Vec<TrieNode>,
}

impl LexiconTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    pub(crate) fn insert(&mut self, key: &str, value: u32) {
        let mut cur = 0usize;
        for ch in key.chars() {
            let next = if let Some(idx) = self.nodes[cur].children.get(&ch) {
                *idx
            } else {
                let idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[cur].children.insert(ch, idx);
                idx
            };
            cur = next;
        }
        self.nodes[cur].value = Some(value);
    }

    pub(crate) fn get(&self, key: &str) -> Option<u32> {
        let mut cur = 0usize;
        for ch in key.chars() {
            cur = *self.nodes.get(cur)?.children.get(&ch)?;
        }
        self.nodes.get(cur)?.value
    }

    pub(crate) fn prefix_ids(&self, prefix: &str, limit: usize) -> Vec<u32> {
        let mut cur = 0usize;
        for ch in prefix.chars() {
            let Some(next) = self
                .nodes
                .get(cur)
                .and_then(|n| n.children.get(&ch))
                .copied()
            else {
                return Vec::new();
            };
            cur = next;
        }
        let mut out = Vec::new();
        let mut stack = vec![cur];
        while let Some(idx) = stack.pop() {
            if let Some(v) = self.nodes[idx].value {
                out.push(v);
                if out.len() >= limit {
                    break;
                }
            }
            for next in self.nodes[idx].children.values() {
                stack.push(*next);
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IntervalEntry {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) chunk_u32: u32,
}

#[derive(Debug, Clone)]
struct IntervalNode {
    center: usize,
    overlaps: Vec<IntervalEntry>,
    left: Option<Box<IntervalNode>>,
    right: Option<Box<IntervalNode>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct IntervalTree {
    root: Option<Box<IntervalNode>>,
}

impl IntervalTree {
    pub(crate) fn build(entries: Vec<IntervalEntry>) -> Self {
        fn build_rec(mut entries: Vec<IntervalEntry>) -> Option<Box<IntervalNode>> {
            if entries.is_empty() {
                return None;
            }
            let mut points = entries
                .iter()
                .map(|e| e.start + (e.end.saturating_sub(e.start) / 2))
                .collect::<Vec<_>>();
            points.sort_unstable();
            let center = points[points.len() / 2];
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut overlaps = Vec::new();
            for e in entries.drain(..) {
                if e.end < center {
                    left.push(e);
                } else if e.start > center {
                    right.push(e);
                } else {
                    overlaps.push(e);
                }
            }
            Some(Box::new(IntervalNode {
                center,
                overlaps,
                left: build_rec(left),
                right: build_rec(right),
            }))
        }
        Self {
            root: build_rec(entries),
        }
    }

    pub(crate) fn query(&self, start: usize, end: usize) -> Vec<u32> {
        fn walk(node: &Option<Box<IntervalNode>>, start: usize, end: usize, out: &mut Vec<u32>) {
            let Some(node) = node else {
                return;
            };
            for e in &node.overlaps {
                if e.start <= end && start <= e.end {
                    out.push(e.chunk_u32);
                }
            }
            if start <= node.center {
                walk(&node.left, start, end, out);
            }
            if end >= node.center {
                walk(&node.right, start, end, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.root, start, end, &mut out);
        out
    }
}

#[derive(Serialize)]
pub struct MemoryIndex {
    pub docs: HashMap<String, DocRecord>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
    #[serde(skip_serializing)]
    pub(crate) filter_postings: HashMap<String, HashMap<String, RoaringBitmap>>,
    #[serde(skip_serializing)]
    pub(crate) lexical: Option<LexicalIndex>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) doc_id_to_u32: HashMap<String, u32>,
    #[serde(skip_serializing)]
    pub(crate) doc_u32_to_id: Vec<String>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) chunk_id_to_u32: HashMap<String, u32>,
    #[serde(skip_serializing)]
    pub(crate) chunks: Vec<ChunkMeta>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) doc_to_chunks: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) term_lexicon: HashMap<String, u32>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) entity_lexicon: HashMap<String, u32>,
    #[serde(skip_serializing)]
    pub(crate) term_postings_chunk: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    pub(crate) entity_postings_chunk: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    pub(crate) entity_postings_doc: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    pub(crate) term_postings_doc: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) chunk_terms: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub(crate) chunk_entities: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    pub(crate) term_trie: LexiconTrie,
    #[serde(skip_serializing)]
    pub(crate) entity_trie: LexiconTrie,
    #[serde(skip_serializing)]
    pub(crate) doc_interval_trees: Vec<IntervalTree>,
    #[serde(skip_serializing)]
    pub(crate) doc_key_entities: Vec<Vec<String>>,
    #[serde(skip_serializing)]
    pub(crate) doc_rerank_texts: Vec<String>,
    #[serde(skip_serializing)]
    pub(crate) doc_rerank_tokens: Vec<Vec<String>>,
    #[serde(skip_serializing)]
    pub(crate) doc_has_number: Vec<bool>,
    #[serde(skip_serializing)]
    pub(crate) claim_scoring: bool,
    #[serde(skip_serializing)]
    pub(crate) text_rerank_ngram: bool,
    #[serde(skip_serializing)]
    pub(crate) text_rerank_lcs: bool,
}

pub(crate) struct LexicalIndex {
    pub(crate) index: Index,
    pub(crate) reader: IndexReader,
    pub(crate) doc_id_f: Field,
    pub(crate) content_f: Field,
    pub(crate) headings_f: Field,
    pub(crate) terms_f: Field,
    pub(crate) entities_f: Field,
    pub(crate) temporal_f: Field,
}

/// A live corpus-wide BM25 statistics provider assembled from shard searchers.
/// Tantivy asks for statistics only for terms present in the parsed query, so
/// this avoids copying a corpus-sized term dictionary into the coordinator.
#[derive(Clone)]
pub struct GlobalBm25Statistics {
    searchers: Arc<Vec<Searcher>>,
    cache: Arc<GlobalBm25StatisticsCache>,
    generation: u64,
}

struct GlobalBm25StatisticsCache {
    total_num_docs: Mutex<Option<u64>>,
    total_num_tokens: Mutex<HashMap<Field, u64>>,
    doc_freq: Mutex<HashMap<Term, u64>>,
}

impl GlobalBm25Statistics {
    pub fn from_indexes<'a>(indexes: impl IntoIterator<Item = &'a MemoryIndex>) -> Self {
        Self::from_indexes_with_generation(indexes, 0)
    }

    pub fn from_indexes_with_generation<'a>(
        indexes: impl IntoIterator<Item = &'a MemoryIndex>,
        generation: u64,
    ) -> Self {
        let searchers = indexes
            .into_iter()
            .filter_map(|index| index.lexical.as_ref())
            .map(|lexical| lexical.reader.searcher())
            .collect();
        Self {
            searchers: Arc::new(searchers),
            cache: Arc::new(GlobalBm25StatisticsCache {
                total_num_docs: Mutex::new(None),
                total_num_tokens: Mutex::new(HashMap::new()),
                doc_freq: Mutex::new(HashMap::new()),
            }),
            generation,
        }
    }

    pub fn shard_count(&self) -> usize {
        self.searchers.len()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Bm25StatisticsProvider for GlobalBm25Statistics {
    fn total_num_tokens(&self, field: Field) -> tantivy::Result<u64> {
        if let Some(total) = self
            .cache
            .total_num_tokens
            .lock()
            .expect("global BM25 token cache lock poisoned")
            .get(&field)
            .copied()
        {
            return Ok(total);
        }
        let total =
            self.searchers
                .iter()
                .try_fold(0u64, |total, searcher| -> tantivy::Result<u64> {
                    Ok(total + Bm25StatisticsProvider::total_num_tokens(searcher, field)?)
                })?;
        self.cache
            .total_num_tokens
            .lock()
            .expect("global BM25 token cache lock poisoned")
            .insert(field, total);
        Ok(total)
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        if let Some(total) = *self
            .cache
            .total_num_docs
            .lock()
            .expect("global BM25 document cache lock poisoned")
        {
            return Ok(total);
        }
        let total =
            self.searchers
                .iter()
                .try_fold(0u64, |total, searcher| -> tantivy::Result<u64> {
                    Ok(total + Bm25StatisticsProvider::total_num_docs(searcher)?)
                })?;
        *self
            .cache
            .total_num_docs
            .lock()
            .expect("global BM25 document cache lock poisoned") = Some(total);
        Ok(total)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        if let Some(total) = self
            .cache
            .doc_freq
            .lock()
            .expect("global BM25 doc-frequency cache lock poisoned")
            .get(term)
            .copied()
        {
            return Ok(total);
        }
        let total =
            self.searchers
                .iter()
                .try_fold(0u64, |total, searcher| -> tantivy::Result<u64> {
                    Ok(total + Bm25StatisticsProvider::doc_freq(searcher, term)?)
                })?;
        self.cache
            .doc_freq
            .lock()
            .expect("global BM25 doc-frequency cache lock poisoned")
            .insert(term.clone(), total);
        Ok(total)
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ScoreBreakdown {
    pub lexical_score: f32,
    pub entity_score: f32,
    pub term_score: f32,
    pub claim_score: f32,
    pub semantic_score: f32,
    pub topic_score: f32,
    pub doc_type_score: f32,
    pub recency_score: f32,
    pub graph_link_score: f32,
    pub entity_graph_score: f32,
    pub sequence_rerank_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub source: String,
    pub group_id: Option<String>,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub matched_entities: Vec<String>,
    pub matched_terms: Vec<String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_status: Option<crate::semantic_relations::SemanticStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relation_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relation_evidence: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QueryTimings {
    pub total_ms: f64,
    pub refresh_ms: f64,
    pub lexical_bm25_ms: f64,
    pub snapshot_query_ms: f64,
    pub rerank_ms: f64,
    pub parse_ms: f64,
    pub sparse_scoring_ms: f64,
    pub lexical_merge_ms: f64,
    pub posting_scoring_ms: f64,
    pub routing_seed_ms: f64,
    pub candidate_accumulation_ms: f64,
    pub candidate_rank_ms: f64,
    pub metadata_ms: f64,
    pub graph_ms: f64,
    pub entity_graph_ms: f64,
    pub sequence_rerank_ms: f64,
    pub evidence_ms: f64,
    pub group_build_ms: f64,
    pub group_sort_ms: f64,
    pub ranking_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QueryDiagnostics {
    pub query_terms: usize,
    pub expanded_terms: usize,
    pub lexical_hits: usize,
    pub candidates: usize,
    /// Generation of the immutable snapshot used for the query. Single-index
    /// callers use zero because they do not publish segmented snapshots.
    pub snapshot_generation: u64,
    /// Present for segmented queries so callers can distinguish no hits from
    /// an incomplete fan-out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_completeness: Option<crate::segments::ShardQueryCompleteness>,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalQueryContext<'a> {
    pub starts_from: Option<&'a str>,
    pub ends_at: Option<&'a str>,
    pub window_days: i64,
    pub hard_filter: bool,
    pub time_hint: Option<TemporalQueryHint>,
    pub query_routing_intent: Option<QueryRoutingIntent>,
    pub has_explicit_temporal: bool,
    pub allowed_doc_ids: Option<&'a HashSet<String>>,
    pub allowed_doc_bitmap: Option<&'a RoaringBitmap>,
    pub allowed_segment_doc_bitmaps: Option<&'a HashMap<String, RoaringBitmap>>,
}

impl<'a> Default for TemporalQueryContext<'a> {
    fn default() -> Self {
        Self {
            starts_from: None,
            ends_at: None,
            window_days: 7,
            hard_filter: false,
            time_hint: None,
            query_routing_intent: None,
            has_explicit_temporal: false,
            allowed_doc_ids: None,
            allowed_doc_bitmap: None,
            allowed_segment_doc_bitmaps: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CandidateState {
    pub(crate) score: f32,
    pub(crate) breakdown: ScoreBreakdown,
    pub(crate) matched_entities: Vec<String>,
    pub(crate) matched_terms: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct EvidenceFeatures {
    pub(crate) score: f32,
    pub(crate) matched_terms: usize,
    pub(crate) has_number: bool,
    pub(crate) has_unit: bool,
    pub(crate) has_date: bool,
    pub(crate) has_temporal_terms: bool,
    pub(crate) predicate_signal: bool,
    pub(crate) distractor_penalty: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedDocRecord {
    pub doc_id: String,
    pub source: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub group_id: Option<String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedMemoryIndex {
    pub docs: HashMap<String, RedactedDocRecord>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalQueryHint {
    Past,
    Present,
    Ongoing,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedMemoryCore {
    pub(crate) entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub(crate) term_to_docs: HashMap<String, Vec<TermPosting>>,
    #[serde(default)]
    pub(crate) claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub(crate) topic_to_docs: HashMap<String, Vec<String>>,
    pub(crate) doc_type_to_docs: HashMap<String, Vec<String>>,
    pub(crate) doc_id_to_u32: HashMap<String, u32>,
    pub(crate) doc_u32_to_id: Vec<String>,
    pub(crate) chunk_id_to_u32: HashMap<String, u32>,
    pub(crate) chunks: Vec<ChunkMeta>,
    pub(crate) doc_to_chunks: Vec<Vec<u32>>,
    pub(crate) term_lexicon: HashMap<String, u32>,
    pub(crate) entity_lexicon: HashMap<String, u32>,
    pub(crate) term_postings_chunk: Vec<Vec<(u32, f32)>>,
    pub(crate) entity_postings_chunk: Vec<Vec<(u32, f32)>>,
    pub(crate) chunk_terms: Vec<Vec<u32>>,
    pub(crate) chunk_entities: Vec<Vec<u32>>,
}
