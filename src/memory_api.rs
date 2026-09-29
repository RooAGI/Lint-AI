//! Memory Add/Search API backed by Lint-AI's `IndexStore`.

use crate::conversational_rerank::{conversational_rerank, RERANK_DEEP_TOP_K, RERANK_WEIGHTS};
use crate::lang::{default_spacy_model_for_lang, Lang};
use crate::pipeline::PipelineOptions;
use crate::query_plan::PreparedQuery;
use crate::query_semantics::analyze_query;
use crate::segments::relations::{
    analyze_fact_question, extract_relations_via_spacy, extractor_model_for_turns,
    query_structured, relation_turns_from_docs, try_extract_key_phrases_via_spacy, RelationIndex,
    RelationTurn,
};
use crate::session_prepare::is_follow_up;
use crate::{IndexStore, SourceDocument};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const USER_FILTER: &str = "memory_user_id";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddRequest {
    pub request_id: String,
    pub messages: Vec<Message>,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub timestamp: Option<i64>,
    pub content: String,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
    #[serde(default)]
    pub supersedes_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[allow(dead_code)]
    pub options: Option<Vec<String>>,
    pub user_id: String,
    pub top_k: usize,
    /// Optional conversation session. When supplied, the search loads the
    /// bounded prior state for (user_id, session_id) and resolves follow-up
    /// phrasing and temporal anchors against it before retrieval. Absent or
    /// empty means stateless search: the query is analyzed on its own. This is
    /// a state key only; it never becomes an implicit `group_id` filter.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Conversation-state scope. When supplied, follow-up resolution and
    /// the conversational reranker scope to this value instead of `user_id`.
    /// This lets callers like the benchmark share conversation state across
    /// a fixed scope while filtering documents by a different user.
    #[serde(default)]
    pub scope: Option<String>,
    /// Additional document filters (e.g. `provider`) merged with the
    /// user-ownership filter. Absent means no additional filtering.
    #[serde(default)]
    pub filters: Option<BTreeMap<String, String>>,
    /// Content language override (`"en"`, `"zh"`, `"ko"`). Absent (default)
    /// auto-detects per text from script statistics. Currently selects the
    /// spaCy model for the structured-relations path; lexical retrieval is
    /// script-aware regardless.
    #[serde(default)]
    pub lang: Option<Lang>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub data: Vec<SearchMemory>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub user_id: String,
    #[serde(default)]
    pub doc_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SupersedeRequest {
    pub user_id: String,
    pub replacement_id: String,
    pub old_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    pub user_id: String,
    #[serde(default)]
    pub memory_id: String,
    #[serde(default)]
    pub include_inactive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRequest {
    pub user_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub include_inactive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse {
    pub data: Vec<MemoryRecord>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub user_id: String,
    #[serde(default)]
    pub memory_id: String,
    pub content: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub role: Option<String>,
    pub user_id: String,
    pub session_id: Option<String>,
    pub created_at: Option<String>,
    pub expires_at_ms: Option<u64>,
    pub supersedes_id: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMemory {
    pub id: String,
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub role: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    /// Why a structured-fact hit was returned (e.g. "shared relation: Rome").
    /// Empty for lexical hits; skipped in serialization when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relation_evidence: Vec<String>,
}

pub struct MemoryService {
    store: IndexStore,
    superseded_ids: HashSet<(String, String)>,
    request_fingerprints: HashMap<(String, String), String>,
    conversation_states:
        std::sync::Arc<std::sync::Mutex<crate::conversation_state::ConversationStateStore>>,
    /// Lazily-built dependency-parse relation index for the structured-fact
    /// path. The write path never pays for it; see [`RelationsCache`].
    relations_cache: Mutex<RelationsCache>,
    /// Background key-phrase enrichment (see [`KeyPhraseEnrichment`]).
    enrichment: std::sync::Arc<Mutex<KeyPhraseEnrichment>>,
    /// Agent bulletin-board state: per-board sequence counters and
    /// request-id dedup, all assigned under one lock by the single writer.
    board_state: Mutex<BoardState>,
}

/// Mutable bulletin-board state owned by the single writer process.
///
/// Per-board sequence counters are assigned here while holding the lock,
/// so concurrent `board_post` calls get distinct increasing sequences.
/// Lazily initialized from the store's max sequence per board, so a
/// restart recovers without gaps or reuse.
#[derive(Default)]
struct BoardState {
    /// board_id -> next sequence to assign.
    next_sequence: std::collections::HashMap<String, u64>,
    /// (board_id, request_id) -> post_id, for retry idempotency.
    request_ids: std::collections::HashMap<(String, String), String>,
}

/// Background key-phrase enrichment for newly written documents.
///
/// The write path stays fast: `add` queues the new documents' turns here
/// and returns. A single worker thread drains the queue in batches, runs
/// the extractor's `key_phrases_only` mode off-thread, and posts results
/// to the inbox. The next `&mut` entry point (add/refresh/...) applies the
/// inbox to the store, so the following refresh folds the phrases into the
/// cap-exempt segment entity channel. Everything is fail-open: enrichment
/// never blocks or fails a write, and a missing/slow extractor simply
/// leaves documents with empty key phrases (today's behavior).
#[derive(Default)]
struct KeyPhraseEnrichment {
    /// Turns awaiting extraction, in queue order.
    pending: std::collections::VecDeque<PendingTurn>,
    /// Extracted phrases awaiting application to the store.
    inbox: Vec<EnrichedDoc>,
    /// True while a worker thread owns the current batch.
    worker_running: bool,
}

/// One queued turn plus the content hash at queue time, so a document that
/// was replaced while its batch was in flight is recognized as stale and
/// its phrases are dropped (the replacement re-queues itself via `add`).
struct PendingTurn {
    turn: crate::segments::relations::RelationTurn,
    content_hash: String,
}

/// Extractor output for one document, ready to backfill.
struct EnrichedDoc {
    doc_id: String,
    content_hash: String,
    phrases: Vec<crate::source::KeyPhrase>,
}

/// Max queued turns waiting for extraction; enrichment is best-effort, so
/// overflow drops the oldest queued turns (the documents stay searchable).
const KEY_PHRASE_PENDING_MAX: usize = 512;
/// Max turns per extractor subprocess run.
const KEY_PHRASE_BATCH_SIZE: usize = 32;
/// How long the worker lingers before draining, so rapid writes coalesce
/// into one extractor run instead of one model load per write.
const KEY_PHRASE_LINGER_MS: u64 = 500;
/// Bound for one synchronous query-time backfill run: long enough for a
/// warm spaCy model over a batch of short documents, short enough that a
/// cold or stuck extractor cannot stall a query. Fail-open on expiry.
const KEY_PHRASE_BACKFILL_TIMEOUT_SECS: u64 = 30;

/// Run the key-phrase extractor on `turns`, giving up after `timeout_secs`.
/// Returns `None` when the run times out or the worker thread fails
/// `Some(phrases)` when the extractor ran to completion — even when it
/// found nothing, which is then a legitimate empty result — or `None` when
/// the run timed out or the extractor subprocess failed (transient: the
/// documents must stay un-stamped so a later attempt retries). Shared by
/// the background worker and the query-time backfill so both paths bound
/// the subprocess the same way.
fn run_key_phrase_extraction_bounded(
    turns: &[crate::segments::relations::RelationTurn],
    script: Option<&std::path::Path>,
    timeout_secs: u64,
) -> Option<Vec<crate::segments::relations::RawKeyPhrase>> {
    let turns: Vec<crate::segments::relations::RelationTurn> = turns.to_vec();
    let script = script.map(|s| s.to_path_buf());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let model = extractor_model_for_turns(&turns);
        let out = try_extract_key_phrases_via_spacy(
            &turns,
            script.as_deref(),
            std::time::Duration::from_secs(timeout_secs),
            model,
        );
        let _ = tx.send(out);
    });
    rx.recv_timeout(std::time::Duration::from_secs(timeout_secs))
        .ok()
        .flatten()
}

impl MemoryService {
    /// Extract key phrases for the given documents. Associated function
    /// (not a method) so the server can run it with no service lock held:
    /// it is slow (subprocess with its own timeout) and holding the lock
    /// would stall concurrent queries behind the extractor. `None` means
    /// the run timed out or the extractor failed (transient: retry later);
    /// `Some` means the extractor ran to completion.
    pub fn extract_key_phrases_for_docs(
        docs: &[SourceDocument],
        script: Option<&std::path::Path>,
    ) -> Option<Vec<crate::segments::relations::RawKeyPhrase>> {
        let refs: Vec<&SourceDocument> = docs.iter().collect();
        let turns = relation_turns_from_docs(&refs);
        run_key_phrase_extraction_bounded(&turns, script, KEY_PHRASE_BACKFILL_TIMEOUT_SECS)
    }
}

/// Lazily-built dependency-parse relation indexes backing the structured-fact
/// retrieval path ("which place did both X and Y visit", "where was X
/// between <dates>").
///
/// One index per user, each composed only from documents visible to that
/// user (ownership, supersession, and expiry are enforced before composition,
/// never after). The index builds on the first structured question in one
/// batched extractor run, and rebuilds when the visible document set changes
/// (fingerprinted by document id, content, and relation-relevant metadata).
/// The write path never pays for it. Extraction is fail-open: a
/// missing/slow/broken extractor leaves the cache empty and callers fall
/// back to lexical retrieval.
#[derive(Default)]
struct RelationsCache {
    per_user: HashMap<String, UserRelations>,
}

/// Relation index for a single user. `fingerprint` covers the exact visible
/// document set the index was composed from; any content or metadata change
/// invalidates it.
#[derive(Default)]
struct UserRelations {
    fingerprint: u64,
    index: Option<Arc<RelationIndex>>,
}

/// Upper bound on distinct users holding a relation index at once. Eviction
/// only costs a future lazy rebuild, so the bound keeps a multi-user service
/// from growing without limit.
const RELATIONS_CACHE_MAX_USERS: usize = 8;

/// Upper bound for the batched extractor subprocess on the query path.
/// Past this, the query degrades to lexical retrieval rather than stalling.
const RELATIONS_EXTRACT_TIMEOUT_SECS: u64 = 120;

/// Fingerprint the exact document set an index was composed from. Every field
/// that feeds `relation_turns_from_docs` (plus the visibility filters) is
/// hashed, so any content or metadata change under an existing id invalidates
/// the cached index.
fn relations_fingerprint(docs: &[&SourceDocument]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Callers pass one user's visible documents sorted by doc id.
    let mut hasher = DefaultHasher::new();
    docs.len().hash(&mut hasher);
    for doc in docs {
        doc.doc_id.hash(&mut hasher);
        doc.content.hash(&mut hasher);
        doc.author_agent.hash(&mut hasher);
        doc.source.hash(&mut hasher);
        doc.group_id.hash(&mut hasher);
        doc.timestamp.hash(&mut hasher);
        // BTreeMap iteration is sorted, so the filter map hashes
        // deterministically; visibility-relevant metadata is part of the
        // fingerprint.
        for (key, value) in &doc.filters {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Pick the spaCy model for structured relation extraction.
/// An explicit non-`Auto` request language pins its model; `Lang::Auto`
/// and an omitted language both follow the turns' detected script.
fn spacy_model_for_request(lang: Option<Lang>, turns: &[RelationTurn]) -> &'static str {
    match lang {
        Some(l) if !matches!(l, Lang::Auto) => default_spacy_model_for_lang(l.resolve("")),
        _ => extractor_model_for_turns(turns),
    }
}

/// Build (or reuse) the relation index for one user's visible document set.
/// Returns `None` when extraction fails or times out, so the caller falls
/// through to the lexical path.
fn relations_index_for(
    docs: &[&SourceDocument],
    user_id: &str,
    cache: &Mutex<RelationsCache>,
    lang: Option<Lang>,
) -> Option<Arc<RelationIndex>> {
    // The lock is held across the build so concurrent structured queries for
    // the same user share one extractor run instead of racing duplicates.
    let mut cached = cache.lock().expect("relations cache lock poisoned");
    if cached.per_user.len() >= RELATIONS_CACHE_MAX_USERS && !cached.per_user.contains_key(user_id)
    {
        // Bounded: evict an arbitrary entry to make room. Eviction only
        // costs a future lazy rebuild.
        if let Some(key) = cached.per_user.keys().next().cloned() {
            cached.per_user.remove(&key);
        }
    }
    let entry = cached.per_user.entry(user_id.to_string()).or_default();
    let fingerprint = relations_fingerprint(docs);
    if entry.fingerprint == fingerprint {
        return entry.index.clone();
    }
    let turns = relation_turns_from_docs(docs);
    // Bound the subprocess: run extraction on a worker thread and give up
    // after the timeout, leaving the cache empty (fail-open to lexical).
    // An explicit request language selects the spaCy model; otherwise the
    // model follows the turns' detected script.
    let model: String = spacy_model_for_request(lang, &turns).to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let output = extract_relations_via_spacy(
            &turns,
            std::time::Duration::from_secs(RELATIONS_EXTRACT_TIMEOUT_SECS),
            &model,
        );
        let _ = tx.send(output.relations);
    });
    let raw_relations = match rx.recv_timeout(std::time::Duration::from_secs(
        RELATIONS_EXTRACT_TIMEOUT_SECS,
    )) {
        Ok(relations) => relations,
        Err(_) => {
            eprintln!(
                "relations: extractor timed out after {RELATIONS_EXTRACT_TIMEOUT_SECS}s; \
                 structured-fact path disabled for this query (not cached, will retry)"
            );
            // Do NOT cache the empty index: a timeout/failure must not
            // poison the cache for unchanged data. Return a transient empty
            // index so this query fails open to lexical, and the next
            // query retries extraction.
            let build_turns = relation_turns_from_docs(docs);
            return Some(Arc::new(RelationIndex::build(&build_turns, &[])));
        }
    };
    let build_turns = relation_turns_from_docs(docs);
    let index = Arc::new(RelationIndex::build(&build_turns, &raw_relations));
    entry.fingerprint = fingerprint;
    entry.index = Some(Arc::clone(&index));
    Some(index)
}

/// Structured-fact retrieval: analyze the question, consult the lazily-built
/// relation index, and return evidence as `SearchResult`s scored above the
/// lexical range. Empty when the option is off, the question is not a
/// structured fact question, or the index holds no evidence.
///
/// Visibility is enforced before composition: the index is built only from
/// this user's live documents (ownership, supersession, expiry), so a
/// foreign, superseded, or expired document can never influence relation
/// composition, even if it would later be filtered from the hits.
#[allow(clippy::too_many_arguments)]
fn structured_fact_results(
    docs: &[&SourceDocument],
    options: &PipelineOptions,
    cache: &Mutex<RelationsCache>,
    superseded_ids: &HashSet<(String, String)>,
    request: &SearchRequest,
) -> Vec<crate::SearchResult> {
    if !options.structured_fact_retrieval || request.query.trim().is_empty() {
        return Vec::new();
    }
    // Classify before building: only structured fact questions pay for the
    // extractor subprocess. (query_structured re-analyzes; the classifier
    // itself is cheap string matching.)
    if analyze_fact_question(&request.query).is_none() {
        return Vec::new();
    }
    let now_ms = unix_time_ms();
    let mut visible: Vec<&SourceDocument> = docs
        .iter()
        .copied()
        .filter(|doc| owns_memory(doc, &request.user_id))
        .filter(|doc| !superseded_ids.contains(&(request.user_id.clone(), doc.doc_id.clone())))
        .filter(|doc| {
            doc.filters
                .get("expires_at_ms")
                .and_then(|value| value.parse::<u64>().ok())
                .is_none_or(|expires_at| expires_at > now_ms)
        })
        .collect();
    visible.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    let index = match relations_index_for(&visible, &request.user_id, cache, request.lang) {
        Some(index) => index,
        None => return Vec::new(),
    };
    let hits = match query_structured(&index, &request.query) {
        Some(hits) => hits,
        None => return Vec::new(),
    };
    // The index holds only visible documents, so every hit maps; the lookup
    // stays defensive against index/query skew.
    let by_id: HashMap<&str, &SourceDocument> =
        visible.iter().map(|d| (d.doc_id.as_str(), *d)).collect();
    hits.into_iter()
        .filter_map(|hit| {
            let doc = by_id.get(hit.doc_id.as_str())?;
            Some(crate::SearchResult {
                doc_id: hit.doc_id,
                source: doc.source.clone(),
                group_id: doc.group_id.clone(),
                score: hit.score,
                score_breakdown: crate::index::ScoreBreakdown::default(),
                matched_entities: Vec::new(),
                matched_terms: Vec::new(),
                probable_topic: None,
                doc_type_guess: None,
                semantic_status: None,
                superseded_by: None,
                relation_confidence: Some(hit.confidence),
                relation_evidence: vec![hit.evidence_label],
            })
        })
        .collect()
}

/// Blend structured-fact evidence ahead of lexical results, deduplicating
/// by document id and respecting `top_k`.
fn blend_structured_first(
    structured: Vec<crate::SearchResult>,
    lexical: Vec<crate::SearchResult>,
    top_k: usize,
) -> Vec<crate::SearchResult> {
    if structured.is_empty() {
        return lexical;
    }
    let seen: HashSet<String> = structured.iter().map(|r| r.doc_id.clone()).collect();
    let mut blended = structured;
    let room = top_k.saturating_sub(blended.len());
    blended.extend(
        lexical
            .into_iter()
            .filter(|r| !seen.contains(r.doc_id.as_str()))
            .take(room),
    );
    blended
}

/// Apply a fixed boost to the given result indices and re-sort by score
/// (stable sort, so unboosted relative order is preserved). Records the
/// amount in each hit's score breakdown via `record`. Returns the number
/// boosted. Pure: no daemon, no I/O. Boost only — results are never
/// removed, demoted, or filtered here.
fn boost_result_indices(
    results: &mut [crate::SearchResult],
    indices: &[usize],
    amount: f32,
    record: impl Fn(&mut crate::index::ScoreBreakdown, f32),
) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut n = 0usize;
    for &ri in indices {
        if ri < results.len() && seen.insert(ri) {
            results[ri].score += amount;
            record(&mut results[ri].score_breakdown, amount);
            n += 1;
        }
    }
    if n > 0 {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    n
}

/// Fixed additive boost for activity↔venue matches (Luyi 2026-09-28).
///
/// The activity→venue relation is WORLD KNOWLEDGE — it lives here in
/// lint-ai, not in bekind (Luyi's ruling, mem-14: "Where does the user
/// like to eat out?" / "The user's favorite restaurant is Din Tai
/// Fung"). bekind's share is purely linguistic: the scope verdict's
/// `activity_phrase` (the extracted verb phrase, e.g. "eat out") and
/// the kind verdicts ("restaurant" is kind=place). This table maps the
/// former to the latter. Boost only — never a filter. Recorded per hit in
/// `score_breakdown.activity_venue_boost` so the boost is measurable in
/// serialized responses.
const ACTIVITY_VENUE_BOOST: f32 = 25.0;

/// Admitted 2026-09-28 (Luyi): activity → typical venue words. Fixed
/// enumeration, fail-open (an activity not listed here yields no boost).
/// Each new activity needs its own explicit admission — the table does
/// not grow by fuzzy matching, and bekind never sees it.
const ACTIVITY_VENUES: &[(&str, &[&str])] = &[
    ("eat out", &["restaurant", "cafe", "diner", "eatery"]),
    ("dine", &["restaurant", "cafe", "diner", "eatery"]),
    ("eat", &["restaurant", "cafe", "diner", "eatery"]),
    ("swim", &["pool"]),
    ("run", &["park", "track"]),
    ("jog", &["park", "track"]),
    ("watch movie", &["cinema", "theater", "theatre"]),
    ("drink coffee", &["cafe", "coffee shop"]),
    ("shop", &["mall", "store"]),
    ("work out", &["gym"]),
    ("hike", &["trail", "park", "mountain"]),
    ("play golf", &["golf course", "country club"]),
];

/// Whether a descriptor's text names one of the venue words: whole-word
/// (for multi-word venues, whole-phrase) match on normalized tokens, so
/// "The user's favorite restaurant" matches "restaurant" and "the local
/// coffee shop" matches "coffee shop", but "parked" never matches "park".
fn venue_text_matches(descriptor_text: &str, venues: &[&str]) -> bool {
    let norm = format!(
        " {} ",
        descriptor_text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
    );
    venues
        .iter()
        .any(|v| norm.contains(&format!(" {v} ")))
}

/// Pure activity↔venue match predicate, unit-testable without the daemon.
///
/// The question's activity is bekind's extracted `activity_phrase`
/// (lowercased); it looks up the venue set in the admitted table. A fact
/// matches when any of its PLACE-kind descriptors names a venue word.
/// Only place-kind descriptors count — a venue is a place, and the kind
/// restriction is the precision control. A `false` here only withholds
/// the boost — it never removes or demotes.
fn activity_venue_match(activity: &str, fact: &crate::behood_query::KindVerdict) -> bool {
    let activity = activity.trim().to_lowercase();
    let venues = match ACTIVITY_VENUES.iter().find(|(a, _)| *a == activity) {
        Some((_, v)) => *v,
        None => return false,
    };
    fact.kinds
        .iter()
        .filter(|hit| hit.kind == "place")
        .any(|hit| venue_text_matches(&hit.text, venues))
}

/// Activity↔venue rank boost over blended search results.
///
/// Query-time only, no reindexing: the question's activity phrase comes
/// from the behood daemon's scope verdict (existing
/// `analyze_scope_verdicts` path — no new protocol), then all candidate
/// fact texts go through the daemon in ONE batched kind request
/// (milliseconds), reusing the same verdict shape the kind boost needs.
/// Fail-open throughout: no daemon, no binary, no scope/kind support,
/// empty activity phrase, or an activity not in the admitted table →
/// results returned unchanged.
fn apply_activity_venue_boost(
    store: &IndexStore,
    query: &str,
    mut results: Vec<crate::SearchResult>,
) -> Vec<crate::SearchResult> {
    if results.is_empty() {
        return results;
    }
    let activity = match crate::behood_query::analyze_scope_verdicts(&[query])
        .into_iter()
        .next()
    {
        Some(v) => v.activity_phrase.trim().to_lowercase(),
        None => return results,
    };
    if !ACTIVITY_VENUES.iter().any(|(a, _)| *a == activity) {
        return results;
    }
    // Map each blended result to its document text; results whose documents
    // are missing are skipped (their verdict slot is simply absent).
    let mut text_to_result: Vec<usize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (ri, r) in results.iter().enumerate() {
        if let Some(doc) = store.source_document_by_id(&r.doc_id) {
            text_to_result.push(ri);
            texts.push(doc.content.clone());
        }
    }
    if texts.is_empty() {
        return results;
    }
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let verdicts = crate::behood_query::analyze_kind_verdicts(&text_refs);
    let matched: Vec<usize> = verdicts
        .iter()
        .filter_map(|v| {
            let ti: usize = v.id.strip_prefix("k:")?.parse().ok()?;
            let ri = *text_to_result.get(ti)?;
            activity_venue_match(&activity, v).then_some(ri)
        })
        .collect();
    boost_result_indices(&mut results, &matched, ACTIVITY_VENUE_BOOST, |b, a| {
        b.activity_venue_boost += a
    });
    results
}

impl MemoryService {
    /// Create an in-memory service without exposing the storage implementation
    /// to application callers.
    pub fn in_memory(options: PipelineOptions) -> Self {
        Self::new(IndexStore::in_memory(options))
    }

    /// Open a persistent memory service without exposing `IndexStore` in the
    /// application-facing construction API. Hooks and the MCP server open a
    /// fresh service per invocation against the same index root; a disk-backed
    /// conversation store lets session state (follow-up context, temporal
    /// anchors) survive across those invocations.
    pub fn at_path(index_root: impl AsRef<Path>, options: PipelineOptions) -> anyhow::Result<Self> {
        let index_root = index_root.as_ref();
        let mut service = Self::new(IndexStore::at_path(index_root, options)?);
        service.conversation_states = std::sync::Arc::new(std::sync::Mutex::new(
            crate::conversation_state::ConversationStateStore::open_under(index_root),
        ));
        Ok(service)
    }

    pub(crate) fn new(store: IndexStore) -> Self {
        let superseded_ids = store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                Some((
                    doc.filters.get(USER_FILTER)?.clone(),
                    doc.filters.get("supersedes_id")?.clone(),
                ))
            })
            .collect();
        let request_fingerprints = store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                let persisted = doc.filters.get("request_fingerprint")?;
                let fingerprint = if persisted.starts_with("v2:") {
                    persisted.clone()
                } else {
                    let messages = serde_json::from_str::<Vec<Message>>(persisted).ok()?;
                    request_fingerprint(doc.group_id.as_deref()?, &messages).ok()?
                };
                Some((
                    (
                        doc.filters.get(USER_FILTER)?.clone(),
                        doc.filters.get("request_id")?.clone(),
                    ),
                    fingerprint,
                ))
            })
            .collect();
        Self {
            store,
            superseded_ids,
            request_fingerprints,
            conversation_states: std::sync::Arc::new(std::sync::Mutex::new(
                crate::conversation_state::ConversationStateStore::new(None),
            )),
            relations_cache: Mutex::new(RelationsCache::default()),
            enrichment: std::sync::Arc::new(Mutex::new(KeyPhraseEnrichment::default())),
            board_state: Mutex::new(BoardState::default()),
        }
    }

    /// Return bounded structural information for operational status views.
    /// Callers that expose this information should sanitize identifiers and
    /// persistence paths before sending it outside the process.
    pub fn inspection(&self) -> crate::pipeline::IndexStoreInspection {
        self.store.inspection()
    }

    /// Publish pending mutations. Normal mutation methods already call this;
    /// this method is provided for hosts that batch lower-level changes.
    pub fn refresh(&mut self) -> anyhow::Result<()> {
        self.drain_enrichment_inbox();
        self.store.refresh()
    }

    /// Queue freshly written documents for background key-phrase
    /// enrichment. No-op when `key_phrase_enrichment` is off. The write
    /// itself already finished; this only schedules the async work.
    fn queue_key_phrase_enrichment(&mut self, docs: &[SourceDocument]) {
        if !self.store.options().key_phrase_enrichment || docs.is_empty() {
            return;
        }
        let refs: Vec<&SourceDocument> = docs.iter().collect();
        let hashes: HashMap<&str, String> = refs
            .iter()
            .map(|doc| {
                (
                    doc.doc_id.as_str(),
                    crate::pipeline::key_phrase_content_hash(&doc.content),
                )
            })
            .collect();
        // relation_turns_from_docs sorts by (session_id, turn_idx); join the
        // staleness hashes back by doc_id.
        let turns = relation_turns_from_docs(&refs);
        {
            let mut state = self.enrichment.lock().expect("enrichment lock poisoned");
            for turn in turns {
                let content_hash = hashes
                    .get(turn.doc_id.as_str())
                    .cloned()
                    .unwrap_or_default();
                // Latest write wins: drop any older queued turn for this doc.
                if let Some(pos) = state
                    .pending
                    .iter()
                    .position(|p| p.turn.doc_id == turn.doc_id)
                {
                    state.pending.remove(pos);
                }
                state.pending.push_back(PendingTurn { turn, content_hash });
                while state.pending.len() > KEY_PHRASE_PENDING_MAX {
                    state.pending.pop_front();
                }
            }
        }
        self.maybe_spawn_enrichment_worker();
    }

    /// Spawn the enrichment worker if none is running and work is pending.
    /// The worker drains the queue completely (batch by batch), running the
    /// extractor off-thread and posting results to the inbox; a later write
    /// landing mid-drain is picked up by the same worker.
    fn maybe_spawn_enrichment_worker(&self) {
        let script = self.store.options().extractor_script.clone();
        let state = std::sync::Arc::clone(&self.enrichment);
        {
            let mut guard = state.lock().expect("enrichment lock poisoned");
            if guard.worker_running || guard.pending.is_empty() {
                return;
            }
            guard.worker_running = true;
        }
        std::thread::spawn(move || {
            // Linger so rapid writes coalesce into one extractor run
            // instead of one model load per write.
            std::thread::sleep(std::time::Duration::from_millis(KEY_PHRASE_LINGER_MS));
            // Drain the queue completely: a single spawn handles every batch,
            // so no later write is needed to pick up stragglers. `worker_running`
            // stays true until the queue is empty, which coalesces any write
            // that lands mid-drain into this same worker.
            loop {
                let batch: Vec<PendingTurn> = {
                    let mut guard = state.lock().expect("enrichment lock poisoned");
                    if guard.pending.is_empty() {
                        guard.worker_running = false;
                        return;
                    }
                    let take = guard.pending.len().min(KEY_PHRASE_BATCH_SIZE);
                    guard.pending.drain(..take).collect()
                };
                let raw = {
                    let turns: Vec<_> = batch.iter().map(|p| p.turn.clone()).collect();
                    run_key_phrase_extraction_bounded(
                        &turns,
                        script.as_deref(),
                        RELATIONS_EXTRACT_TIMEOUT_SECS,
                    )
                };
                let mut guard = state.lock().expect("enrichment lock poisoned");
                let Some(phrases) = raw else {
                    // Transient failure (timeout): re-queue the batch so a
                    // later worker — or a query-time backfill — retries it.
                    // The documents stay un-stamped; stamping them now would
                    // mark a failed run as "done" permanently.
                    for pending in batch.into_iter().rev() {
                        guard.pending.push_front(pending);
                    }
                    guard.worker_running = false;
                    return;
                };
                for pending in &batch {
                    let doc_phrases: Vec<crate::source::KeyPhrase> = phrases
                        .iter()
                        .filter(|p| p.doc_id == pending.turn.doc_id)
                        .map(|p| crate::source::KeyPhrase {
                            text: p.text.clone(),
                            kind: p.kind.clone(),
                        })
                        .collect();
                    guard.inbox.push(EnrichedDoc {
                        doc_id: pending.turn.doc_id.clone(),
                        content_hash: pending.content_hash.clone(),
                        phrases: doc_phrases,
                    });
                }
            }
        });
    }

    /// Apply finished enrichment batches to the store. Documents whose
    /// content changed while their batch was in flight are skipped as
    /// stale (their replacement re-queued itself via `add`/`update`).
    fn drain_enrichment_inbox(&mut self) {
        let inbox: Vec<EnrichedDoc> = self
            .enrichment
            .lock()
            .expect("enrichment lock poisoned")
            .inbox
            .drain(..)
            .collect();
        if inbox.is_empty() {
            return;
        }
        for item in inbox {
            self.store
                .set_key_phrases(&item.doc_id, &item.content_hash, item.phrases);
        }
    }

    /// Drop queued and finished enrichment work for a deleted document.
    fn drop_enrichment_for_doc(&self, doc_id: &str) {
        let mut state = self.enrichment.lock().expect("enrichment lock poisoned");
        state.pending.retain(|p| p.turn.doc_id != doc_id);
        state.inbox.retain(|e| e.doc_id != doc_id);
    }

    /// Whether any document still needs key-phrase extraction. Cheap scan
    /// with early exit: read-only callers (the server search handler) use it
    /// to decide whether a write-lock backfill is worthwhile, keeping the
    /// steady state at zero extra cost.
    pub fn key_phrase_backfill_needed(&self) -> bool {
        if !self.store.options().key_phrase_enrichment {
            return false;
        }
        if !self
            .enrichment
            .lock()
            .expect("enrichment lock poisoned")
            .pending
            .is_empty()
        {
            return true;
        }
        self.store
            .source_documents()
            .iter()
            .any(|doc| Self::doc_needs_key_phrases(doc))
    }

    /// True when `doc` has no key phrases and extraction has not completed
    /// for its current content. A document whose extraction finished with a
    /// legitimately empty phrase list carries a stamp matching its content
    /// hash, so it is not mistaken for pending work.
    fn doc_needs_key_phrases(doc: &SourceDocument) -> bool {
        if doc.content.trim().is_empty() || !doc.key_phrases.is_empty() {
            return false;
        }
        doc.key_phrase_extraction_hash != crate::pipeline::key_phrase_content_hash(&doc.content)
    }

    /// Read-only snapshot for a query-time backfill: the documents still
    /// needing key-phrase extraction (deterministic order, at most one
    /// extractor batch) plus the extractor script override. The caller runs
    /// the subprocess via [`extract_key_phrases_for_docs`] without holding
    /// any service lock, then applies the result with
    /// [`apply_key_phrase_backfill`]. Returns no documents when enrichment
    /// is disabled.
    pub fn key_phrase_backfill_snapshot(
        &self,
    ) -> (Vec<SourceDocument>, Option<std::path::PathBuf>) {
        if !self.store.options().key_phrase_enrichment {
            return (Vec::new(), None);
        }
        // Deterministic order so repeated calls converge instead of
        // starving documents.
        let mut missing: Vec<&SourceDocument> = self
            .store
            .source_documents()
            .into_iter()
            .filter(|doc| Self::doc_needs_key_phrases(doc))
            .collect();
        missing.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        missing.truncate(KEY_PHRASE_BATCH_SIZE);
        let docs: Vec<SourceDocument> = missing.into_iter().cloned().collect();
        let script = self.store.options().extractor_script.clone();
        (docs, script)
    }

    /// Apply extractor output from a query-time backfill: first apply
    /// anything the background worker already finished, then validate
    /// content hashes and stamp each document (a legitimately empty phrase
    /// list still gets its stamp), drop the handled documents from the
    /// background queue so the worker does not redo them, and refresh so
    /// the current query retries them. Fast; refresh failures are fail-open.
    /// Returns the number of documents extracted.
    pub fn apply_key_phrase_backfill(
        &mut self,
        docs: &[SourceDocument],
        raw: Option<Vec<crate::segments::relations::RawKeyPhrase>>,
    ) -> usize {
        // Apply anything the background worker already finished; those
        // documents are done without needing the extractor output.
        self.drain_enrichment_inbox();
        let raw = match raw {
            Some(raw) => raw,
            None => {
                if let Err(error) = self.store.refresh() {
                    eprintln!("key-phrase backfill refresh failed (fail-open): {error:#}");
                }
                return 0;
            }
        };
        let mut extracted = 0;
        for doc in docs {
            // The extractor ran off-thread without any service lock; a
            // concurrent write may have changed the content since the
            // snapshot — `set_key_phrases` re-validates the hash and skips
            // stale results.
            let hash = crate::pipeline::key_phrase_content_hash(&doc.content);
            let doc_phrases: Vec<crate::source::KeyPhrase> = raw
                .iter()
                .filter(|p| p.doc_id == doc.doc_id)
                .map(|p| crate::source::KeyPhrase {
                    text: p.text.clone(),
                    kind: p.kind.clone(),
                })
                .collect();
            self.store.set_key_phrases(&doc.doc_id, &hash, doc_phrases);
            // The worker need not redo these documents; only drop the queue
            // entry when it is for the same content this run extracted (a
            // newer write re-queued itself and must be left alone).
            self.enrichment
                .lock()
                .expect("enrichment lock poisoned")
                .pending
                .retain(|p| !(p.turn.doc_id == doc.doc_id && p.content_hash == hash));
            extracted += 1;
        }
        if let Err(error) = self.store.refresh() {
            eprintln!("key-phrase backfill refresh failed (fail-open): {error:#}");
        }
        extracted
    }

    /// Synchronously extract key phrases for documents that still need them
    /// (at most one extractor batch) and refresh the index so the current
    /// query benefits. This is the correctness backstop for documents
    /// written by short-lived processes — provider hooks queue enrichment,
    /// but their worker dies with the process before the linger elapses, so
    /// without this those documents would stay phrase-less until some later
    /// write happened to restart the worker. For the lock-holding server
    /// path use the snapshot/extract/apply split instead, so the subprocess
    /// run does not block concurrent queries.
    ///
    /// Bounded and fail-open: an extractor failure or timeout leaves the
    /// documents phrase-less (a later query retries). Returns the number of
    /// documents extracted.
    pub fn backfill_key_phrases(&mut self) -> usize {
        let (docs, script) = self.key_phrase_backfill_snapshot();
        if docs.is_empty() {
            return 0;
        }
        let raw = Self::extract_key_phrases_for_docs(&docs, script.as_deref());
        self.apply_key_phrase_backfill(&docs, raw)
    }

    #[cfg(test)]
    pub(crate) fn enrichment_queue_len(&self) -> usize {
        self.enrichment
            .lock()
            .expect("enrichment lock poisoned")
            .pending
            .len()
    }

    pub fn add(&mut self, request: AddRequest) -> anyhow::Result<AddResponse> {
        self.drain_enrichment_inbox();
        let response = self.add_unpublished(request)?;
        self.store.refresh()?;
        Ok(response)
    }

    /// Adds several requests and publishes one snapshot after all mutations.
    /// Each request retains the normal per-request message limit and validation.
    pub fn add_batch(&mut self, requests: Vec<AddRequest>) -> anyhow::Result<Vec<AddResponse>> {
        if requests.is_empty() {
            anyhow::bail!("requests must not be empty");
        }
        self.drain_enrichment_inbox();
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            responses.push(self.add_unpublished(request)?);
        }
        self.store.refresh()?;
        Ok(responses)
    }

    fn add_unpublished(&mut self, request: AddRequest) -> anyhow::Result<AddResponse> {
        validate_identifier(&request.request_id, "request_id")?;
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.session_id, "session_id")?;
        if request.messages.is_empty() {
            anyhow::bail!("messages must not be empty");
        }
        if request.messages.len() > MAX_MESSAGES_PER_REQUEST {
            anyhow::bail!("messages must contain at most {MAX_MESSAGES_PER_REQUEST} items");
        }
        for (message_index, message) in request.messages.iter().enumerate() {
            validate_message(message, message_index)?;
        }
        let fingerprint = request_fingerprint(&request.session_id, &request.messages)?;
        let request_key = (request.user_id.clone(), request.request_id.clone());
        if let Some(previous) = self.request_fingerprints.get(&request_key) {
            if previous != &fingerprint {
                anyhow::bail!("request_id was already used with different content");
            }
            return Ok(AddResponse {
                success: true,
                request_id: request.request_id,
                user_id: request.user_id,
                session_id: request.session_id,
            });
        }

        let mut new_docs = Vec::with_capacity(request.messages.len());
        for (message_index, message) in request.messages.iter().enumerate() {
            let source = format!(
                "memory://{}/{}/{}",
                request.user_id, request.session_id, message_index
            );
            let mut filters = BTreeMap::new();
            filters.insert(USER_FILTER.to_string(), request.user_id.clone());
            filters.insert("request_id".to_string(), request.request_id.clone());
            filters.insert("request_fingerprint".to_string(), fingerprint.clone());
            if let Some(expires_at_ms) = message.expires_at_ms {
                filters.insert("expires_at_ms".to_string(), expires_at_ms.to_string());
            }
            if let Some(supersedes_id) = &message.supersedes_id {
                validate_identifier(supersedes_id, "supersedes_id")?;
                self.superseded_ids
                    .insert((request.user_id.clone(), supersedes_id.clone()));
                filters.insert("supersedes_id".to_string(), supersedes_id.clone());
            }
            let timestamp = message
                .timestamp
                .map(|millis| timestamp_to_rfc3339(millis, "timestamp"))
                .transpose()?;
            new_docs.push(SourceDocument {
                doc_id: crate::stable_doc_id_from_source(&format!(
                    "{}:{}:{message_index}",
                    request.user_id, request.request_id
                )),
                source,
                content: format!("{}: {}", message.role, message.content),
                concept: "memory".to_string(),
                group_id: Some(request.session_id.clone()),
                headings: vec![],
                links: vec![],
                timestamp,
                doc_length: message.content.len(),
                author_agent: Some(message.role.clone()),
                filters,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
        // The documents are written; key phrases arrive asynchronously.
        // `upsert` queues each document for background enrichment.
        for doc in new_docs {
            self.upsert(doc);
        }

        self.request_fingerprints.insert(request_key, fingerprint);
        Ok(AddResponse {
            success: true,
            request_id: request.request_id,
            user_id: request.user_id,
            session_id: request.session_id,
        })
    }

    /// The API-facing search entry point (HTTP server, Python bindings).
    ///
    /// This is a thin wrapper — it is NOT a separate retrieval path. All
    /// retrieval logic lives in [`MemoryService::search_with_filters`], the
    /// single search implementation. This function only translates the
    /// [`SearchRequest`] into the filter/scope args and shapes the raw hits
    /// into a [`SearchResponse`].
    ///
    /// Single-path guarantee: any change to retrieval behavior must go in
    /// `search_with_filters`, never here. This wrapper must stay logic-free
    /// (argument translation + response shaping only) so the HTTP/Python
    /// path and the MCP/benchmark path can never diverge.
    pub fn search(&mut self, request: SearchRequest) -> anyhow::Result<SearchResponse> {
        if !request.user_id.trim().is_empty() {
            validate_identifier(&request.user_id, "user_id")?;
        }
        if let Some(session_id) = request.session_id.as_deref() {
            validate_identifier(session_id, "session_id")?;
        }
        if request.query.trim().is_empty() {
            return Ok(SearchResponse { data: vec![] });
        }
        // The conversation-state scope defaults to the user; an explicit
        // scope overrides it (e.g. the benchmark shares one scope across
        // many document owners).
        let scope = request
            .scope
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| request.user_id.clone());
        // Build the document filters: ownership (unless empty, for
        // provider-scoped callers) plus any caller-supplied extras.
        let mut filters = BTreeMap::new();
        if !request.user_id.trim().is_empty() {
            filters.insert(USER_FILTER.to_string(), request.user_id.clone());
        }
        if let Some(extra) = request.filters.as_ref() {
            for (key, value) in extra {
                if key != USER_FILTER {
                    filters.insert(key.clone(), value.clone());
                }
            }
        }
        let results = self.search_with_filters(
            &request.query,
            &scope,
            request.session_id.as_deref(),
            request.top_k,
            &filters,
        )?;
        Ok(self.format_search_response(results))
    }

    pub fn get(&self, request: GetRequest) -> anyhow::Result<Option<MemoryRecord>> {
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.memory_id, "memory_id")?;
        Ok(self
            .store
            .source_document_by_id(&request.memory_id)
            .filter(|doc| owns_memory(doc, &request.user_id))
            .filter(|doc| request.include_inactive || self.is_visible(doc))
            .map(memory_record))
    }

    pub fn list(&self, request: ListRequest) -> anyhow::Result<ListResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if let Some(session_id) = &request.session_id {
            validate_identifier(session_id, "session_id")?;
        }
        let limit = request.limit.clamp(1, 100);
        let mut documents = self
            .store
            .source_documents()
            .into_iter()
            .filter(|doc| owns_memory(doc, &request.user_id))
            .filter(|doc| {
                request
                    .session_id
                    .as_deref()
                    .is_none_or(|session| doc.group_id.as_deref() == Some(session))
            })
            .filter(|doc| request.include_inactive || self.is_visible(doc))
            .collect::<Vec<_>>();
        documents.sort_by(|left, right| left.doc_id.cmp(&right.doc_id));
        if let Some(cursor) = &request.cursor {
            validate_identifier(cursor, "cursor")?;
            documents.retain(|doc| doc.doc_id > *cursor);
        }
        let has_more = documents.len() > limit;
        documents.truncate(limit);
        let next_cursor = has_more
            .then(|| documents.last().map(|doc| doc.doc_id.clone()))
            .flatten();
        Ok(ListResponse {
            data: documents.into_iter().map(memory_record).collect(),
            next_cursor,
        })
    }

    pub fn update(&mut self, request: UpdateRequest) -> anyhow::Result<Option<MemoryRecord>> {
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.memory_id, "memory_id")?;
        let Some(mut document) = self
            .store
            .source_document_by_id(&request.memory_id)
            .cloned()
        else {
            return Ok(None);
        };
        if !owns_memory(&document, &request.user_id) {
            return Ok(None);
        }
        if !self.is_visible(&document) {
            return Ok(None);
        }
        if request.content.trim().is_empty() || request.content.len() > MAX_MESSAGE_CONTENT_BYTES {
            anyhow::bail!(
                "content must be non-empty and at most {MAX_MESSAGE_CONTENT_BYTES} bytes"
            );
        }
        let role = request.role.unwrap_or_else(|| {
            document
                .author_agent
                .clone()
                .unwrap_or_else(|| "user".to_string())
        });
        if role != "user" && role != "assistant" {
            anyhow::bail!("role must be user or assistant");
        }
        document.content = format!("{role}: {}", request.content);
        document.author_agent = Some(role);
        document.timestamp = request
            .timestamp
            .map(|millis| timestamp_to_rfc3339(millis, "timestamp"))
            .transpose()?;
        match request.expires_at_ms {
            Some(expires_at) => {
                document
                    .filters
                    .insert("expires_at_ms".to_string(), expires_at.to_string());
            }
            None => {
                document.filters.remove("expires_at_ms");
            }
        }
        self.drain_enrichment_inbox();
        // Content changed under the same id: previously extracted phrases
        // describe the old text, so clear them and their extraction stamp;
        // `upsert` re-queues. The stamp must go too: a stale stamp matching
        // new content would make an empty phrase list look "done".
        document.key_phrases = Vec::new();
        document.key_phrase_extraction_hash = String::new();
        self.upsert(document);
        self.store.refresh()?;
        Ok(self
            .store
            .source_document_by_id(&request.memory_id)
            .map(memory_record))
    }

    fn is_visible(&self, doc: &SourceDocument) -> bool {
        let not_superseded = doc
            .filters
            .get(USER_FILTER)
            .map(|user| (user.clone(), doc.doc_id.clone()))
            .is_none_or(|key| !self.superseded_ids.contains(&key));
        let not_expired = doc
            .filters
            .get("expires_at_ms")
            .and_then(|value| value.parse::<u64>().ok())
            .is_none_or(|expires_at| expires_at > unix_time_ms());
        not_superseded && not_expired
    }

    /// The single search implementation. ALL retrieval goes through this
    /// function — the MCP `search` tools (Claude Code, Codex, Gemini CLI,
    /// Muse), the benchmark harnesses, and [`MemoryService::search`] (thin
    /// wrapper for the HTTP/Python API) all converge here.
    ///
    /// Pipeline: query-time key-phrase backfill → conversational query
    /// preparation → lexical retrieval with optional conversational rerank →
    /// structured-fact retrieval blended ahead of lexical → session-state
    /// observation. Returns raw hits; callers shape them via
    /// [`MemoryService::format_search_response`] (API) or the shared
    /// `search_results` formatter (agent-facing).
    ///
    /// Single-path guarantee: this is the ONLY place retrieval behavior may
    /// change. Do not add a second implementation — extend this one.
    pub fn search_with_filters(
        &mut self,
        query: &str,
        scope: &str,
        session_id: Option<&str>,
        top_k: usize,
        filters: &BTreeMap<String, String>,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        // Query-time key-phrase backfill: documents written by provider
        // hooks (separate short-lived processes whose background workers
        // die with them) get their phrases extracted synchronously here, so
        // this query benefits. Bounded to one extractor batch and fail-open;
        // usually a no-op once every document carries its extraction stamp.
        self.backfill_key_phrases();
        // Luyi 2026-09-27: wire the augmented query into production.
        // analyze_query builds "original + terms" (focus terms, entities)
        // for better routing coverage. This is the same formulation
        // validated on LongMemEval (92.4% Any@5, 84.49% Frac@5).
        let analysis = analyze_query(query);
        let query_text = analysis.augmented_query.as_str();
        let mut prepared = prepare_session_query(&self.conversation_states, scope, session_id, query_text);
        // Definitional semantic tags (Luyi 2026-09-28): computed from the
        // ORIGINAL user query, not the augmented text. Closed-set temporal
        // words ("weekend"/"weekday"), "habitual", and admitted kind tags
        // ("herb") become SHOULD TermQueries on the index's `semantic_tags`
        // field, scored by BM25 inside tantivy — a match, not a bonus.
        // Fail-open: no tags when the daemon is unavailable or the question
        // carries no definitional content.
        prepared.set_semantic_tags(crate::semantic_tags::query_semantic_tags(query));
        let do_rerank = should_conversational_rerank(
            self.store.options().conversational_rerank,
            session_id,
            query,
        );
        let depth = if do_rerank {
            top_k.max(RERANK_DEEP_TOP_K)
        } else {
            top_k
        };
        let mut lexical = self.store.query_prepared(&prepared, depth, filters)?;
        if do_rerank {
            let entities = session_resolved_entities(
                &self.conversation_states,
                scope,
                session_id.unwrap_or(""),
            );
            lexical = conversational_rerank(
                lexical,
                &self.store,
                query,
                &entities,
                filters,
                RERANK_WEIGHTS,
            );
            lexical.truncate(top_k);
        }
        // Structured-fact path: dependency-parse relation evidence blends
        // ahead of the lexical results. The ownership filter comes from the
        // caller's filters when present (benchmark); provider-scoped callers
        // (MCP) get no structured results, which is safe — lexical still works.
        let structured = {
            let user_id = filters.get(USER_FILTER).cloned().unwrap_or_default();
            if user_id.is_empty() {
                Vec::new()
            } else {
                let request = SearchRequest {
                    query: query.to_string(),
                    options: None,
                    user_id,
                    top_k,
                    session_id: session_id.map(String::from),
                    scope: Some(scope.to_string()),
                    filters: None,
                    lang: None,
                };
                let docs: Vec<&SourceDocument> = self.store.source_documents();
                structured_fact_results(
                    &docs,
                    self.store.options(),
                    &self.relations_cache,
                    &self.superseded_ids,
                    &request,
                )
            }
        };
        let results = blend_structured_first(structured, lexical, top_k);
        // Activity↔venue rank boost (Luyi 2026-09-28): additive only,
        // never a filter. No-op when the behood daemon is unavailable,
        // the question names no admitted activity, or no fact names one
        // of its venues.
        let results = apply_activity_venue_boost(&self.store, query, results);
        observe_session_search(
            &self.conversation_states,
            scope,
            session_id,
            query,
            &prepared,
            &results,
        );
        Ok(results)
    }

    /// Sync shared memory documents into the index. Runs before each MCP
    /// search so the workspace sees memories recorded by other providers.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn sync_shared_memory(
        &mut self,
        memory_root: &std::path::Path,
    ) -> anyhow::Result<bool> {
        crate::integrations::mcp_index::sync_memory_documents(memory_root, self)
    }

    /// Refresh the index after syncing, so newly synced documents are
    /// searchable. Only the Gemini CLI path needed this explicitly.
    pub(crate) fn refresh_index(&mut self) -> anyhow::Result<()> {
        // Provider hooks run refresh_index() on a fresh service: apply any
        // completed background enrichment first, so phrases the worker
        // finished are published by this refresh instead of being stranded
        // in the inbox.
        self.drain_enrichment_inbox();
        self.store.refresh()
    }

    /// Compose a workspace service with one provider's memory service into a
    /// single in-memory query view. Transfers already-published segments
    /// rather than reprocessing source documents.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn compose_segmented(
        workspace: Self,
        provider_memory: Option<Self>,
    ) -> anyhow::Result<Self> {
        // The composed view must keep the conversation-state store the hooks
        // write to: the provider memory's disk-backed store (opened under the
        // same `.lint-ai/memory/` root the hooks use), falling back to the
        // workspace store when there is no provider memory. A fresh
        // memory-only store here would lose hook-written active sessions,
        // drop explicit session state across MCP invocations, and break
        // omitted-session_id inheritance.
        let conversation_states = provider_memory
            .as_ref()
            .map(|service| Arc::clone(&service.conversation_states))
            .unwrap_or_else(|| Arc::clone(&workspace.conversation_states));
        let composed = crate::IndexStore::compose_segmented(
            workspace.store,
            provider_memory.map(|service| service.store),
        )?;
        let mut service = Self::new(composed);
        service.conversation_states = conversation_states;
        Ok(service)
    }

    /// True when the store holds no documents. Integration read paths use
    /// this for the same early exit they had on the raw store.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Upsert one source document. Integration capture paths call this, then
    /// [`MemoryService::refresh_index`], exactly as they did on the raw store.
    /// Insert or replace a source document directly. Low-level bulk-load;
    /// prefer `add` for application writes.
    /// Insert or replace one document. This is the single content-write
    /// funnel: every document entering the store is queued for background
    /// key-phrase enrichment (no-op when enrichment is disabled), so no
    /// content write path can silently bypass it. Metadata-only touches
    /// such as `supersede` use the store directly: the content is
    /// unchanged, so re-queuing would only redo identical extraction.
    pub fn upsert(&mut self, document: crate::SourceDocument) {
        self.queue_key_phrase_enrichment(std::slice::from_ref(&document));
        self.store.upsert(document)
    }

    // -----------------------------------------------------------------------
    // Agent bulletin board (see crate::board).
    // -----------------------------------------------------------------------

    /// Resolve an optional `board_id` to a concrete board ID.
    ///
    /// `None` (or empty, or the `"default"` alias) means the current
    /// session's board: the default board *points to* the board whose ID
    /// is derived from the session ID. `session_id` must resolve when the
    /// default is used, otherwise this errors — there is no silent
    /// fallback to a shared board. When `ensure_default` is set, the
    /// session board's persistent record is created if missing — used by
    /// `board_post` so the first post creates it under the writer path.
    /// Reads pass false and treat a never-created session board as empty.
    fn resolve_board_id(
        &mut self,
        board_id: Option<&str>,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
        ensure_default: bool,
    ) -> anyhow::Result<String> {
        use crate::board::*;
        let trimmed = board_id.map(str::trim).filter(|s| !s.is_empty());
        // The "default" alias resolves exactly like an omitted board_id.
        let trimmed = match trimmed {
            Some(id) if is_default_alias(id) => None,
            other => other,
        };
        match trimmed {
            Some(id) => {
                if ensure_default {
                    if let Some(sid) = session_id.map(str::trim).filter(|s| !s.is_empty()) {
                        if is_default_board(id, owner, workspace, sid) {
                            self.board_open_session(owner, workspace, sid, DEFAULT_BOARD_TITLE)?;
                        }
                    }
                }
                if self.board_info(id, owner, workspace, session_id)?.is_none() {
                    anyhow::bail!("unknown board_id: {id}");
                }
                Ok(id.to_string())
            }
            None => {
                let sid = session_id
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "no session_id: pass session_id (or a board_id) to use the default board"
                        )
                    })?;
                if ensure_default {
                    let board = self.board_open_session(owner, workspace, sid, DEFAULT_BOARD_TITLE)?;
                    Ok(board.board_id)
                } else {
                    Ok(default_board_id(owner, workspace, sid))
                }
            }
        }
    }

    /// Idempotently open a board for (owner, workspace, key).
    ///
    /// Returns the existing board when the key is already taken; otherwise
    /// creates it. The board ID is deterministic for the triple, so
    /// concurrent opens converge without coordination.
    ///
    /// The key `"default"` is an alias for the current session's board —
    /// use [`Self::board_open_session`] for it (the MCP layer redirects).
    /// Keys starting with `"session:"` are rejected: that namespace is
    /// reserved for session boards.
    pub fn board_open(
        &mut self,
        owner: &str,
        workspace: &str,
        key: &str,
        title: &str,
    ) -> anyhow::Result<crate::board::Board> {
        use crate::board::*;
        if key.trim().is_empty() {
            anyhow::bail!("board key must not be empty");
        }
        if key.starts_with(SESSION_BOARD_KEY_PREFIX) {
            anyhow::bail!(
                "board key namespace {SESSION_BOARD_KEY_PREFIX:?} is reserved for session boards"
            );
        }
        if is_default_alias(key) {
            anyhow::bail!(
                "board key \"default\" is reserved: it aliases the current session's board; use board_open_session"
            );
        }
        self.board_open_inner(owner, workspace, key, title)
    }

    /// Inner open without the reserved-namespace guard; session boards
    /// are created through this path.
    fn board_open_inner(
        &mut self,
        owner: &str,
        workspace: &str,
        key: &str,
        title: &str,
    ) -> anyhow::Result<crate::board::Board> {
        use crate::board::*;
        let board_id = board_id_for(owner, workspace, key);
        // Fast path: board already persisted. Check the store directly
        // rather than board_info: board_info synthesizes the default
        // board when unpersisted, but board_open must persist it.
        let persisted = self
            .store
            .source_documents()
            .into_iter()
            .find(|d| d.doc_id == board_doc_id(&board_id))
            .filter(|d| {
                d.filters.get(BOARD_ID_FILTER).map(String::as_str) == Some(board_id.as_str())
            });
        if let Some(doc) = persisted {
            if let Some(board) = board_from_doc_content(&doc.content) {
                return Ok(board);
            }
        }
        let board = Board {
            board_id: board_id.clone(),
            owner: owner.to_string(),
            workspace: workspace.to_string(),
            key: key.to_string(),
            title: title.to_string(),
            created_at: Utc::now().to_rfc3339(),
        };
        let doc = crate::SourceDocument {
            doc_id: board_doc_id(&board_id),
            source: "board-registry".to_string(),
            content: board_doc_content(&board),
            concept: format!("board {}", board.key),
            group_id: None,
            headings: vec![],
            links: vec![],
            timestamp: Some(board.created_at.clone()),
            doc_length: board.title.len(),
            author_agent: None,
            filters: board_doc_filters(&board),
            key_phrases: vec![],
            key_phrase_extraction_hash: String::new(),
        };
        // Bypass the enrichment funnel: board definitions are metadata, not
        // searchable content.
        self.store.upsert(doc);
        self.store.refresh()?;
        Ok(board)
    }

    /// Idempotently open the current session's board — the board the
    /// `"default"` alias points to. The board key is the reserved
    /// `session:{session_id}`; the persistent record is created on first
    /// call.
    pub fn board_open_session(
        &mut self,
        owner: &str,
        workspace: &str,
        session_id: &str,
        title: &str,
    ) -> anyhow::Result<crate::board::Board> {
        use crate::board::*;
        let sid = session_id.trim();
        if sid.is_empty() {
            anyhow::bail!("session_id must not be empty");
        }
        // The session key namespace is reserved, so this cannot collide
        // with a user-keyed board.
        self.board_open_inner(owner, workspace, &session_board_key(sid), title)
    }

    /// List boards owned by `owner` in `workspace`. When `session_id`
    /// resolves, the list also includes the default board: the `"default"`
    /// alias *pointing to* the current session's board (synthesized until
    /// its persistent record is created on first post).
    pub fn board_list(
        &self,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<crate::board::Board>> {
        use crate::board::*;
        let mut boards = Vec::new();
        let sid = session_id.map(str::trim).filter(|s| !s.is_empty());
        for doc in self.store.source_documents() {
            let f = &doc.filters;
            if f.get(BOARD_DOC_KIND_FILTER).map(String::as_str) != Some(BOARD_DOC_KIND) {
                continue;
            }
            if f.get(BOARD_OWNER_FILTER).map(String::as_str) != Some(owner) {
                continue;
            }
            if f.get(BOARD_WORKSPACE_FILTER).map(String::as_str) != Some(workspace) {
                continue;
            }
            if let Some(board) = board_from_doc_content(&doc.content) {
                boards.push(board);
            }
        }
        // The default alias always points at the session's board, whether
        // or not that board's record exists yet. The alias key can never
        // collide with a persisted board (session boards use the
        // "session:" key namespace).
        if let Some(s) = sid {
            boards.push(default_board(owner, workspace, s));
        }
        boards.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(boards)
    }

    /// Fetch a board by ID. Returns `None` for unknown IDs. The
    /// `"default"` alias resolves to the current session's board
    /// (synthesized until its persistent record is created).
    pub fn board_info(
        &self,
        board_id: &str,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
    ) -> anyhow::Result<Option<crate::board::Board>> {
        use crate::board::*;
        // Resolve the default alias to the session board ID first.
        let resolved;
        let board_id = if is_default_alias(board_id) {
            let sid = session_id
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!("no session_id: cannot resolve the default board alias")
                })?;
            resolved = default_board_id(owner, workspace, sid);
            resolved.as_str()
        } else {
            board_id
        };
        let doc_id = board_doc_id(board_id);
        let doc = self
            .store
            .source_documents()
            .into_iter()
            .find(|d| d.doc_id == doc_id);
        match doc {
            Some(d) => {
                // Verify the board_id filter matches: a forged doc_id must
                // not alias another board.
                if d.filters.get(BOARD_ID_FILTER).map(String::as_str) != Some(board_id) {
                    return Ok(None);
                }
                if d.filters.get(BOARD_OWNER_FILTER).map(String::as_str) != Some(owner)
                    || d.filters.get(BOARD_WORKSPACE_FILTER).map(String::as_str)
                        != Some(workspace)
                {
                    return Ok(None);
                }
                let Some(board) = board_from_doc_content(&d.content) else {
                    return Ok(None);
                };
                if board.owner != owner || board.workspace != workspace || board.board_id != board_id
                {
                    return Ok(None);
                }
                Ok(Some(board))
            }
            None => {
                let sid = session_id.map(str::trim).filter(|s| !s.is_empty());
                match sid {
                    Some(s) if is_default_board(board_id, owner, workspace, s) => {
                        Ok(Some(default_board(owner, workspace, s)))
                    }
                    _ => Ok(None),
                }
            }
        }
    }

    /// Post to a board. `board_id` may be omitted (or be the `"default"`
    /// alias) for the current session's board, whose persistent record is
    /// created on first post. An explicit unknown `board_id` is an error,
    /// never an implicit creation.
    ///
    /// Assigns the next per-board sequence while holding the board lock.
    /// `request_id` makes retries safe: a repeated `request_id` for the same
    /// board returns the original post instead of creating a duplicate —
    /// including across restarts, via a store scan when the in-memory map
    /// misses.
    pub fn board_post(
        &mut self,
        board_id: Option<&str>,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
        author_agent_id: &str,
        provider: &str,
        content: &str,
        request_id: &str,
    ) -> anyhow::Result<crate::board::BoardPost> {
        use crate::board::*;
        let board_id = self.resolve_board_id(board_id, owner, workspace, session_id, true)?;
        // The board must exist; posting never creates one implicitly
        // (the session board is ensured by resolve_board_id above).
        if self.board_info(&board_id, owner, workspace, session_id)?.is_none() {
            anyhow::bail!("unknown board_id: {board_id}");
        }
        // Idempotency: a retried request_id returns the original post.
        // Check the in-memory map first, then the store (restart recovery).
        if !request_id.is_empty() {
            let cached = {
                let state = self.board_state.lock().expect("board state lock poisoned");
                state
                    .request_ids
                    .get(&(board_id.clone(), request_id.to_string()))
                    .cloned()
            };
            if let Some(post_id) = cached {
                return self
                    .board_get(Some(&board_id), owner, workspace, session_id, &post_id)?
                    .ok_or_else(|| anyhow::anyhow!("request_id mapped to missing post"));
            }
            if let Some(post_id) = self.board_find_request_id(&board_id, request_id)? {
                // Repopulate the map so later retries stay cheap.
                let mut state = self.board_state.lock().expect("board state lock poisoned");
                state.request_ids.insert(
                    (board_id.clone(), request_id.to_string()),
                    post_id.clone(),
                );
                drop(state);
                return self
                    .board_get(Some(&board_id), owner, workspace, session_id, &post_id)?
                    .ok_or_else(|| anyhow::anyhow!("request_id mapped to missing post"));
            }
        }
        // Assign the next sequence. Lazily initialize from the store's max
        // so a restart recovers without gaps or reuse.
        // Assign the next sequence. Lazily initialize from the store's max
        // so a restart recovers without gaps or reuse.
        let mut state = self.board_state.lock().expect("board state lock poisoned");
        let sequence = match state.next_sequence.get(board_id.as_str()) {
            Some(&next) => next,
            None => {
                let max = self.board_max_sequence(&board_id)?;
                let next = max + 1;
                state.next_sequence.insert(board_id.clone(), next + 1);
                next
            }
        };
        // Bump the counter for the next post.
        state.next_sequence.insert(board_id.clone(), sequence + 1);
        let post = BoardPost {
            post_id: board_post_doc_id(&board_id, sequence),
            board_id: board_id.clone(),
            author_agent_id: author_agent_id.to_string(),
            provider: provider.to_string(),
            content: content.to_string(),
            sequence,
            created_at: Utc::now().to_rfc3339(),
        };
        if !request_id.is_empty() {
            state.request_ids.insert(
                (board_id.clone(), request_id.to_string()),
                post.post_id.clone(),
            );
        }
        drop(state);
        let doc = crate::SourceDocument {
            doc_id: post.post_id.clone(),
            source: "board-post".to_string(),
            content: post.content.clone(),
            concept: format!("board post {}:{}", board_id, sequence),
            group_id: Some(board_id.clone()),
            headings: vec![],
            links: vec![],
            timestamp: Some(post.created_at.clone()),
            doc_length: post.content.len(),
            author_agent: Some(author_agent_id.to_string()),
            filters: board_post_doc_filters(&post, request_id),
            key_phrases: vec![],
            key_phrase_extraction_hash: String::new(),
        };
        self.store.upsert(doc);
        self.store.refresh()?;
        Ok(post)
    }

    /// Find a post by (board_id, request_id) in the store. Used for
    /// retry idempotency across restarts, when the in-memory map is cold.
    fn board_find_request_id(
        &self,
        board_id: &str,
        request_id: &str,
    ) -> anyhow::Result<Option<String>> {
        use crate::board::*;
        for doc in self.store.source_documents() {
            let f = &doc.filters;
            if f.get(BOARD_DOC_KIND_FILTER).map(String::as_str) != Some(BOARD_POST_KIND) {
                continue;
            }
            if f.get(BOARD_ID_FILTER).map(String::as_str) != Some(board_id) {
                continue;
            }
            if f.get(BOARD_REQUEST_ID_FILTER).map(String::as_str) == Some(request_id) {
                return Ok(Some(doc.doc_id.clone()));
            }
        }
        Ok(None)
    }

    /// Max assigned sequence for a board, from the store. Used to
    /// initialize the in-memory counter after a restart.
    fn board_max_sequence(&self, board_id: &str) -> anyhow::Result<u64> {
        use crate::board::*;
        let mut max = 0u64;
        for doc in self.store.source_documents() {
            let f = &doc.filters;
            if f.get(BOARD_DOC_KIND_FILTER).map(String::as_str) != Some(BOARD_POST_KIND) {
                continue;
            }
            if f.get(BOARD_ID_FILTER).map(String::as_str) != Some(board_id) {
                continue;
            }
            if let Some(seq_str) = f.get(BOARD_SEQUENCE_FILTER) {
                if let Ok(seq) = seq_str.parse::<u64>() {
                    max = max.max(seq);
                }
            }
        }
        Ok(max)
    }

    /// Read posts in posting order. `board_id` may be omitted (or be the
    /// `"default"` alias) for the current session's board.
    /// `after_sequence` is exclusive; pass 0 (or None) to read from the
    /// start. A never-posted-to board reads as empty, not an error.
    pub fn board_read(
        &mut self,
        board_id: Option<&str>,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
        after_sequence: Option<u64>,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::board::BoardPost>> {
        use crate::board::*;
        let board_id = self.resolve_board_id(board_id, owner, workspace, session_id, false)?;
        let after = after_sequence.unwrap_or(0);
        let limit = limit.clamp(1, 100);
        let mut posts = Vec::new();
        for doc in self.store.source_documents() {
            let f = &doc.filters;
            if f.get(BOARD_DOC_KIND_FILTER).map(String::as_str) != Some(BOARD_POST_KIND) {
                continue;
            }
            if f.get(BOARD_ID_FILTER).map(String::as_str) != Some(board_id.as_str()) {
                continue;
            }
            let sequence: u64 = f
                .get(BOARD_SEQUENCE_FILTER)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if sequence <= after {
                continue;
            }
            posts.push(BoardPost {
                post_id: doc.doc_id.clone(),
                board_id: board_id.clone(),
                author_agent_id: f
                    .get(BOARD_AUTHOR_FILTER)
                    .cloned()
                    .unwrap_or_default(),
                provider: f.get(BOARD_PROVIDER_FILTER).cloned().unwrap_or_default(),
                content: doc.content.clone(),
                sequence,
                created_at: doc.timestamp.clone().unwrap_or_default(),
            });
        }
        posts.sort_by_key(|p| p.sequence);
        posts.truncate(limit);
        Ok(posts)
    }

    /// Get one post. `board_id` may be omitted (or be the `"default"`
    /// alias) for the current session's board. Returns `None` when the
    /// post doesn't exist **or** belongs to a different board: a post ID
    /// from another board cannot be read through the wrong board.
    pub fn board_get(
        &mut self,
        board_id: Option<&str>,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
        post_id: &str,
    ) -> anyhow::Result<Option<crate::board::BoardPost>> {
        use crate::board::*;
        let board_id = self.resolve_board_id(board_id, owner, workspace, session_id, false)?;
        let doc = self
            .store
            .source_documents()
            .into_iter()
            .find(|d| d.doc_id == post_id);
        match doc {
            Some(d) => {
                let f = &d.filters;
                if f.get(BOARD_DOC_KIND_FILTER).map(String::as_str) != Some(BOARD_POST_KIND) {
                    return Ok(None);
                }
                // Cross-board isolation: the post's board must match.
                if f.get(BOARD_ID_FILTER).map(String::as_str) != Some(board_id.as_str()) {
                    return Ok(None);
                }
                let sequence: u64 = f
                    .get(BOARD_SEQUENCE_FILTER)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                Ok(Some(BoardPost {
                    post_id: d.doc_id.clone(),
                    board_id: board_id.clone(),
                    author_agent_id: f
                        .get(BOARD_AUTHOR_FILTER)
                        .cloned()
                        .unwrap_or_default(),
                    provider: f.get(BOARD_PROVIDER_FILTER).cloned().unwrap_or_default(),
                    content: d.content.clone(),
                    sequence,
                    created_at: d.timestamp.clone().unwrap_or_default(),
                }))
            }
            None => Ok(None),
        }
    }

    /// Search a board's posts. `board_id` may be omitted (or be the
    /// `"default"` alias) for the current session's board. Scoped to the
    /// resolved board: results never leak posts from other boards. A
    /// never-posted-to board searches as empty, not an error.
    pub fn board_search(
        &mut self,
        board_id: Option<&str>,
        owner: &str,
        workspace: &str,
        session_id: Option<&str>,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::board::BoardPost>> {
        use crate::board::*;
        let board_id = self.resolve_board_id(board_id, owner, workspace, session_id, false)?;
        let mut filters = BTreeMap::new();
        filters.insert(
            BOARD_DOC_KIND_FILTER.to_string(),
            BOARD_POST_KIND.to_string(),
        );
        filters.insert(BOARD_ID_FILTER.to_string(), board_id.clone());
        // Scope to the board's posts; no user filter (board posts are shared
        // within the board's owner/workspace scope).
        let results = self.search_with_filters(
            query,
            /* scope = */ board_id.as_str(),
            /* session_id = */ None,
            top_k.clamp(1, 50),
            &filters,
        )?;
        let mut posts = Vec::new();
        for r in results {
            // search_with_filters already applied the board_id filter, but
            // verify again: never leak across boards.
            let doc = self
                .store
                .source_document_by_id(&r.doc_id)
                .filter(|d| {
                    d.filters.get(BOARD_ID_FILTER).map(String::as_str) == Some(board_id.as_str())
                });
            let Some(d) = doc else { continue };
            let f = &d.filters;
            let sequence: u64 = f
                .get(BOARD_SEQUENCE_FILTER)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            posts.push(BoardPost {
                post_id: d.doc_id.clone(),
                board_id: board_id.clone(),
                author_agent_id: f
                    .get(BOARD_AUTHOR_FILTER)
                    .cloned()
                    .unwrap_or_default(),
                provider: f.get(BOARD_PROVIDER_FILTER).cloned().unwrap_or_default(),
                content: d.content.clone(),
                sequence,
                created_at: d.timestamp.clone().unwrap_or_default(),
            });
        }
        Ok(posts)
    }

    /// Remove one document by id, returning the removed document if present.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn remove(&mut self, doc_id: &str) -> Option<crate::SourceDocument> {
        self.store.remove(doc_id)
    }

    /// Look up a raw index record by document id, for response formatting in
    /// integration read paths.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn record_by_id(&self, doc_id: &str) -> Option<&crate::index::DocRecord> {
        self.store.record_by_id(doc_id)
    }

    /// All source documents, for migration and sync iteration.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn source_documents(&self) -> Vec<&crate::SourceDocument> {
        self.store.source_documents()
    }

    /// Stateless plain query for integration read paths (recall).
    /// Runs the service query pipeline with no session: identical to the raw
    /// store's `query`, with no conversation state read or recorded. Hook
    /// retrieval uses [`MemoryService::observe_plain_query`] instead so hook
    /// turns contribute session state.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn query_plain(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        self.search_with_filters(query, "integration", None, top_k, &BTreeMap::new())
    }

    /// Record the session most recently seen active for `provider` in this
    /// workspace. Hooks call this on every event carrying a session id, and
    /// the MCP search dispatch refreshes it when it resolves a session; the
    /// dispatch reads it back as the default session when the caller did not
    /// pass `session_id` explicitly. Fail-open: the write never fails.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn note_active_session(&self, provider: &str, session_id: &str) {
        if let Ok(store) = self.conversation_states.lock() {
            store.note_active_session(provider, session_id);
        }
    }

    /// The session most recently marked active for `provider`, or `None`
    /// when there is none, it is unreadable, or it is stale. Never fails:
    /// every problem degrades to stateless search.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn current_session_id(&self, provider: &str) -> Option<String> {
        self.conversation_states
            .lock()
            .ok()?
            .current_session_id(provider)
    }

    /// Plain query that records the turn into the session's conversation
    /// state without rewriting it. Hook retrieval uses this: hooks inject
    /// background context rather than answering a conversational turn, so
    /// they contribute state (what was asked, what was returned, entity
    /// mentions) for later follow-up resolution without consuming it.
    /// A blank session id degrades to stateless; hooks fail open and never
    /// reject.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn observe_plain_query(
        &mut self,
        query: &str,
        scope: &str,
        session_id: Option<&str>,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        let session_id = session_id.filter(|id| !id.trim().is_empty());
        let prepared = PreparedQuery::new(query);
        let results = self
            .store
            .query_prepared(&prepared, top_k, &BTreeMap::new())?;
        observe_session_search(
            &self.conversation_states,
            scope,
            session_id,
            query,
            &prepared,
            &results,
        );
        Ok(results)
    }

    /// Raw record access for tests asserting on capture idempotency and
    /// document attribution. Production code uses the typed service methods.
    #[cfg(test)]
    pub(crate) fn records(&self) -> Vec<&crate::index::DocRecord> {
        self.store.records()
    }

    /// Index-structure access for tests asserting on segment layout.
    #[cfg(test)]
    pub(crate) fn memory_index_snapshot(&self) -> Option<&crate::pipeline::MemoryIndexSnapshot> {
        self.store.memory_index_snapshot()
    }

    /// Preferred-document selection for hook retrieval: one document per
    /// `session_id`, preferring `session-summary` over `outcome` over
    /// `checkpoint`, ties broken by timestamp. Ranked hits are deduplicated
    /// by session and rewritten to the preferred record's identity.
    /// This is the shared implementation of the logic previously duplicated
    /// in the Claude Code and Codex hook modules.
    #[cfg(any(feature = "claude-code", feature = "codex"))]
    pub(crate) fn select_session_documents(
        &self,
        results: Vec<crate::SearchResult>,
        limit: usize,
    ) -> Vec<crate::SearchResult> {
        use std::collections::{HashMap, HashSet};
        fn document_type_priority(document_type: &str) -> u8 {
            match document_type {
                "session-summary" => 3,
                "outcome" => 2,
                "checkpoint" => 1,
                _ => 0,
            }
        }
        let mut preferred = HashMap::<String, (u8, &crate::index::DocRecord)>::new();
        for record in self.store.records() {
            let Some(session) = record.filters.get("session_id") else {
                continue;
            };
            let priority = document_type_priority(
                record
                    .filters
                    .get("document_type")
                    .map(String::as_str)
                    .unwrap_or_default(),
            );
            match preferred.get(session) {
                Some((current_priority, current))
                    if (*current_priority, current.timestamp.as_deref())
                        >= (priority, record.timestamp.as_deref()) => {}
                _ => {
                    preferred.insert(session.clone(), (priority, record));
                }
            }
        }

        let mut seen_sessions = HashSet::new();
        let mut selected = Vec::new();
        for mut result in results {
            let Some(record) = self.store.record_by_id(&result.doc_id) else {
                continue;
            };
            let session = record
                .filters
                .get("session_id")
                .cloned()
                .or_else(|| result.group_id.clone())
                .unwrap_or_else(|| result.doc_id.clone());
            if !seen_sessions.insert(session.clone()) {
                continue;
            }
            if let Some((_, preferred_record)) = preferred.get(&session) {
                result.doc_id = preferred_record.doc_id.clone();
                result.source = preferred_record.source.clone();
                result.group_id = preferred_record.group_id.clone();
            }
            selected.push(result);
            if selected.len() == limit {
                break;
            }
        }
        selected
    }

    /// Number of source documents in the index, for the MCP info tool.
    pub(crate) fn docs_count(&self) -> usize {
        self.store.source_documents().len()
    }

    /// Format search results as the MCP search payload.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn search_results_payload(
        &self,
        results: Vec<crate::SearchResult>,
    ) -> serde_json::Value {
        crate::integrations::mcp_tools::search_results(self, results)
    }

    /// Format the memory list as the MCP list_memories payload.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn list_memories_payload(&self, limit: usize) -> serde_json::Value {
        crate::integrations::mcp_tools::list_memories(self, limit)
    }

    /// Look up a source document by id, for tests that verify indexed
    /// content through the service API.
    pub(crate) fn source_document_by_id(&self, doc_id: &str) -> Option<&crate::SourceDocument> {
        self.store.source_document_by_id(doc_id)
    }
    /// Shape raw search results into the API response type. The HTTP server
    /// and language bindings call this after [`MemoryService::search`];
    /// agent-facing callers use the shared `search_results` formatter instead.
    pub fn format_search_response(&self, results: Vec<crate::SearchResult>) -> SearchResponse {
        let data = results
            .into_iter()
            .filter_map(|result| {
                self.store
                    .source_document_by_id(&result.doc_id)
                    .filter(|doc| self.is_visible(doc))
                    .map(|doc| SearchMemory {
                        id: result.doc_id,
                        content: doc.content.clone(),
                        score: result.score,
                        created_at: doc.timestamp.clone(),
                        role: doc.author_agent.clone(),
                        user_id: doc.filters.get(USER_FILTER).cloned(),
                        session_id: doc.group_id.clone(),
                        relation_evidence: result.relation_evidence.clone(),
                    })
            })
            .collect();
        SearchResponse { data }
    }

    /// Delete one memory, returning false for an already absent or foreign id.
    pub fn delete(&mut self, user_id: &str, doc_id: &str) -> anyhow::Result<bool> {
        validate_identifier(user_id, "user_id")?;
        validate_identifier(doc_id, "doc_id")?;
        let owned = self
            .store
            .source_document_by_id(doc_id)
            .is_some_and(|doc| doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id));
        if !owned {
            return Ok(false);
        }
        self.drain_enrichment_inbox();
        self.drop_enrichment_for_doc(doc_id);
        self.store.remove(doc_id);
        self.store.refresh()?;
        Ok(true)
    }

    /// Mark an existing memory as replacing another memory in the same user scope.
    pub fn supersede(
        &mut self,
        user_id: &str,
        replacement_id: &str,
        old_id: &str,
    ) -> anyhow::Result<bool> {
        validate_identifier(user_id, "user_id")?;
        validate_identifier(replacement_id, "replacement_id")?;
        validate_identifier(old_id, "old_id")?;
        let Some(mut replacement) = self.store.source_document_by_id(replacement_id).cloned()
        else {
            return Ok(false);
        };
        if replacement.filters.get(USER_FILTER).map(String::as_str) != Some(user_id) {
            return Ok(false);
        }
        replacement
            .filters
            .insert("supersedes_id".to_string(), old_id.to_string());
        self.superseded_ids
            .insert((user_id.to_string(), old_id.to_string()));
        // Metadata-only touch: the content is unchanged, so the store is
        // written directly without re-queuing key-phrase enrichment.
        self.store.upsert(replacement);
        self.store.refresh()?;
        Ok(true)
    }

    /// Remove all expired memories owned by a user.
    pub fn expire(&mut self, user_id: &str) -> anyhow::Result<usize> {
        validate_identifier(user_id, "user_id")?;
        let now_ms = unix_time_ms();
        let ids = self
            .store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                let owned = doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id);
                let expired = doc
                    .filters
                    .get("expires_at_ms")
                    .and_then(|v| v.parse::<u64>().ok())
                    .is_some_and(|expires_at| expires_at <= now_ms);
                (owned && expired).then(|| doc.doc_id.clone())
            })
            .collect::<Vec<_>>();
        for id in &ids {
            self.store.remove(id);
        }
        if !ids.is_empty() {
            self.store.refresh()?;
        }
        Ok(ids.len())
    }
}

fn default_list_limit() -> usize {
    100
}

fn owns_memory(doc: &SourceDocument, user_id: &str) -> bool {
    doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id)
}

fn memory_record(doc: &SourceDocument) -> MemoryRecord {
    let mut metadata = doc.filters.clone();
    metadata.retain(|key, _| {
        !matches!(
            key.as_str(),
            USER_FILTER | "request_id" | "request_fingerprint" | "expires_at_ms" | "supersedes_id"
        )
    });
    MemoryRecord {
        id: doc.doc_id.clone(),
        content: doc.content.clone(),
        role: doc.author_agent.clone(),
        user_id: doc.filters.get(USER_FILTER).cloned().unwrap_or_default(),
        session_id: doc.group_id.clone(),
        created_at: doc.timestamp.clone(),
        expires_at_ms: doc
            .filters
            .get("expires_at_ms")
            .and_then(|value| value.parse().ok()),
        supersedes_id: doc.filters.get("supersedes_id").cloned(),
        metadata,
    }
}

/// Prepare a query against an optional conversation session. Without a
/// session id this is exactly `PreparedQuery::new(query)`; with one, the
/// query is rewritten against the session's bounded prior state (follow-up
/// resolution and temporal-anchor seeding) before analysis.
///
/// `scope` namespaces the state (a user id for the API path, a provider
/// name for the MCP path); `session_id` is the conversation key within it.
/// Shared by `MemoryService` and the MCP dispatchers so both paths get the
/// same stateful behavior.
pub(crate) fn prepare_session_query(
    states: &std::sync::Mutex<crate::conversation_state::ConversationStateStore>,
    scope: &str,
    session_id: Option<&str>,
    query: &str,
) -> PreparedQuery {
    let Some(session_id) = session_id else {
        return PreparedQuery::new(query);
    };
    let now_ms = unix_time_ms();
    let rewritten = {
        let mut states = states.lock().expect("conversation state lock poisoned");
        let state = states.get(scope, session_id, now_ms);
        crate::session_prepare::resolve_follow_up(query, state)
    };
    PreparedQuery::new(&rewritten)
}

/// Record a completed search turn in the session state. No-op without a
/// session id. The original query text is recorded (not the rewritten one),
/// along with the retrieved document ids, entity mentions from the analysis,
/// and the temporal anchor when the query set one explicitly.
pub(crate) fn observe_session_search(
    states: &std::sync::Mutex<crate::conversation_state::ConversationStateStore>,
    scope: &str,
    session_id: Option<&str>,
    original_query: &str,
    prepared: &PreparedQuery,
    results: &[crate::SearchResult],
) {
    let Some(session_id) = session_id else {
        return;
    };
    let analysis = prepared.analysis();
    let entities: Vec<String> = analysis
        .spans
        .iter()
        .map(|span| span.text.clone())
        .collect();
    let temporal_anchor = analysis
        .temporal
        .as_ref()
        .and_then(|temporal| temporal.resolved_at.clone());
    let doc_ids: Vec<String> = results.iter().map(|r| r.doc_id.clone()).collect();
    let now_ms = unix_time_ms();
    states
        .lock()
        .expect("conversation state lock poisoned")
        .observe(
            scope,
            session_id,
            original_query,
            &doc_ids,
            &entities,
            temporal_anchor,
            now_ms,
        );
}

/// Entity mentions resolved from the session's recent turns, most recent
/// first. Feeds the conversational rerank's speaker-match feature.
fn session_resolved_entities(
    states: &Mutex<crate::conversation_state::ConversationStateStore>,
    scope: &str,
    session_id: &str,
) -> Vec<String> {
    let now_ms = unix_time_ms();
    states
        .lock()
        .expect("conversation state lock poisoned")
        .get(scope, session_id, now_ms)
        .map(|state| state.resolved_entities.clone())
        .unwrap_or_default()
}

/// True when the conversational rerank should fire for this query: a
/// follow-up inside a live session, with the feature enabled.
fn should_conversational_rerank(enabled: bool, session_id: Option<&str>, query: &str) -> bool {
    enabled && session_id.is_some() && is_follow_up(query)
}

fn request_fingerprint(session_id: &str, messages: &[Message]) -> anyhow::Result<String> {
    let canonical = serde_json::to_vec(&(session_id, messages))?;
    let digest = Sha256::digest(canonical);
    Ok(format!("v2:{digest:x}"))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

/// Identifiers reach the index as filter values and as `memory://` source URIs,
/// so they must stay short, single-line, and free of control characters.
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_MESSAGES_PER_REQUEST: usize = 1024;
const MAX_MESSAGE_CONTENT_BYTES: usize = 1024 * 1024;

fn validate_identifier(value: &str, name: &str) -> anyhow::Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        anyhow::bail!("{name} must be at most {MAX_IDENTIFIER_BYTES} bytes");
    }
    if value.chars().any(|character| character.is_control()) {
        anyhow::bail!("{name} must not contain control characters");
    }
    Ok(())
}

fn validate_message(message: &Message, index: usize) -> anyhow::Result<()> {
    if message.content.trim().is_empty() {
        anyhow::bail!("messages[{index}].content must not be empty");
    }
    if message.content.len() > MAX_MESSAGE_CONTENT_BYTES {
        anyhow::bail!("messages[{index}].content exceeds {MAX_MESSAGE_CONTENT_BYTES} bytes");
    }
    if message.role != "user" && message.role != "assistant" {
        anyhow::bail!("messages[{index}].role must be user or assistant");
    }
    if let Some(supersedes_id) = &message.supersedes_id {
        validate_identifier(supersedes_id, "supersedes_id")?;
    }
    if let Some(millis) = message.timestamp {
        timestamp_to_rfc3339(millis, "timestamp")
            .map_err(|error| anyhow::anyhow!("messages[{index}].{error}"))?;
    }
    Ok(())
}

fn timestamp_to_rfc3339(millis: i64, field: &str) -> anyhow::Result<String> {
    Utc.timestamp_millis_opt(millis)
        .single()
        .map(|date| date.to_rfc3339())
        .ok_or_else(|| anyhow::anyhow!("{field} must be a valid Unix timestamp in milliseconds"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelineOptions;

    #[test]
    fn lang_auto_selects_spacy_model_from_turns() {
        // P2: `"lang": "auto"` must behave like an omitted language — the
        // model follows the document turns' script, not English.
        let ko_turns = vec![RelationTurn {
            speaker: "지민".to_string(),
            text: "지민은 서울에서 일합니다".to_string(),
            session_id: "s1".to_string(),
            turn_idx: 0,
            doc_id: "d1".to_string(),
            session_date: None,
        }];
        assert_eq!(
            spacy_model_for_request(Some(Lang::Auto), &ko_turns),
            "ko_core_news_sm"
        );
        assert_eq!(spacy_model_for_request(None, &ko_turns), "ko_core_news_sm");

        let zh_turns = vec![RelationTurn {
            speaker: "小明".to_string(),
            text: "小明在北京工作".to_string(),
            session_id: "s1".to_string(),
            turn_idx: 0,
            doc_id: "d1".to_string(),
            session_date: None,
        }];
        assert_eq!(
            spacy_model_for_request(Some(Lang::Auto), &zh_turns),
            "zh_core_web_sm"
        );

        let es_turns = vec![RelationTurn {
            speaker: "María".to_string(),
            text: "¿Dónde está la biblioteca de Madrid? Fui ayer por la mañana.".to_string(),
            session_id: "s1".to_string(),
            turn_idx: 0,
            doc_id: "d1".to_string(),
            session_date: None,
        }];
        assert_eq!(
            spacy_model_for_request(Some(Lang::Auto), &es_turns),
            "es_core_news_sm"
        );
        assert_eq!(spacy_model_for_request(None, &es_turns), "es_core_news_sm");

        // Explicit languages still pin their model.
        assert_eq!(
            spacy_model_for_request(Some(Lang::Ko), &ko_turns),
            "ko_core_news_sm"
        );
        assert_eq!(
            spacy_model_for_request(Some(Lang::En), &ko_turns),
            "en_core_web_sm"
        );
    }

    fn service() -> MemoryService {
        MemoryService::in_memory(PipelineOptions::default())
    }

    /// Temp dir with symlinks resolved (on macOS `TMPDIR` lives under
    /// `/var`, which is a symlink to `/private/var`; comparing or
    /// reopening paths through the unresolved prefix breaks).
    fn canonical_temp_dir(name: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        base.join(format!("lint-ai-{name}-{}", std::process::id()))
    }

    fn kind_verdict(kinds: &[(&str, &str)]) -> crate::behood_query::KindVerdict {
        crate::behood_query::KindVerdict {
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

    fn bare_search_result(doc_id: &str, score: f32) -> crate::SearchResult {
        crate::SearchResult {
            doc_id: doc_id.to_string(),
            source: "test".to_string(),
            group_id: None,
            score,
            score_breakdown: Default::default(),
            matched_entities: vec![],
            matched_terms: vec![],
            probable_topic: None,
            doc_type_guess: None,
            semantic_status: None,
            superseded_by: None,
            relation_confidence: None,
            relation_evidence: vec![],
        }
    }

    #[test]
    fn activity_venue_match_predicate() {
        // mem-14: "eat out" + place-kind "restaurant" -> match.
        assert!(activity_venue_match(
            "eat out",
            &kind_verdict(&[("The user's favorite restaurant", "place")]),
        ));
        // Case/whitespace-insensitive on the activity side.
        assert!(activity_venue_match(
            "  Eat Out ",
            &kind_verdict(&[("a cafe", "place")]),
        ));
        // "park" is a run venue, not an eat-out venue -> no match.
        assert!(!activity_venue_match(
            "eat out",
            &kind_verdict(&[("Golden Gate Park", "place")]),
        ));
        // Multi-word venue: "coffee shop" matches as a phrase.
        assert!(activity_venue_match(
            "drink coffee",
            &kind_verdict(&[("the local coffee shop", "place")]),
        ));
        // Whole-word control: "parked" must not match "park".
        assert!(!activity_venue_match(
            "run",
            &kind_verdict(&[("parked cars", "place")]),
        ));
        // Only place-kind descriptors count: a venue word under any
        // other kind is not a venue.
        assert!(!activity_venue_match(
            "eat out",
            &kind_verdict(&[("restaurant", "thing")]),
        ));
        // Activity not in the admitted table -> no match, never a guess.
        assert!(!activity_venue_match(
            "skydive",
            &kind_verdict(&[("the airport", "place")]),
        ));
        // Empty activity phrase -> no match.
        assert!(!activity_venue_match(
            "",
            &kind_verdict(&[("a restaurant", "place")]),
        ));
    }

    #[test]
    fn activity_venue_boost_adds_resorts_and_never_removes() {
        // Lexical order: park-run (10) > restaurant (9). Boost restaurant.
        let mut results = vec![
            bare_search_result("parkrun", 10.0),
            bare_search_result("restaurant", 9.0),
        ];
        let n = boost_result_indices(&mut results, &[1], ACTIVITY_VENUE_BOOST, |b, a| {
            b.activity_venue_boost += a
        });
        assert_eq!(n, 1);
        assert_eq!(results.len(), 2, "boost must never remove results");
        assert_eq!(results[0].doc_id, "restaurant");
        assert_eq!(results[0].score, 9.0 + ACTIVITY_VENUE_BOOST);
        assert_eq!(
            results[0].score_breakdown.activity_venue_boost,
            ACTIVITY_VENUE_BOOST
        );
        // Untouched hits keep their score and carry no venue boost.
        assert_eq!(results[1].doc_id, "parkrun");
        assert_eq!(results[1].score, 10.0);
        assert_eq!(results[1].score_breakdown.activity_venue_boost, 0.0);
    }

    #[test]
    fn add_is_immediately_searchable() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "I prefer dark mode in every editor".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "dark mode editor".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 100,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("dark mode"));
    }

    #[test]
    fn korean_particle_mismatch_still_retrieves() {
        // Smoke test: the document inflects "학교" as "학교에"/"학교에서";
        // the bare-stem query "학교" must still retrieve it via the
        // particle-stripped index tokens.
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "ko-1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "김철수는 학교에 갔다. 학교에서 친구를 만났다.".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-ko".into(),
                session_id: "session-ko".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "학교".into(),
                options: None,
                user_id: "user-ko".into(),
                top_k: 100,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
            })
            .unwrap();
        assert_eq!(response.data.len(), 1, "bare stem query should retrieve the doc");
        assert!(response.data[0].content.contains("학교에"));

        // Inflected query form also retrieves.
        let response = service
            .search(SearchRequest {
                query: "학교에서".into(),
                options: None,
                user_id: "user-ko".into(),
                top_k: 100,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
            })
            .unwrap();
        assert_eq!(response.data.len(), 1);
    }

    #[test]
    fn korean_question_retrieves_answer_memory() {
        // End-to-end: a natural Korean question (with interrogative 어디)
        // retrieves the memory holding its answer and ranks it above a
        // distractor that shares no content words.
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "ko-q-target".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "박영희는 부산에서 태어났다. 지금은 서울에 산다.".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-ko".into(),
                session_id: "session-ko".into(),
            })
            .unwrap();
        service
            .add(AddRequest {
                request_id: "ko-q-d1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "김철수는 주말에 공원에서 조깅을 즐긴다.".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-ko".into(),
                session_id: "session-ko".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "박영희는 어디에서 태어났나?".into(),
                options: None,
                user_id: "user-ko".into(),
                top_k: 100,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
            })
            .unwrap();
        assert!(
            !response.data.is_empty(),
            "Korean question should retrieve the answer memory"
        );
        assert!(
            response.data[0].content.contains("부산에서"),
            "target should rank first, got {:?}",
            response.data.iter().map(|r| &r.content).collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_cannot_cross_user_boundaries() {
        let mut service = service();
        for (request_id, user_id, content) in [
            ("request-a", "user-a", "the secret project uses amber"),
            ("request-b", "user-b", "the secret project uses cobalt"),
        ] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: content.into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: user_id.into(),
                    session_id: "session".into(),
                })
                .unwrap();
        }
        let response = service
            .search(SearchRequest {
                query: "secret project color".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 100,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert!(response
            .data
            .iter()
            .all(|memory| memory.content.contains("amber")));
    }

    #[test]
    fn search_with_empty_user_id_skips_ownership_filter() {
        // Provider-scoped callers (MCP adapters) pass no user id: documents
        // without a memory_user_id filter must still be returned. This is the
        // regression test for the Hermes 0-hits finding — an unconditional
        // ownership filter would zero out the adapter's local corpus.
        let mut service = service();
        let mut owned_filters = BTreeMap::new();
        owned_filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        for (doc_id, content, filters) in [
            (
                "owned-doc",
                "The owned ledger records amber transactions.",
                owned_filters,
            ),
            (
                "unowned-doc",
                "The shared ledger records amber transactions.",
                BTreeMap::new(),
            ),
        ] {
            service.store.upsert(SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("hook://{doc_id}"),
                content: content.to_string(),
                concept: doc_id.to_string(),
                group_id: Some(doc_id.to_string()),
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: content.len(),
                author_agent: None,
                filters,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
        service.store.refresh().unwrap();

        let response = service
            .search(SearchRequest {
                query: "amber ledger transactions".into(),
                options: None,
                user_id: "".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
            })
            .unwrap();
        let ids: Vec<&str> = response
            .data
            .iter()
            .map(|memory| memory.id.as_str())
            .collect();
        assert!(
            ids.contains(&"owned-doc"),
            "empty user_id must not hide owned docs, got {ids:?}"
        );
        assert!(
            ids.contains(&"unowned-doc"),
            "empty user_id must not hide unowned docs, got {ids:?}"
        );
    }

    #[test]
    fn search_with_user_id_still_enforces_ownership_filter() {
        // The server path is unaffected: a supplied user id filters exactly
        // as before.
        let mut service = service();
        let mut owned_filters = BTreeMap::new();
        owned_filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        for (doc_id, content, filters) in [
            (
                "owned-doc",
                "The owned ledger records amber transactions.",
                owned_filters,
            ),
            (
                "unowned-doc",
                "The shared ledger records amber transactions.",
                BTreeMap::new(),
            ),
        ] {
            service.store.upsert(SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("hook://{doc_id}"),
                content: content.to_string(),
                concept: doc_id.to_string(),
                group_id: Some(doc_id.to_string()),
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: content.len(),
                author_agent: None,
                filters,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
        service.store.refresh().unwrap();

        let response = service
            .search(SearchRequest {
                query: "amber ledger transactions".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
            })
            .unwrap();
        let ids: Vec<&str> = response
            .data
            .iter()
            .map(|memory| memory.id.as_str())
            .collect();
        assert!(
            ids.contains(&"owned-doc"),
            "supplied user_id must return owned docs, got {ids:?}"
        );
        assert!(
            !ids.contains(&"unowned-doc"),
            "supplied user_id must hide unowned docs, got {ids:?}"
        );
    }

    #[test]
    fn identical_request_ids_are_isolated_between_users() {
        let mut service = service();
        for (user_id, content) in [("user-a", "alpha secret"), ("user-b", "beta secret")] {
            service
                .add(AddRequest {
                    request_id: "same-request".into(),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: content.into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: user_id.into(),
                    session_id: "session".into(),
                })
                .unwrap();
        }
        let a = service
            .search(SearchRequest {
                query: "secret".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        let b = service
            .search(SearchRequest {
                query: "secret".into(),
                options: None,
                user_id: "user-b".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert_eq!(a.data.len(), 1);
        assert_eq!(b.data.len(), 1);
        assert!(a.data[0].content.contains("alpha"));
        assert!(b.data[0].content.contains("beta"));
    }

    #[test]
    fn request_fingerprint_is_a_digest_and_includes_session_id() {
        let mut service = service();
        let messages = vec![Message {
            role: "user".into(),
            timestamp: None,
            content: "sensitive preference".into(),
            expires_at_ms: None,
            supersedes_id: None,
        }];
        service
            .add(AddRequest {
                request_id: "same-request".into(),
                messages: messages.clone(),
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();

        let persisted = service.store.source_documents()[0]
            .filters
            .get("request_fingerprint")
            .unwrap();
        assert!(persisted.starts_with("v2:"));
        assert!(!persisted.contains("sensitive preference"));

        let error = service
            .add(AddRequest {
                request_id: "same-request".into(),
                messages,
                user_id: "user-a".into(),
                session_id: "session-b".into(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("different content"));
    }

    #[test]
    fn invalid_later_message_does_not_partially_add_request() {
        let mut service = service();
        let error = service
            .add(AddRequest {
                request_id: "atomic-request".into(),
                messages: vec![
                    Message {
                        role: "user".into(),
                        timestamp: None,
                        content: "first valid message".into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    },
                    Message {
                        role: "system".into(),
                        timestamp: None,
                        content: "invalid role".into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    },
                ],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("role"));
        assert!(service.store.source_documents().is_empty());
    }

    #[test]
    fn search_reflects_writes_with_no_republish_step() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "first".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "project codename zephyr".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();

        service
            .add(AddRequest {
                request_id: "second".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "database codename quartz".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-b".into(),
            })
            .unwrap();

        // One service, one search entry point: reads see the writes directly,
        // with no snapshot re-publish step in between.
        let response = service
            .search(SearchRequest {
                query: "database codename quartz".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert!(response
            .data
            .iter()
            .any(|memory| memory.content.contains("quartz")));
    }

    #[test]
    fn expired_memories_are_not_searchable_and_can_be_deleted_idempotently() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-expired".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "temporary migration decision".into(),
                    expires_at_ms: Some(1),
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "temporary migration".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert!(response.data.is_empty());
        assert!(!service.delete("user-a", "missing").unwrap());
    }

    #[test]
    fn superseded_memory_is_hidden_but_replacement_remains_searchable() {
        let mut service = service();
        let old_id = crate::stable_doc_id_from_source("user-a:old:0");
        for (request_id, content, supersedes_id) in [
            ("old", "old deployment decision", None),
            ("new", "new deployment decision", Some(old_id.as_str())),
        ] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: content.into(),
                        expires_at_ms: None,
                        supersedes_id: supersedes_id.map(str::to_string),
                    }],
                    user_id: "user-a".into(),
                    session_id: "session-a".into(),
                })
                .unwrap();
        }
        let response = service
            .search(SearchRequest {
                query: "deployment decision".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("new deployment"));
    }

    #[test]
    fn supersession_is_scoped_to_the_requesting_user() {
        let mut service = service();
        let old_id = crate::stable_doc_id_from_source("old:0");
        service
            .add(AddRequest {
                request_id: "old".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "shared deployment decision".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        service
            .add(AddRequest {
                request_id: "replacement".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "replacement from another user".into(),
                    expires_at_ms: None,
                    supersedes_id: Some(old_id),
                }],
                user_id: "user-b".into(),
                session_id: "session-b".into(),
            })
            .unwrap();

        let response = service
            .search(SearchRequest {
                query: "shared deployment decision".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("shared deployment"));
    }

    #[test]
    fn list_get_and_update_preserve_identity_and_scope() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "old preference".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let listed = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: Some("session-a".into()),
                limit: 10,
                cursor: None,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(listed.data.len(), 1);
        let id = listed.data[0].id.clone();
        assert!(service
            .get(GetRequest {
                user_id: "user-b".into(),
                memory_id: id.clone(),
                include_inactive: false,
            })
            .unwrap()
            .is_none());

        let updated = service
            .update(UpdateRequest {
                user_id: "user-a".into(),
                memory_id: id.clone(),
                content: "new preference".into(),
                role: None,
                timestamp: None,
                expires_at_ms: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(updated.id, id);
        assert!(updated.content.contains("new preference"));
        assert_eq!(updated.session_id.as_deref(), Some("session-a"));

        let invalid_update = service.update(UpdateRequest {
            user_id: "user-a".into(),
            memory_id: id.clone(),
            content: "should not be applied".into(),
            role: None,
            timestamp: Some(i64::MAX),
            expires_at_ms: None,
        });
        assert!(invalid_update.is_err());
        let unchanged = service
            .get(GetRequest {
                user_id: "user-a".into(),
                memory_id: id,
                include_inactive: false,
            })
            .unwrap()
            .unwrap();
        assert!(unchanged.content.contains("new preference"));
    }

    #[test]
    fn list_cursor_is_deterministic_and_excludes_inactive_memories() {
        let mut service = service();
        for request_id in ["a", "b"] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "assistant".into(),
                        timestamp: None,
                        content: format!("memory {request_id}"),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: "user-a".into(),
                    session_id: "session-a".into(),
                })
                .unwrap();
        }
        let first = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: None,
                limit: 1,
                cursor: None,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(first.data.len(), 1);
        let second = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: None,
                limit: 1,
                cursor: first.next_cursor,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(second.data.len(), 1);
        assert_ne!(first.data[0].id, second.data[0].id);
    }

    #[test]
    fn group_id_is_an_exact_filter_only_when_explicitly_requested() {
        use std::collections::BTreeMap;

        let mut service = service();
        // Two sessions for the same user; group_id comes from session_id.
        for (session, text) in [
            (
                "session-one",
                "the quartz database uses append-only segments",
            ),
            (
                "session-two",
                "the quartz database uses append-only segments",
            ),
        ] {
            service
                .add(AddRequest {
                    request_id: format!("req-{session}"),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: text.into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: "user-a".into(),
                    session_id: session.to_string(),
                })
                .unwrap();
        }

        // No group filter: both sessions' documents are searchable. A
        // session_id on the *search* is a state key only and must not
        // implicitly scope the corpus.
        let prepared = PreparedQuery::new("quartz database segments");
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        let unfiltered = service
            .store
            .query_prepared(&prepared, 10, &filters)
            .unwrap();
        assert_eq!(unfiltered.len(), 2);

        // Explicit group_id filter: only the requested session's documents.
        filters.insert("group_id".to_string(), "session-one".to_string());
        let filtered = service
            .store
            .query_prepared(&prepared, 10, &filters)
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].group_id.as_deref(), Some("session-one"));
    }

    #[test]
    fn session_temporal_anchor_carries_into_follow_up_retrieval() {
        use chrono::Duration;

        let options = PipelineOptions {
            memory_index_layout: crate::pipeline::MemoryIndexLayout::Segmented {
                query_top_n: 8,
                routing_strategy: crate::segments::SegmentRoutingStrategy::SparseOverlap,
            },
            ..PipelineOptions::default()
        };
        let mut service = MemoryService::new(IndexStore::in_memory(options));
        let today = Utc::now().date_naive();
        let recent_date = (today - Duration::days(1)).format("%Y-%m-%d").to_string();
        let old_date = (today - Duration::days(30)).format("%Y-%m-%d").to_string();

        for (doc_id, group_id, timestamp, content) in [
            (
                "recent-doc",
                "recent",
                recent_date.as_str(),
                "Budget planning notes: allocate 20 percent to research.",
            ),
            (
                "old-doc",
                "old",
                old_date.as_str(),
                "Budget planning notes: allocate 5 percent to research.",
            ),
        ] {
            let mut doc_filters = BTreeMap::new();
            doc_filters.insert(USER_FILTER.to_string(), "user-a".to_string());
            service.store.upsert(SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("artifact://{doc_id}"),
                content: content.to_string(),
                concept: doc_id.to_string(),
                group_id: Some(group_id.to_string()),
                headings: vec![],
                links: vec![],
                timestamp: Some(timestamp.to_string()),
                doc_length: content.len(),
                author_agent: None,
                filters: doc_filters,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
        service.store.refresh().unwrap();

        // Turn 1 sets a temporal anchor via a relative phrase ("yesterday").
        service
            .search(SearchRequest {
                query: "What did we discuss yesterday?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: Some("s-temporal".into()),
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();

        // Turn 2 is a follow-up: the carried anchor must restrict retrieval
        // to the in-window segment.
        let turn2 = service
            .search(SearchRequest {
                query: "what about the budget?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: Some("s-temporal".into()),
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        let ids: Vec<&str> = turn2.data.iter().map(|memory| memory.id.as_str()).collect();
        assert!(
            ids.contains(&"recent-doc"),
            "follow-up should retrieve in-window doc, got {ids:?}"
        );
        assert!(
            !ids.contains(&"old-doc"),
            "follow-up should not retrieve out-of-window doc, got {ids:?}"
        );

        // Stateless baseline: the same wording without a session gets no
        // carried anchor, so the out-of-window segment is still searchable.
        let baseline = service
            .search(SearchRequest {
                query: "what about the budget?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        let baseline_ids: Vec<&str> = baseline
            .data
            .iter()
            .map(|memory| memory.id.as_str())
            .collect();
        assert!(
            baseline_ids.contains(&"old-doc"),
            "stateless baseline should still see the out-of-window doc, got {baseline_ids:?}"
        );
    }

    fn add_quartz_fixture(service: &mut MemoryService) {
        for (doc_id, content) in [
            (
                "quartz-doc",
                "The Quartz database is a distributed store with strong consistency and tunable replication.",
            ),
            (
                "sourdough-doc",
                "Baking sourdough bread requires patience, a hot oven, and a lively starter.",
            ),
        ] {
            let mut doc_filters = BTreeMap::new();
            doc_filters.insert(USER_FILTER.to_string(), "user-a".to_string());
            service.store.upsert(SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("artifact://{doc_id}"),
                content: content.to_string(),
                concept: doc_id.to_string(),
                group_id: Some(doc_id.to_string()),
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: content.len(),
                author_agent: None,
                filters: doc_filters,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
        service.store.refresh().unwrap();
    }

    fn search(service: &mut MemoryService, query: &str, session_id: Option<&str>) -> SearchResponse {
        service
            .search(SearchRequest {
                query: query.into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: session_id.map(str::to_string),
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap()
    }

    #[test]
    fn search_resolves_follow_up_against_session() {
        let mut service = service();
        add_quartz_fixture(&mut service);

        // Turn 1 establishes the session's entities.
        let first = search(&mut service, "Tell me about the Quartz database", Some("s1"));
        assert!(first.data.iter().any(|m| m.content.contains("Quartz")));

        // A follow-up with no standalone meaning resolves against the session.
        let follow_up = search(&mut service, "what are its limitations?", Some("s1"));
        assert!(
            follow_up.data.iter().any(|m| m.content.contains("Quartz")),
            "follow-up should resolve against session state"
        );

        // Without a session the same query stays stateless and cannot borrow
        // the session's entities.
        let stateless = search(&mut service, "what are its limitations?", None);
        assert!(
            !stateless.data.iter().any(|m| m.content.contains("Quartz")),
            "stateless search must not resolve against session state"
        );
    }

    #[test]
    fn search_rejects_blank_session_id() {
        let mut service = service();
        for session_id in [Some(""), Some("   ")] {
            let error = service
                .search(SearchRequest {
                    query: "hello".into(),
                    options: None,
                    user_id: "user-a".into(),
                    top_k: 10,
                    session_id: session_id.map(str::to_string),
                    scope: None,
                    filters: None,
                lang: None,
})
                .unwrap_err();
            assert!(
                error.to_string().contains("session_id must not be empty"),
                "unexpected error: {error:#}"
            );
        }
    }

    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    #[test]
    fn observe_plain_query_contributes_session_state_for_follow_ups() {
        // Enrichment disabled: this test exercises session state, and the
        // query-time backfill must not shell out to the real extractor here.
        let mut service = MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: false,
            ..PipelineOptions::default()
        });
        add_quartz_fixture(&mut service);

        // Hook-style retrieval: the query is not rewritten, but the turn is
        // recorded into the session state.
        let results = service
            .observe_plain_query(
                "Tell me about the Quartz database",
                "claude",
                Some("hook-session"),
                10,
            )
            .unwrap();
        assert!(results.iter().any(|r| r.doc_id == "quartz-doc"));

        // A later MCP-style search in the same scope/session resolves the
        // follow-up against what the hook observed.
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        let follow_up = service
            .search_with_filters(
                "what are its limitations?",
                "claude",
                Some("hook-session"),
                10,
                &filters,
            )
            .unwrap();
        assert!(
            follow_up.iter().any(|r| r.doc_id == "quartz-doc"),
            "follow-up should resolve against hook-observed state"
        );

        // A blank hook session id degrades to stateless instead of failing.
        let blank = service
            .observe_plain_query(
                "Tell me about the Quartz database",
                "claude",
                Some("  "),
                10,
            )
            .unwrap();
        assert!(blank.iter().any(|r| r.doc_id == "quartz-doc"));
    }

    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    #[test]
    fn at_path_persists_conversation_state_across_instances() {
        let dir = canonical_temp_dir("conv-state");
        std::fs::create_dir_all(&dir).unwrap();
        // Enrichment disabled: this test exercises conversation state, and
        // the query-time backfill must not shell out to the real extractor.
        let options = PipelineOptions {
            key_phrase_enrichment: false,
            ..PipelineOptions::default()
        };
        {
            let mut service = MemoryService::at_path(&dir, options.clone()).unwrap();
            add_quartz_fixture(&mut service);
            service
                .observe_plain_query(
                    "Tell me about the Quartz database",
                    "claude",
                    Some("persist-s1"),
                    10,
                )
                .unwrap();
        }
        // A fresh service against the same root reloads the session state
        // from disk, so the follow-up still resolves.
        let mut reopened = MemoryService::at_path(&dir, options).unwrap();
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        let follow_up = reopened
            .search_with_filters(
                "what are its limitations?",
                "claude",
                Some("persist-s1"),
                10,
                &filters,
            )
            .unwrap();
        assert!(
            follow_up.iter().any(|r| r.doc_id == "quartz-doc"),
            "follow-up should resolve against disk-persisted session state"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    #[test]
    fn compose_segmented_preserves_provider_conversation_state() {
        // Regression: compose_segmented() must not replace the disk-backed
        // conversation-state store with a memory-only one. Hooks write the
        // active-session pointer into the provider memory root; the composed
        // service has to see it, and explicit session state must survive a
        // new MCP process.
        let base = canonical_temp_dir("compose-state");
        let workspace_root = base.join("workspace-memory");
        let provider_root = base.join("memory");
        std::fs::create_dir_all(&workspace_root).unwrap();
        std::fs::create_dir_all(&provider_root).unwrap();
        let options = crate::pipeline::PipelineOptions {
            memory_index_layout: crate::pipeline::MemoryIndexLayout::Segmented {
                query_top_n: 3,
                routing_strategy: crate::segments::SegmentRoutingStrategy::LocalDistinctiveness,
            },
            ..crate::pipeline::PipelineOptions::default()
        };
        {
            let provider = MemoryService::at_path(&provider_root, options.clone()).unwrap();
            // Simulate a hook invocation marking a session active.
            provider.note_active_session("claude", "sess-abc");
        }
        let workspace = MemoryService::at_path(&workspace_root, options.clone()).unwrap();
        let provider = MemoryService::at_path(&provider_root, options.clone()).unwrap();
        let composed = MemoryService::compose_segmented(workspace, Some(provider)).unwrap();
        assert_eq!(
            composed.current_session_id("claude"),
            Some("sess-abc".to_string()),
            "composed service must inherit the provider's conversation-state store"
        );

        // Without provider memory the composed service falls back to the
        // workspace store rather than a memory-only store.
        let workspace = MemoryService::at_path(&workspace_root, options.clone()).unwrap();
        workspace.note_active_session("claude", "sess-ws");
        let composed = MemoryService::compose_segmented(workspace, None).unwrap();
        assert_eq!(
            composed.current_session_id("claude"),
            Some("sess-ws".to_string()),
            "composed service must fall back to the workspace conversation-state store"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    // --- Structured-fact production wiring ---

    use crate::segments::relations::{RawRelation, RelationIndex, RelationTurn};

    fn structured_doc(
        doc_id: &str,
        content: &str,
        user: &str,
        extra_filters: &[(&str, &str)],
    ) -> SourceDocument {
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), user.to_string());
        for (k, v) in extra_filters {
            filters.insert(k.to_string(), v.to_string());
        }
        SourceDocument {
            doc_id: doc_id.to_string(),
            source: format!("test/{doc_id}"),
            content: content.to_string(),
            concept: "test".to_string(),
            group_id: Some("sess-a".to_string()),
            headings: Vec::new(),
            links: Vec::new(),
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
            filters,
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    /// Relation index where Gina and Jon both went to Rome (docs d1, d2).
    fn rome_index() -> RelationIndex {
        let turns = ["Gina", "Jon"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| RelationTurn {
                speaker: s.to_string(),
                text: format!("{s} went to Rome"),
                session_id: "sess-a".to_string(),
                turn_idx: i,
                doc_id: format!("d{}", i + 1),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw = ["Gina", "Jon"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| RawRelation {
                subject: s.to_string(),
                predicate: "go_to".to_string(),
                object: "Rome".to_string(),
                is_place: true,
                object_kind: "place".to_string(),
                is_activity: true,
                session_id: "sess-a".to_string(),
                turn_idx: i,
                doc_id: format!("d{}", i + 1),
                session_date: None,
                evidence: format!("{s}: went to Rome"),
                confidence: 0.9,
                coref: None,
            })
            .collect::<Vec<_>>();
        RelationIndex::build(&turns, &raw)
    }

    fn seeded_cache(
        user_id: &str,
        docs: &[&SourceDocument],
        index: RelationIndex,
    ) -> Mutex<RelationsCache> {
        let cache = Mutex::new(RelationsCache::default());
        {
            let mut guard = cache.lock().unwrap();
            let entry = guard.per_user.entry(user_id.to_string()).or_default();
            entry.fingerprint = relations_fingerprint(docs);
            entry.index = Some(Arc::new(index));
        }
        cache
    }

    /// Relation index where Gina and Jon both went to Paris (docs d5, d6).
    fn paris_index() -> RelationIndex {
        let turns = ["Gina", "Jon"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| RelationTurn {
                speaker: s.to_string(),
                text: format!("{s} went to Paris"),
                session_id: "sess-b".to_string(),
                turn_idx: i,
                doc_id: format!("d{}", i + 5),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw = ["Gina", "Jon"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| RawRelation {
                subject: s.to_string(),
                predicate: "go_to".to_string(),
                object: "Paris".to_string(),
                is_place: true,
                object_kind: "place".to_string(),
                is_activity: true,
                session_id: "sess-b".to_string(),
                turn_idx: i,
                doc_id: format!("d{}", i + 5),
                session_date: None,
                evidence: format!("{s}: went to Paris"),
                confidence: 0.9,
                coref: None,
            })
            .collect::<Vec<_>>();
        RelationIndex::build(&turns, &raw)
    }

    fn fact_request(query: &str, user: &str) -> SearchRequest {
        SearchRequest {
            query: query.to_string(),
            options: None,
            user_id: user.to_string(),
            top_k: 10,
            session_id: None,
            scope: None,
            filters: None,
            lang: None,
        }
    }

    #[test]
    fn structured_results_return_evidence_hits() {
        let docs = vec![
            structured_doc("d1", "Gina: went to Rome", "u1", &[]),
            structured_doc("d2", "Jon: went to Rome", "u1", &[]),
        ];
        let refs: Vec<&SourceDocument> = docs.iter().collect();
        let cache = seeded_cache("u1", &refs, rome_index());
        let request = fact_request("Which city have both Gina and Jon visited?", "u1");
        let results = structured_fact_results(
            &refs,
            &PipelineOptions::default(),
            &cache,
            &HashSet::new(),
            &request,
        );
        let mut ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["d1", "d2"]);
        for r in &results {
            assert!(r.score > 1000.0);
            assert_eq!(r.relation_evidence, vec!["shared relation: Rome"]);
            assert!(r.relation_confidence.is_some());
        }
    }

    #[test]
    fn structured_results_enforce_visibility() {
        let expired = unix_time_ms().saturating_sub(1_000).to_string();
        let docs = vec![
            structured_doc("d1", "Gina: went to Rome", "u1", &[]),
            structured_doc("d2", "Jon: went to Rome", "u1", &[]),
            // Foreign user: must never compose into u1's relation index.
            structured_doc("d3", "Jon: went to Paris", "u2", &[]),
            // Superseded: filtered.
            structured_doc("d4", "Gina: went to Rome", "u1", &[]),
            // Expired: filtered.
            structured_doc(
                "d5",
                "Jon: went to Rome",
                "u1",
                &[("expires_at_ms", expired.as_str())],
            ),
        ];
        // The relation index is composed from u1's visible documents only;
        // d3 (foreign), d4 (superseded) and d5 (expired) contribute nothing.
        let visible: Vec<&SourceDocument> = docs[..2].iter().collect();
        let mut superseded = HashSet::new();
        superseded.insert(("u1".to_string(), "d4".to_string()));
        let request = fact_request("Which city have both Gina and Jon visited?", "u1");
        let cache = seeded_cache("u1", &visible, rome_index());
        let refs: Vec<&SourceDocument> = docs.iter().collect();
        let results = structured_fact_results(
            &refs,
            &PipelineOptions::default(),
            &cache,
            &superseded,
            &request,
        );
        let mut ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec!["d1", "d2"],
            "only the visible, live, owned docs survive"
        );
        for r in &results {
            assert_eq!(r.relation_evidence, vec!["shared relation: Rome"]);
        }
        // A non-fact question never consults the index (no extractor cost).
        let plain = fact_request("Tell me about Rome.", "u1");
        assert!(structured_fact_results(
            &refs,
            &PipelineOptions::default(),
            &cache,
            &HashSet::new(),
            &plain
        )
        .is_empty());
        // The option flag disables the path entirely.
        let mut off = PipelineOptions::default();
        off.structured_fact_retrieval = false;
        assert!(structured_fact_results(&refs, &off, &cache, &HashSet::new(), &request).is_empty());
    }

    #[test]
    fn structured_results_are_scoped_per_user() {
        let u1_docs = vec![
            structured_doc("d1", "Gina: went to Rome", "u1", &[]),
            structured_doc("d2", "Jon: went to Rome", "u1", &[]),
        ];
        let u2_docs = vec![
            structured_doc("d5", "Gina: went to Paris", "u2", &[]),
            structured_doc("d6", "Jon: went to Paris", "u2", &[]),
        ];
        let u1_refs: Vec<&SourceDocument> = u1_docs.iter().collect();
        let u2_refs: Vec<&SourceDocument> = u2_docs.iter().collect();
        let cache = Mutex::new(RelationsCache::default());
        {
            let mut guard = cache.lock().unwrap();
            let e1 = guard.per_user.entry("u1".to_string()).or_default();
            e1.fingerprint = relations_fingerprint(&u1_refs);
            e1.index = Some(Arc::new(rome_index()));
            let e2 = guard.per_user.entry("u2".to_string()).or_default();
            e2.fingerprint = relations_fingerprint(&u2_refs);
            e2.index = Some(Arc::new(paris_index()));
        }
        // Both users' documents reach the retrieval call; each user still
        // only ever consults their own relation index.
        let mut all: Vec<&SourceDocument> = Vec::new();
        all.extend(u1_refs.iter().copied());
        all.extend(u2_refs.iter().copied());
        let opts = PipelineOptions::default();
        let r1 = structured_fact_results(
            &all,
            &opts,
            &cache,
            &HashSet::new(),
            &fact_request("Which city have both Gina and Jon visited?", "u1"),
        );
        let mut ids1: Vec<&str> = r1.iter().map(|r| r.doc_id.as_str()).collect();
        ids1.sort_unstable();
        assert_eq!(ids1, vec!["d1", "d2"]);
        assert!(r1
            .iter()
            .all(|r| r.relation_evidence == vec!["shared relation: Rome"]));
        let r2 = structured_fact_results(
            &all,
            &opts,
            &cache,
            &HashSet::new(),
            &fact_request("Which city have both Gina and Jon visited?", "u2"),
        );
        let mut ids2: Vec<&str> = r2.iter().map(|r| r.doc_id.as_str()).collect();
        ids2.sort_unstable();
        assert_eq!(ids2, vec!["d5", "d6"]);
        assert!(r2
            .iter()
            .all(|r| r.relation_evidence == vec!["shared relation: Paris"]));
    }

    fn scored_result(doc_id: &str, score: f32) -> crate::SearchResult {
        crate::SearchResult {
            doc_id: doc_id.to_string(),
            source: "test".to_string(),
            group_id: None,
            score,
            score_breakdown: crate::index::ScoreBreakdown::default(),
            matched_entities: Vec::new(),
            matched_terms: Vec::new(),
            probable_topic: None,
            doc_type_guess: None,
            semantic_status: None,
            superseded_by: None,
            relation_confidence: None,
            relation_evidence: Vec::new(),
        }
    }

    #[test]
    fn blend_puts_structured_first_dedupes_and_respects_top_k() {
        let structured = vec![scored_result("d1", 1001.0)];
        let lexical = vec![
            scored_result("d1", 9.0), // duplicate: dropped
            scored_result("d2", 8.0),
            scored_result("d3", 7.0),
        ];
        let blended = blend_structured_first(structured, lexical, 2);
        let ids: Vec<&str> = blended.iter().map(|r| r.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["d1", "d2"], "structured first, deduped, top_k=2");
        // Empty structured input returns the lexical list untouched.
        let lexical = vec![scored_result("d9", 5.0)];
        let blended = blend_structured_first(Vec::new(), lexical, 10);
        assert_eq!(blended.len(), 1);
        assert_eq!(blended[0].doc_id, "d9");
    }

    // ---------------- key-phrase enrichment ----------------

    /// Fake extractor: one content-derived phrase per input turn, no spaCy.
    /// `slow` keeps a batch "in flight" longer for the staleness test.
    fn write_fake_extractor(name: &str, slow: bool) -> std::path::PathBuf {
        let dir = canonical_temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake_extractor.py");
        let delay = if slow { "0.3" } else { "0" };
        std::fs::write(
            &script,
            format!(
                r#"#!/usr/bin/env python3
import json, sys, time
time.sleep({delay})
payload = json.load(sys.stdin)
out = []
for t in payload.get("turns", []):
    text = t.get("text", "")
    marker = text[-12:] if len(text) >= 12 else text
    out.append({{
        "text": "phrase-" + marker,
        "kind": "thing",
        "session_id": t.get("session_id", ""),
        "doc_id": t.get("doc_id", ""),
        "turn_idx": t.get("turn_idx", 0),
    }})
json.dump({{"relations": [], "key_phrases": out}}, sys.stdout)
"#
            ),
        )
        .unwrap();
        script
    }

    fn enrichment_service(script: &std::path::Path) -> MemoryService {
        MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: true,
            extractor_script: Some(script.to_path_buf()),
            ..PipelineOptions::default()
        })
    }

    fn add_message(
        service: &mut MemoryService,
        request_id: &str,
        user_id: &str,
        session_id: &str,
        content: &str,
    ) {
        service
            .add(AddRequest {
                request_id: request_id.into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: content.into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: user_id.into(),
                session_id: session_id.into(),
            })
            .unwrap();
    }

    static FLUSH_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn flush_id() -> String {
        let n = FLUSH_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("flush-{n}")
    }

    fn doc_id_for(user_id: &str, request_id: &str) -> String {
        crate::stable_doc_id_from_source(&format!("{user_id}:{request_id}:0"))
    }

    fn key_phrases_of(service: &MemoryService, doc_id: &str) -> Vec<String> {
        service
            .store
            .source_document_by_id(doc_id)
            .map(|d| d.key_phrases.iter().map(|p| p.text.clone()).collect())
            .unwrap_or_default()
    }

    /// Poll `cond` until it holds or `timeout` elapses.
    fn poll_until(timeout: std::time::Duration, mut cond: impl FnMut() -> bool) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if cond() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        cond()
    }

    #[test]
    fn key_phrase_enrichment_backfills_documents_in_background() {
        let script = write_fake_extractor("enrich-fast", false);
        let mut service = enrichment_service(&script);
        add_message(
            &mut service,
            "req-1",
            "user-a",
            "sess-a",
            "I love the Paris jazz festival",
        );
        // The write returned immediately with no phrases; the worker needs
        // its linger plus one script run before a follow-up write applies
        // the inbox.
        let doc_id = doc_id_for("user-a", "req-1");
        assert!(key_phrases_of(&service, &doc_id).is_empty());
        let applied = poll_until(std::time::Duration::from_secs(15), || {
            add_message(&mut service, &flush_id(), "user-a", "sess-a", "flush");
            !key_phrases_of(&service, &doc_id).is_empty()
        });
        assert!(
            applied,
            "enriched phrases should land after a follow-up write"
        );
        let phrases = key_phrases_of(&service, &doc_id);
        assert_eq!(phrases.len(), 1);
        assert!(
            phrases[0].starts_with("phrase-"),
            "unexpected phrase: {}",
            phrases[0]
        );
        // Search still works after the backfill + refresh cycle.
        let response = service
            .search(SearchRequest {
                query: "jazz festival".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert!(response
            .data
            .iter()
            .any(|m| m.content.contains("jazz festival")));
    }

    #[test]
    fn key_phrase_enrichment_fails_open_without_extractor() {
        let mut service = MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: true,
            extractor_script: Some(std::path::PathBuf::from("/nonexistent/extractor.py")),
            ..PipelineOptions::default()
        });
        add_message(
            &mut service,
            "req-1",
            "user-a",
            "sess-a",
            "I love the Paris jazz festival",
        );
        // Give the worker a chance to fail; the write path is unaffected.
        std::thread::sleep(std::time::Duration::from_secs(2));
        add_message(&mut service, "req-2", "user-a", "sess-a", "second note");
        assert!(key_phrases_of(&service, &doc_id_for("user-a", "req-1")).is_empty());
        let response = service
            .search(SearchRequest {
                query: "jazz festival".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("jazz festival"));
    }

    #[test]
    fn key_phrase_enrichment_disabled_queues_nothing() {
        let script = write_fake_extractor("enrich-off", false);
        let mut service = MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: false,
            extractor_script: Some(script),
            ..PipelineOptions::default()
        });
        add_message(
            &mut service,
            "req-1",
            "user-a",
            "sess-a",
            "I love the Paris jazz festival",
        );
        std::thread::sleep(std::time::Duration::from_secs(2));
        add_message(&mut service, "req-2", "user-a", "sess-a", "second note");
        assert_eq!(service.enrichment_queue_len(), 0);
        assert!(key_phrases_of(&service, &doc_id_for("user-a", "req-1")).is_empty());
    }

    #[test]
    fn key_phrase_enrichment_drops_stale_batches() {
        let script = write_fake_extractor("enrich-slow", true);
        let mut service = enrichment_service(&script);
        add_message(&mut service, "req-1", "user-a", "sess-a", "old content AAA");
        let doc_id = doc_id_for("user-a", "req-1");
        // The worker drains the queue after its 500ms linger; the slow
        // script keeps that batch in flight, so this update lands mid-flight.
        std::thread::sleep(std::time::Duration::from_millis(700));
        service
            .update(UpdateRequest {
                user_id: "user-a".into(),
                memory_id: doc_id.clone(),
                content: "new content BBB".into(),
                role: None,
                timestamp: None,
                expires_at_ms: None,
            })
            .unwrap();
        // Flush until the re-queued (new-content) batch lands. The stale
        // batch's phrases must never be applied: the marker is
        // content-derived, so old phrases would contain "AAA".
        let settled = poll_until(std::time::Duration::from_secs(25), || {
            add_message(&mut service, &flush_id(), "user-a", "sess-a", "flush");
            std::thread::sleep(std::time::Duration::from_millis(1200));
            !key_phrases_of(&service, &doc_id).is_empty()
        });
        assert!(settled, "re-queued enrichment should land");
        let phrases = key_phrases_of(&service, &doc_id);
        assert!(
            phrases.iter().any(|p| p.contains("BBB")),
            "phrases should describe the new content: {phrases:?}"
        );
        assert!(
            phrases.iter().all(|p| !p.contains("AAA")),
            "stale phrases must be dropped: {phrases:?}"
        );
    }

    #[test]
    fn key_phrase_enrichment_stress_concurrent_writes_and_reads() {
        use std::sync::{Arc, RwLock};
        let script = write_fake_extractor("enrich-stress", false);
        let service = Arc::new(RwLock::new(enrichment_service(&script)));
        let writers: Vec<_> = (0..6)
            .map(|w| {
                let service = Arc::clone(&service);
                std::thread::spawn(move || {
                    for i in 0..40 {
                        let mut guard = service.write().unwrap();
                        guard
                            .add(AddRequest {
                                request_id: format!("stress-{w}-{i}"),
                                messages: vec![Message {
                                    role: "user".into(),
                                    timestamp: None,
                                    content: format!("writer {w} note number {i} about topology"),
                                    expires_at_ms: None,
                                    supersedes_id: None,
                                }],
                                user_id: "user-a".into(),
                                session_id: format!("sess-{w}"),
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        let readers: Vec<_> = (0..2)
            .map(|_| {
                let service = Arc::clone(&service);
                std::thread::spawn(move || {
                    for _ in 0..60 {
                        let mut guard = service.write().unwrap();
                        let _ = guard.search(SearchRequest {
                            query: "topology".into(),
                            options: None,
                            user_id: "user-a".into(),
                            top_k: 5,
                            session_id: None,
                            scope: None,
                            filters: None,
                lang: None,
});
                        drop(guard);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        for r in readers {
            r.join().unwrap();
        }
        // Flush the enrichment pipeline: each flush write drains the inbox,
        // and the 1.2s pause lets the worker finish its batch.
        let mut guard = service.write().unwrap();
        let all_doc_ids: Vec<String> = (0..6)
            .flat_map(|w| (0..40).map(move |i| doc_id_for("user-a", &format!("stress-{w}-{i}"))))
            .collect();
        let mut done = false;
        for _ in 0..25 {
            add_message(&mut guard, &flush_id(), "user-a", "sess-flush", "flush");
            std::thread::sleep(std::time::Duration::from_millis(1200));
            if all_doc_ids
                .iter()
                .all(|id| !key_phrases_of(&guard, id).is_empty())
            {
                done = true;
                break;
            }
        }
        assert!(done, "all 240 stress docs should be enriched");
        assert!(guard.enrichment_queue_len() <= KEY_PHRASE_PENDING_MAX);
        // Spot-check search quality after the backfill storm.
        let response = guard
            .search(SearchRequest {
                query: "topology".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        assert!(!response.data.is_empty());
    }

    // ------------------------------------------------------------------
    // Query-time key-phrase backfill coverage.
    //
    // Provider hooks (Claude Code, Codex, Gemini) run as short-lived
    // processes: they upsert, refresh, and exit before the background
    // enrichment worker's linger completes. These tests cover the
    // synchronous backfill that closes that gap.
    // ------------------------------------------------------------------

    /// Fake extractor emitting one fixed phrase per turn. The phrase is a
    /// content mention (like real extractor output), so tests put it in the
    /// document text.
    fn write_canary_extractor(name: &str) -> std::path::PathBuf {
        let dir = canonical_temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("canary_extractor.py");
        std::fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json, sys
payload = json.load(sys.stdin)
out = []
for t in payload.get("turns", []):
    out.append({
        "text": "canary phrase",
        "kind": "thing",
        "session_id": t.get("session_id", ""),
        "doc_id": t.get("doc_id", ""),
        "turn_idx": t.get("turn_idx", 0),
    })
json.dump({"relations": [], "key_phrases": out}, sys.stdout)
"#,
        )
        .unwrap();
        script
    }

    /// Fake extractor that runs clean but finds no phrases: a legitimate
    /// empty result, distinct from a failed run.
    fn write_empty_extractor(name: &str) -> std::path::PathBuf {
        let dir = canonical_temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("empty_extractor.py");
        std::fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json, sys
json.load(sys.stdin)
json.dump({"relations": [], "key_phrases": []}, sys.stdout)
"#,
        )
        .unwrap();
        script
    }

    fn extractor_service(script: &std::path::Path) -> MemoryService {
        MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: true,
            extractor_script: Some(script.to_path_buf()),
            ..PipelineOptions::default()
        })
    }

    /// A document written the way a dead hook process leaves it: present
    /// in the store with no phrases and no extraction stamp.
    fn direct_doc(doc_id: &str, content: &str) -> SourceDocument {
        SourceDocument {
            doc_id: doc_id.to_string(),
            source: format!("test://{doc_id}"),
            content: content.to_string(),
            concept: String::new(),
            group_id: None,
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
            filters: BTreeMap::new(),
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    #[test]
    fn key_phrase_backfill_applies_synchronously_without_worker() {
        let script = write_canary_extractor("backfill-sync");
        let mut service = extractor_service(&script);
        // store.upsert bypasses the background queue entirely, the way a
        // hook-written document looks after its process died.
        service.store.upsert(direct_doc(
            "sync-doc",
            "the canary phrase in the quick brown fox story",
        ));
        service.store.refresh().unwrap();
        assert!(service.key_phrase_backfill_needed());
        assert_eq!(service.backfill_key_phrases(), 1);
        assert_eq!(
            key_phrases_of(&service, "sync-doc"),
            vec!["canary phrase".to_string()]
        );
        assert!(!service.key_phrase_backfill_needed());
        // Idempotent: a second backfill finds nothing to do.
        assert_eq!(service.backfill_key_phrases(), 0);
    }

    #[test]
    fn key_phrase_backfill_runs_inside_search_with_filters() {
        // The MCP query entry point triggers the backfill itself, so a
        // hook-written document's phrases are extracted before this query
        // runs: phrases are absent before the search and present after.
        let script = write_canary_extractor("backfill-search");
        let mut service = extractor_service(&script);
        service.store.upsert(direct_doc(
            "search-doc",
            "the canary phrase in the quick brown fox story",
        ));
        service.store.refresh().unwrap();
        assert!(key_phrases_of(&service, "search-doc").is_empty());
        let results = service
            .search_with_filters("canary phrase", "test", None, 10, &BTreeMap::new())
            .unwrap();
        assert!(
            results.iter().any(|r| r.doc_id == "search-doc"),
            "query should retrieve the backfilled document"
        );
        assert_eq!(
            key_phrases_of(&service, "search-doc"),
            vec!["canary phrase".to_string()]
        );
        // The rebuilt record carries the phrase as a key entity, so the
        // query that triggered the backfill already sees it indexed.
        // Head-noun admission: the key entity is "phrase", not the whole
        // "canary phrase" (modifiers are not the thing denoted).
        let record = service.store.record_by_id("search-doc").unwrap();
        assert!(record
            .key_entities
            .iter()
            .any(|e| e.text == "phrase"));
    }

    #[test]
    fn key_phrase_empty_extraction_is_durably_complete() {
        // An extractor that runs clean but finds nothing still stamps the
        // document: it must not be re-extracted on every later query.
        let script = write_empty_extractor("backfill-empty");
        let mut service = extractor_service(&script);
        service
            .store
            .upsert(direct_doc("empty-doc", "some ordinary content here"));
        assert_eq!(service.backfill_key_phrases(), 1);
        assert!(key_phrases_of(&service, "empty-doc").is_empty());
        assert!(!service.key_phrase_backfill_needed());
        assert_eq!(service.backfill_key_phrases(), 0);
    }

    #[test]
    fn key_phrase_backfill_persists_across_restart_with_stamp() {
        let script = write_canary_extractor("backfill-restart");
        let dir = canonical_temp_dir("backfill-restart");
        std::fs::create_dir_all(&dir).unwrap();
        let options = PipelineOptions {
            key_phrase_enrichment: true,
            extractor_script: Some(script),
            ..PipelineOptions::default()
        };
        {
            let mut service = MemoryService::at_path(&dir, options.clone()).unwrap();
            service
                .store
                .upsert(direct_doc("restart-doc", "persistent content for restart"));
            service.store.refresh().unwrap();
            assert_eq!(service.backfill_key_phrases(), 1);
            // refresh persists; dropping here closes the "hook process".
        }
        let reopened = MemoryService::at_path(&dir, options).unwrap();
        assert_eq!(
            key_phrases_of(&reopened, "restart-doc"),
            vec!["canary phrase".to_string()]
        );
        // The extraction stamp survived the restart: no re-extraction.
        assert!(!reopened.key_phrase_backfill_needed());
    }

    #[test]
    fn key_phrase_timeout_stays_retryable() {
        // A timed-out extractor run must not stamp the document as done.
        let script = write_fake_extractor("backfill-slow", true);
        let probe = direct_doc("slow-doc", "some slow content here");
        let turns = crate::segments::relations::relation_turns_from_docs(&[&probe]);
        assert!(
            run_key_phrase_extraction_bounded(&turns, Some(&script), 0).is_none(),
            "zero-second timeout must report a transient failure"
        );

        let mut service = extractor_service(&script);
        service
            .store
            .upsert(direct_doc("slow-doc", "some slow content here"));
        let docs = vec![direct_doc("slow-doc", "some slow content here")];
        // None = transient failure: nothing applied, nothing stamped, so a
        // later query retries the document.
        assert_eq!(service.apply_key_phrase_backfill(&docs, None), 0);
        assert!(key_phrases_of(&service, "slow-doc").is_empty());
        assert!(service.key_phrase_backfill_needed());
    }

    #[test]
    fn key_phrase_backfill_drops_handled_docs_from_worker_queue() {
        let script = write_canary_extractor("backfill-drop");
        let mut service = extractor_service(&script);
        // MemoryService::upsert queues the turn for the background worker.
        service.upsert(direct_doc("drop-doc", "content for queue drop"));
        assert_eq!(service.enrichment_queue_len(), 1);
        assert_eq!(service.backfill_key_phrases(), 1);
        assert_eq!(service.enrichment_queue_len(), 0);
        assert!(!service.key_phrase_backfill_needed());
    }

    #[test]
    fn key_phrase_backfill_rejects_stale_results_after_replace() {
        let script = write_canary_extractor("backfill-stale");
        let mut service = extractor_service(&script);
        service
            .store
            .upsert(direct_doc("stale-doc", "version one content here"));
        assert_eq!(service.backfill_key_phrases(), 1);
        assert!(!service.key_phrase_backfill_needed());
        // Same id, new content: the old stamp no longer matches, so the
        // document needs extraction again.
        service
            .store
            .upsert(direct_doc("stale-doc", "version two completely different"));
        assert!(service.key_phrase_backfill_needed());
        assert_eq!(service.backfill_key_phrases(), 1);
        assert_eq!(
            key_phrases_of(&service, "stale-doc"),
            vec!["canary phrase".to_string()]
        );
        assert!(!service.key_phrase_backfill_needed());
    }

    #[test]
    fn key_phrase_refresh_index_drains_enrichment_inbox() {
        let script = write_fake_extractor("refresh-drain", false);
        let mut service = enrichment_service(&script);
        add_message(
            &mut service,
            "req-drain",
            "user-a",
            "sess-a",
            "I love the Paris jazz festival",
        );
        let doc_id = doc_id_for("user-a", "req-drain");
        // No follow-up write: the worker finishes on its own, and
        // refresh_index applies whatever the inbox holds.
        let applied = poll_until(std::time::Duration::from_secs(20), || {
            let _ = service.refresh_index();
            !key_phrases_of(&service, &doc_id).is_empty()
        });
        assert!(applied, "refresh_index should drain the enrichment inbox");
    }

    #[test]
    fn key_phrase_worker_drains_more_than_one_batch() {
        let script = write_fake_extractor("worker-multibatch", false);
        let mut service = enrichment_service(&script);
        for i in 0..40 {
            add_message(
                &mut service,
                &format!("req-batch-{i}"),
                "user-a",
                "sess-a",
                &format!("message number {i} about topology"),
            );
        }
        // 40 documents exceed one 32-document worker batch; the worker must
        // keep draining without another write to wake it.
        let applied = poll_until(std::time::Duration::from_secs(60), || {
            let _ = service.refresh_index();
            (0..40).all(|i| {
                !key_phrases_of(&service, &doc_id_for("user-a", &format!("req-batch-{i}")))
                    .is_empty()
            })
        });
        assert!(
            applied,
            "worker should drain all batches without another write"
        );
    }

    #[test]
    fn key_phrase_backfill_fail_open_without_extractor() {
        let mut service = MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: true,
            extractor_script: Some(std::path::PathBuf::from("/nonexistent/extractor.py")),
            ..PipelineOptions::default()
        });
        service
            .store
            .upsert(direct_doc("fail-doc", "content that cannot be enriched"));
        // Broken script: the subprocess cannot run, so the document is NOT
        // stamped — it stays queued for a later retry — and the service
        // keeps working: fail-open at the query level.
        assert_eq!(service.backfill_key_phrases(), 0);
        assert!(key_phrases_of(&service, "fail-doc").is_empty());
        assert!(service.key_phrase_backfill_needed());
        let response = service
            .search(SearchRequest {
                query: "enriched".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: None,
                scope: None,
                filters: None,
                lang: None,
})
            .unwrap();
        let _ = response;
    }

    /// Mirror of a provider hook's write sequence: a short-lived process
    /// opens the service with the hook's options, upserts, refreshes, and
    /// exits before the background worker finishes. The follow-up query
    /// (a fresh process, like the MCP server) must still get phrases.
    fn hook_write_then_query(name: &str, provider: &str, doc_id: &str, content: &str, query: &str) {
        let script = write_canary_extractor(name);
        let dir = canonical_temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        // Same option shape as the Claude Code / Codex / Gemini hooks:
        // segmented layout with top-3 routing, enrichment on.
        let options = PipelineOptions {
            memory_index_layout: crate::MemoryIndexLayout::Segmented {
                query_top_n: 3,
                routing_strategy: crate::segments::SegmentRoutingStrategy::LocalDistinctiveness,
            },
            key_phrase_enrichment: true,
            extractor_script: Some(script),
            ..PipelineOptions::default()
        };
        {
            let mut service = MemoryService::at_path(&dir, options.clone()).unwrap();
            let mut doc = direct_doc(doc_id, content);
            doc.author_agent = Some(provider.to_string());
            service.upsert(doc);
            service.refresh_index().unwrap();
            // Hook process exits here; its worker never finishes.
        }
        let mut query_side = MemoryService::at_path(&dir, options).unwrap();
        // The hook's worker died with its process: nothing extracted yet.
        assert!(
            key_phrases_of(&query_side, doc_id).is_empty(),
            "{provider} hook-written doc should start without phrases"
        );
        let results = query_side
            .search_with_filters(query, "test", None, 10, &BTreeMap::new())
            .unwrap();
        assert!(
            results.iter().any(|r| r.doc_id == doc_id),
            "{provider} hook-written doc should be retrievable"
        );
        assert_eq!(
            key_phrases_of(&query_side, doc_id),
            vec!["canary phrase".to_string()],
            "{provider} hook-written doc should carry backfilled phrases"
        );
    }

    #[test]
    fn key_phrase_backfill_covers_claude_code_hook_path() {
        hook_write_then_query(
            "hook-claude",
            "claude-code",
            "claude-hook-doc",
            "claude code session outcome notes with the canary phrase about the deployment",
            "canary phrase",
        );
    }

    #[test]
    fn key_phrase_backfill_covers_codex_hook_path() {
        hook_write_then_query(
            "hook-codex",
            "codex",
            "codex-hook-doc",
            "codex session transcript with the canary phrase about the refactor",
            "canary phrase",
        );
    }

    #[test]
    fn key_phrase_backfill_covers_gemini_hook_path() {
        hook_write_then_query(
            "hook-gemini",
            "gemini-cli",
            "gemini-hook-doc",
            "gemini cli tool call record with the canary phrase for the migration",
            "canary phrase",
        );
    }
}

#[cfg(test)]
mod board_integration_tests {
    use super::*;
    use crate::board::*;
    use crate::PipelineOptions;
    use std::sync::{Arc, Mutex};

    const OWNER: &str = "mcp";
    const WS: &str = "ws-board-test";
    const S1: &str = "sess-1";
    const S2: &str = "sess-2";

    fn svc() -> MemoryService {
        MemoryService::in_memory(PipelineOptions::default())
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let dir = base.join(format!("lint-ai-board-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn post(
        service: &mut MemoryService,
        board_id: Option<&str>,
        session: Option<&str>,
        content: &str,
        request_id: &str,
    ) -> BoardPost {
        service
            .board_post(
                board_id, OWNER, WS, session, "agent-1", "claude", content, request_id,
            )
            .unwrap()
    }

    #[test]
    fn default_board_listed_before_persistence() {
        let service = svc();
        let boards = service.board_list(OWNER, WS, Some(S1)).unwrap();
        let alias = boards.iter().find(|b| b.key == DEFAULT_BOARD_KEY).unwrap();
        assert_eq!(alias.board_id, default_board_id(OWNER, WS, S1));
        // board_info synthesizes the default too.
        let info = service
            .board_info("default", OWNER, WS, Some(S1))
            .unwrap()
            .unwrap();
        assert_eq!(info.board_id, default_board_id(OWNER, WS, S1));
    }

    #[test]
    fn no_session_no_default_entry() {
        let service = svc();
        let boards = service.board_list(OWNER, WS, None).unwrap();
        assert!(boards.iter().all(|b| b.key != DEFAULT_BOARD_KEY));
        assert!(service.board_info("default", OWNER, WS, None).is_err());
    }

    #[test]
    fn first_default_post_creates_persistent_record() {
        let mut service = svc();
        let p = post(&mut service, None, Some(S1), "hello", "r1");
        assert_eq!(p.sequence, 1);
        assert_eq!(p.board_id, default_board_id(OWNER, WS, S1));
        // The record is now persistent: board_info finds it by real ID.
        let info = service
            .board_info(&p.board_id, OWNER, WS, Some(S1))
            .unwrap()
            .unwrap();
        assert!(!info.created_at.is_empty());
        assert_eq!(info.key, session_board_key(S1));
    }

    #[test]
    fn omitted_and_alias_resolve_to_same_board() {
        let mut service = svc();
        let a = post(&mut service, None, Some(S1), "a", "r1");
        let b = post(&mut service, Some("default"), Some(S1), "b", "r2");
        assert_eq!(a.board_id, b.board_id);
        assert_eq!(b.sequence, 2);
    }

    #[test]
    fn default_requires_session() {
        let mut service = svc();
        let err = service
            .board_post(None, OWNER, WS, None, "agent-1", "claude", "x", "r1")
            .unwrap_err();
        assert!(err.to_string().contains("session_id"), "{err}");
    }

    #[test]
    fn empty_default_read_and_search() {
        let mut service = svc();
        let posts = service
            .board_read(None, OWNER, WS, Some(S1), None, 20)
            .unwrap();
        assert!(posts.is_empty());
        let hits = service
            .board_search(None, OWNER, WS, Some(S1), "anything", 10)
            .unwrap();
        assert!(hits.is_empty());
        assert!(service
            .board_get(None, OWNER, WS, Some(S1), "nope")
            .unwrap()
            .is_none());
    }

    #[test]
    fn explicit_unknown_board_post_errors() {
        let mut service = svc();
        let err = post_unknown(&mut service).unwrap_err();
        assert!(err.to_string().contains("unknown board_id"), "{err}");
    }

    fn post_unknown(service: &mut MemoryService) -> anyhow::Result<BoardPost> {
        service.board_post(
            Some("board:nope:nope:nope"),
            OWNER,
            WS,
            Some(S1),
            "agent-1",
            "claude",
            "x",
            "r1",
        )
    }

    #[test]
    fn board_open_is_idempotent() {
        let mut service = svc();
        let a = service.board_open(OWNER, WS, "task-a", "Task A").unwrap();
        let b = service.board_open(OWNER, WS, "task-a", "Different title").unwrap();
        assert_eq!(a.board_id, b.board_id);
        assert_eq!(a.title, "Task A");
    }

    #[test]
    fn reserved_keys_rejected() {
        let mut service = svc();
        assert!(service.board_open(OWNER, WS, "session:evil", "x").is_err());
        assert!(service.board_open(OWNER, WS, "default", "x").is_err());
        // The MCP-facing session opener still works.
        let b = service.board_open_session(OWNER, WS, S1, "T").unwrap();
        assert_eq!(b.board_id, default_board_id(OWNER, WS, S1));
    }

    #[test]
    fn sessions_are_isolated() {
        let mut service = svc();
        let p1 = post(&mut service, None, Some(S1), "hello s1", "r1");
        let p2 = post(&mut service, None, Some(S2), "hello s2", "r1");
        assert_ne!(p1.board_id, p2.board_id);
        // Same request_id on different sessions' boards: no collision.
        assert_ne!(p1.post_id, p2.post_id);
        let s1_posts = service
            .board_read(None, OWNER, WS, Some(S1), None, 20)
            .unwrap();
        assert_eq!(s1_posts.len(), 1);
        assert_eq!(s1_posts[0].sequence, 1);
        assert_eq!(s1_posts[0].content, "hello s1");
        let s2_posts = service
            .board_read(None, OWNER, WS, Some(S2), None, 20)
            .unwrap();
        assert_eq!(s2_posts.len(), 1);
        assert_eq!(s2_posts[0].content, "hello s2");
    }

    #[test]
    fn sequences_increase_per_board() {
        let mut service = svc();
        let named = service.board_open(OWNER, WS, "k", "K").unwrap();
        for i in 1..=3u64 {
            let p = post(
                &mut service,
                Some(&named.board_id),
                Some(S1),
                &format!("n{i}"),
                &format!("nr{i}"),
            );
            assert_eq!(p.sequence, i);
        }
        // The default board has its own sequence space.
        let d = post(&mut service, None, Some(S1), "d1", "dr1");
        assert_eq!(d.sequence, 1);
    }

    #[test]
    fn request_id_retry_returns_original() {
        let mut service = svc();
        let a = post(&mut service, None, Some(S1), "first", "retry-1");
        let b = post(&mut service, None, Some(S1), "second", "retry-1");
        assert_eq!(a.post_id, b.post_id);
        assert_eq!(b.content, "first");
        assert_eq!(b.sequence, 1);
        let posts = service
            .board_read(None, OWNER, WS, Some(S1), None, 20)
            .unwrap();
        assert_eq!(posts.len(), 1);
    }

    #[test]
    fn after_sequence_catchup() {
        let mut service = svc();
        post(&mut service, None, Some(S1), "one", "r1");
        post(&mut service, None, Some(S1), "two", "r2");
        let posts = service
            .board_read(None, OWNER, WS, Some(S1), Some(1), 20)
            .unwrap();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0].content, "two");
    }

    #[test]
    fn wrong_board_post_id_rejected() {
        let mut service = svc();
        let named = service.board_open(OWNER, WS, "k", "K").unwrap();
        let p = post(
            &mut service,
            Some(&named.board_id),
            Some(S1),
            "secret",
            "r1",
        );
        // Same post ID through the default board: not found.
        assert!(service
            .board_get(None, OWNER, WS, Some(S1), &p.post_id)
            .unwrap()
            .is_none());
        // Through the right board: found.
        assert!(service
            .board_get(Some(&named.board_id), OWNER, WS, Some(S1), &p.post_id)
            .unwrap()
            .is_some());
    }

    #[test]
    fn focused_and_default_boards_isolated() {
        let mut service = svc();
        let named = service.board_open(OWNER, WS, "focus", "Focus").unwrap();
        post(
            &mut service,
            Some(&named.board_id),
            Some(S1),
            "focused note",
            "r1",
        );
        // Default board is untouched.
        assert!(service
            .board_read(None, OWNER, WS, Some(S1), None, 20)
            .unwrap()
            .is_empty());
        // Search on the default board doesn't leak the focused post.
        assert!(service
            .board_search(None, OWNER, WS, Some(S1), "focused", 10)
            .unwrap()
            .is_empty());
        // And vice versa.
        post(&mut service, None, Some(S1), "default note", "r2");
        assert!(service
            .board_search(Some(&named.board_id), OWNER, WS, Some(S1), "default", 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn board_search_finds_posts() {
        let mut service = svc();
        post(&mut service, None, Some(S1), "the parser failure comes from empty input", "r1");
        post(&mut service, None, Some(S1), "unrelated status update", "r2");
        let hits = service
            .board_search(None, OWNER, WS, Some(S1), "parser failure", 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("parser failure"));
    }

    #[test]
    fn owner_workspace_scoping() {
        let mut service = svc();
        post(&mut service, None, Some(S1), "x", "r1");
        // Other workspace: no boards, default alias resolves to its own board.
        let other = service.board_list(OWNER, "other-ws", Some(S1)).unwrap();
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].key, DEFAULT_BOARD_KEY);
        assert_eq!(other[0].board_id, default_board_id(OWNER, "other-ws", S1));
        // Other owner: same.
        let other_owner = service.board_list("someone-else", WS, Some(S1)).unwrap();
        assert_eq!(other_owner.len(), 1);
        assert_eq!(
            other_owner[0].board_id,
            default_board_id("someone-else", WS, S1)
        );
    }

    #[test]
    fn request_id_and_sequence_survive_restart() {
        let dir = temp_dir("restart");
        let options = PipelineOptions::default();
        let mut service = MemoryService::at_path(&dir, options.clone()).unwrap();
        let p1 = post(&mut service, None, Some(S1), "first", "keep-me");
        assert_eq!(p1.sequence, 1);
        post(&mut service, None, Some(S1), "second", "r2");
        drop(service);
        // Reopen: the store scan recovers request IDs and sequences.
        let mut reopened = MemoryService::at_path(&dir, options).unwrap();
        let retry = post(&mut reopened, None, Some(S1), "changed", "keep-me");
        assert_eq!(retry.post_id, p1.post_id);
        assert_eq!(retry.content, "first");
        let p3 = post(&mut reopened, None, Some(S1), "third", "r3");
        assert_eq!(p3.sequence, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_posts_get_unique_sequences() {
        let service = Arc::new(Mutex::new(svc()));
        let mut handles = Vec::new();
        for t in 0..8u32 {
            let service = Arc::clone(&service);
            handles.push(std::thread::spawn(move || {
                for i in 0..5u32 {
                    let mut guard = service.lock().unwrap();
                    post(
                        &mut guard,
                        None,
                        Some(S1),
                        &format!("t{t} i{i}"),
                        &format!("conc-{t}-{i}"),
                    );
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let mut guard = service.lock().unwrap();
        let posts = guard
            .board_read(None, OWNER, WS, Some(S1), None, 100)
            .unwrap();
        assert_eq!(posts.len(), 40);
        let mut seqs: Vec<u64> = posts.iter().map(|p| p.sequence).collect();
        seqs.sort_unstable();
        assert_eq!(seqs, (1..=40).collect::<Vec<_>>());
    }
}
