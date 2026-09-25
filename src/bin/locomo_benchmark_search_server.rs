//! Long-lived Lint-AI search server over LoCoMo conversations (pillar-4 agent tool).
//!
//! Builds one in-memory [`MemoryService`] per conversation at startup
//! (production pipeline options, segmented layout with the gated-local
//! routing strategy), then serves:
//!   GET /health                        -> {"ok":true}
//!   GET /search?conv=<sample_id>&q=<query>&k=<n>
//!       -> {"results":[{"session_id":"...","score":1.23,"text":"..."}]}
//!
//! Searches run through the agent's entry point --
//! [`MemoryService::search_with_filters`], the same function the MCP `search`
//! tools call -- with results shaped by the shared `search_results`
//! formatter, so the benchmark measures exactly what an agent experiences:
//! follow-up phrasing resolved against observed session state, the
//! conversational reranker, structured-fact relation evidence, and the
//! query-time key-phrase backfill. The server owns no query logic of its
//! own; it only indexes the turns and serves the agent's ranking.
//!
//! Hits return the matched turn text (not the whole session) with the
//! session date prefixed, so the reader can resolve relative dates
//! ("yesterday", "last month") without a wall of full-session text.
//!
//! This is benchmark scaffolding, not part of the shipped product.

use anyhow::{Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use clap::Parser;
use lint_ai::memory_api::MemoryService;
use lint_ai::segments::relations::{extract_relations_via_spacy, RelationTurn};
use lint_ai::{default_production_pipeline_options, search_results, SourceDocument};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// User id stamped on every LoCoMo document; the agent query path filters on
/// it, mirroring how `AddRequest` stamps real writes.
const LOCOMO_USER_ID: &str = "locomo-agent";

/// Scope key for the benchmark's conversation state. The MCP `search` tools
/// pass their provider name here ("muse", "claude-code", ...); the benchmark
/// is its own scope, so follow-up resolution and the conversational reranker
/// behave exactly as they do for an agent.
const BENCHMARK_SCOPE: &str = "benchmark";

#[derive(Debug, Parser)]
struct Args {
    /// Path to locomo10.json from snap-research/locomo.
    #[arg(long)]
    locomo: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8099")]
    bind: String,
}

#[derive(Debug, Deserialize)]
struct LocomoConversation {
    sample_id: String,
    #[allow(dead_code)]
    qa: Vec<serde_json::Value>,
    conversation: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    text: String,
    /// BLIP image caption for turns sharing a photo. Deserialization drops it
    /// by default; we index it so image content is retrievable and visible to
    /// the reader (e.g. conv-41_q3's certificate).
    #[serde(default)]
    blip_caption: Option<String>,
    /// Image topic/query accompanying the photo, when present.
    #[serde(default)]
    query: Option<String>,
}

struct ConvIndex {
    searcher: Mutex<MemoryService>,
}

struct AppState {
    convs: HashMap<String, Arc<ConvIndex>>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    conv: String,
    q: String,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
struct Hit {
    session_id: String,
    score: f32,
    text: String,
}

#[derive(Debug, Serialize)]
struct SearchResp {
    results: Vec<Hit>,
}

fn session_number(key: &str) -> Option<usize> {
    key.strip_prefix("session_")?
        .split('_')
        .next()?
        .parse()
        .ok()
}

fn build_conv_index(conv: &LocomoConversation) -> Result<ConvIndex> {
    let mut session_nums: Vec<usize> = conv
        .conversation
        .keys()
        .filter_map(|k| session_number(k))
        .collect();
    session_nums.sort_unstable();
    session_nums.dedup();

    let mut docs = Vec::new();
    let mut rel_turns: Vec<RelationTurn> = Vec::new();
    for n in session_nums {
        let key = format!("session_{n}");
        let date_key = format!("session_{n}_date_time");
        let date = conv
            .conversation
            .get(&date_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let turns: Vec<LocomoTurn> = serde_json::from_value(
            conv.conversation
                .get(&key)
                .cloned()
                .unwrap_or(serde_json::Value::Array(vec![])),
        )
        .with_context(|| format!("failed to parse turns for {key}"))?;
        let group_id = format!("{}::session_{n}", conv.sample_id);
        for (turn_idx, turn) in turns.iter().enumerate() {
            // Image turns carry their visual content in blip_caption/query;
            // index it inline so it is routable and reader-visible. One
            // systematic place: the turn line that feeds the indexed
            // document content.
            let mut line = format!("{}: {}", turn.speaker, turn.text);
            if let Some(caption) = turn
                .blip_caption
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                line.push_str(&format!(" [image: {}]", caption));
            }
            if let Some(q) = turn
                .query
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                line.push_str(&format!(" [image topic: {}]", q));
            }
            let mut filters = BTreeMap::new();
            filters.insert("memory_user_id".to_string(), LOCOMO_USER_ID.to_string());
            let doc_id = format!("{group_id}::turn{turn_idx}");
            rel_turns.push(RelationTurn {
                speaker: turn.speaker.clone(),
                text: turn.text.clone(),
                session_id: group_id.clone(),
                turn_idx,
                doc_id: doc_id.clone(),
                session_date: date.clone(),
            });
            docs.push(SourceDocument {
                doc_id,
                source: format!("locomo/{}/session/{n}/turn/{turn_idx}", conv.sample_id),
                content: line,
                concept: "locomo-turn".to_string(),
                group_id: Some(group_id.clone()),
                filters,
                headings: vec![format!("session:{group_id}")],
                links: vec![],
                timestamp: date.clone(),
                doc_length: turn.text.len(),
                author_agent: None,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
    }

    // Production retrieval options: the benchmark measures the production
    // path, so it builds from the same defaults the server uses rather
    // than a hand-duplicated copy. Only LoCoMo-harness-specific choices
    // are overridden below (per-conversation index, pre-attached phrases).
    let mut options = default_production_pipeline_options();
    // Grammar-accepted entity mentions (behood noun-phrase layer) run in one
    // spaCy subprocess per conversation, BEFORE upsert, so each turn-doc
    // carries its session's key phrases into the segment summaries.
    // Any failure degrades to empty phrases and the adaptive path.
    // Enrichment is off: this is a one-shot snapshot builder and the
    // phrases are already attached above; queuing them again would only
    // redo identical extraction in the background.
    options.key_phrase_enrichment = false;
    let extractor_output =
        extract_relations_via_spacy(&rel_turns, std::time::Duration::from_secs(120));
    let mut phrases_by_session: HashMap<String, Vec<lint_ai::KeyPhrase>> = HashMap::new();
    for kp in extractor_output.key_phrases {
        phrases_by_session
            .entry(kp.session_id.clone())
            .or_default()
            .push(lint_ai::KeyPhrase {
                text: kp.text,
                kind: kp.kind,
            });
    }
    for doc in docs.iter_mut() {
        if let Some(group_id) = doc.group_id.clone() {
            if let Some(phrases) = phrases_by_session.get(&group_id) {
                doc.key_phrases = phrases.clone();
            }
        }
    }
    let mut service = MemoryService::in_memory(options);
    for doc in docs {
        service.upsert(doc);
    }
    service.refresh().context("failed to build conv index")?;
    Ok(ConvIndex {
        searcher: Mutex::new(service),
    })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"ok": true}))
}

/// Run one benchmark query through the agent's search entry point:
/// [`MemoryService::search_with_filters`], the same function the MCP `search`
/// tools call, with hits shaped by the shared `search_results` formatter so
/// the reader sees exactly what an agent sees. The per-conversation session
/// id scopes follow-up resolution and the conversational reranker; hits are
/// matched turns only (no full-session expansion).
fn run_search(index: &ConvIndex, conv: &str, query: &str, k: usize) -> Result<Vec<Hit>> {
    let mut filters = BTreeMap::new();
    filters.insert("memory_user_id".to_string(), LOCOMO_USER_ID.to_string());
    let shaped: Value = {
        let mut searcher = index
            .searcher
            .lock()
            .expect("benchmark searcher lock poisoned");
        let results =
            searcher.search_with_filters(query, BENCHMARK_SCOPE, Some(conv), k, &filters)?;
        search_results(&searcher, results)
    };
    let mut out = Vec::new();
    for hit in shaped
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        // The reader is text-only: flatten the agent-facing JSON the way the
        // benchmark always has -- content (date-prefixed and truncated
        // exactly as the agent sees it) with relation evidence inline.
        let content = hit.get("content").and_then(Value::as_str).unwrap_or("");
        let evidence: Vec<&str> = hit
            .get("relation_evidence")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let text = if evidence.is_empty() {
            content.to_string()
        } else {
            format!("[{}] {content}", evidence.join("; "))
        };
        out.push(Hit {
            session_id: hit
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            score: hit.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            text,
        });
    }
    Ok(out)
}

async fn search(
    State(state): State<Arc<AppState>>,
    Query(p): Query<SearchParams>,
) -> Result<Json<SearchResp>, (StatusCode, String)> {
    let conv = state
        .convs
        .get(&p.conv)
        .ok_or((StatusCode::NOT_FOUND, format!("unknown conv {}", p.conv)))?;
    let k = p.k.clamp(1, 20);
    let results = run_search(conv, &p.conv, &p.q, k)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(SearchResp { results }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("loading LoCoMo data...");
    let data = fs::read_to_string(&args.locomo)
        .with_context(|| format!("failed to read {}", args.locomo.display()))?;
    let conversations: Vec<LocomoConversation> =
        serde_json::from_str(&data).context("failed to parse LoCoMo JSON")?;
    eprintln!("building {} conversation snapshots...", conversations.len());
    let mut convs = HashMap::new();
    for conv in &conversations {
        let idx = build_conv_index(conv)
            .with_context(|| format!("failed to index {}", conv.sample_id))?;
        eprintln!("  indexed {}", conv.sample_id);
        convs.insert(conv.sample_id.clone(), Arc::new(idx));
    }
    let state = Arc::new(AppState { convs });
    let app = Router::new()
        .route("/health", get(health))
        .route("/search", get(search))
        .with_state(state);
    let addr: std::net::SocketAddr = args.bind.parse().context("bad --bind")?;
    eprintln!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal per-conversation index built directly (bypassing
    /// `build_conv_index`, which shells out to spaCy) to test the aligned
    /// search path: `search_with_filters` + the shared agent-facing shaping.
    fn test_index() -> ConvIndex {
        let mut options = default_production_pipeline_options();
        options.key_phrase_enrichment = false;
        let mut service = MemoryService::in_memory(options);
        let mut filters = BTreeMap::new();
        filters.insert("memory_user_id".to_string(), LOCOMO_USER_ID.to_string());
        service.upsert(SourceDocument {
            doc_id: "conv-1::session_1::turn0".to_string(),
            source: "locomo/conv-1/session/1/turn/0".to_string(),
            content: "Ted: I visited the Eiffel Tower in Paris".to_string(),
            concept: "locomo-turn".to_string(),
            group_id: Some("conv-1::session_1".to_string()),
            filters,
            headings: vec![],
            links: vec![],
            timestamp: Some("2024-05-01".to_string()),
            doc_length: 41,
            author_agent: None,
            key_phrases: vec![lint_ai::KeyPhrase {
                text: "Eiffel Tower".to_string(),
                kind: "place".to_string(),
            }],
            key_phrase_extraction_hash: String::new(),
        });
        service.refresh().expect("refresh test index");
        ConvIndex {
            searcher: Mutex::new(service),
        }
    }

    #[test]
    fn benchmark_search_uses_agent_entry_point_and_shaping() {
        let index = test_index();
        let hits = run_search(&index, "conv-1", "Eiffel Tower Paris", 5).expect("search");
        assert_eq!(hits.len(), 1, "expected the indexed turn, got {hits:?}");
        let hit = &hits[0];
        // Agent-facing shaping from the shared `search_results` formatter:
        // absolute date prefix and the turn text an agent would see.
        assert!(
            hit.text.contains("[session date: 2024-05-01]"),
            "missing date prefix: {}",
            hit.text
        );
        assert!(hit.text.contains("Eiffel Tower"), "text: {}", hit.text);
        assert_eq!(hit.session_id, "conv-1::session_1");
        assert!(hit.score > 0.0);
    }

    #[test]
    fn benchmark_search_resolves_follow_ups_through_session_state() {
        let index = test_index();
        // The first query is observed into the (scope, session) state, the
        // way the MCP tools observe it; a follow-up phrasing then runs the
        // same resolution + conversational rerank an agent gets.
        let first = run_search(&index, "conv-1", "Eiffel Tower Paris", 5).expect("search");
        assert_eq!(first.len(), 1);
        let follow_up = run_search(&index, "conv-1", "when did he visit it", 5).expect("follow-up");
        assert_eq!(
            follow_up.len(),
            1,
            "follow-up should still retrieve the turn, got {follow_up:?}"
        );
        assert!(follow_up[0].text.contains("Eiffel Tower"));
    }

    #[test]
    fn benchmark_search_unknown_query_returns_no_hits() {
        let index = test_index();
        let hits = run_search(&index, "conv-1", "quantum chromodynamics", 5).expect("search");
        assert!(hits.is_empty(), "unexpected hits: {hits:?}");
    }
}
