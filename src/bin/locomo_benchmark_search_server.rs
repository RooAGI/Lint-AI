//! Long-lived Lint-AI search server over LoCoMo conversations (pillar-4 agent tool).
//!
//! Builds one in-memory [`MemorySearchService`] per conversation at startup
//! (same pipeline options as `locomo_benchmark.rs`, segmented layout with the
//! gated-local routing strategy), then serves:
//!   GET /health                        -> {"ok":true}
//!   GET /search?conv=<sample_id>&q=<query>&k=<n>
//!       -> {"results":[{"session_id":"...","score":1.23,"text":"..."}]}
//!
//! Searches run through the production [`MemoryService`] query path with a
//! per-conversation `session_id`, so the conversational reranker fires on
//! follow-up queries exactly as it does in production.
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
use lint_ai::memory_api::{MemorySearchService, MemoryService, SearchRequest};
use lint_ai::segments::SegmentRoutingStrategy;
use lint_ai::{IndexStore, MemoryIndexLayout, PipelineOptions, SourceDocument};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

/// User id stamped on every LoCoMo document; the production query path
/// filters on it, mirroring how `AddRequest` stamps real writes.
const LOCOMO_USER_ID: &str = "locomo-agent";

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
}

struct ConvIndex {
    searcher: MemorySearchService,
    session_text: HashMap<String, String>,
    doc_group: HashMap<String, String>,
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
    let mut session_text: HashMap<String, String> = HashMap::new();
    let mut doc_group: HashMap<String, String> = HashMap::new();
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
        let mut lines = Vec::new();
        for (turn_idx, turn) in turns.iter().enumerate() {
            let line = format!("{}: {}", turn.speaker, turn.text);
            lines.push(line.clone());
            let mut filters = BTreeMap::new();
            filters.insert("memory_user_id".to_string(), LOCOMO_USER_ID.to_string());
            let doc_id = format!("{group_id}::turn{turn_idx}");
            doc_group.insert(doc_id.clone(), group_id.clone());
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
            });
        }
        session_text.insert(group_id, lines.join("\n"));
    }

    // Segmented layout with the gated-local routing strategy (the settled
    // default), top_n=5 matching the retrieval benchmark.
    let mut options = PipelineOptions::default();
    options.memory_index_layout = MemoryIndexLayout::Segmented {
        query_top_n: 5,
        routing_strategy: SegmentRoutingStrategy::TypedEvidenceMultiplicative,
    };
    let mut service = MemoryService::in_memory(options);
    for doc in docs {
        service.upsert(doc);
    }
    service.refresh().context("failed to build conv index")?;
    let searcher = service.published_search();
    Ok(ConvIndex {
        searcher,
        session_text,
        doc_group,
    })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"ok": true}))
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
    // Per-conversation session id: the production path resolves follow-up
    // phrasing against prior observed queries and fires the conversational
    // reranker when the query classifies as a follow-up. Over-fetch docs,
    // then collapse to sessions in rank order.
    let request = SearchRequest {
        query: p.q.clone(),
        options: None,
        user_id: LOCOMO_USER_ID.to_string(),
        top_k: k * 6,
        session_id: Some(p.conv.clone()),
    };
    let response = conv
        .searcher
        .search(request)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for r in response.data {
        let gid = conv
            .doc_group
            .get(&r.id)
            .cloned()
            .unwrap_or_else(|| r.id.clone());
        if !seen.insert(gid.clone()) {
            continue;
        }
        if let Some(text) = conv.session_text.get(&gid) {
            out.push(Hit {
                session_id: gid,
                score: r.score,
                text: text.clone(),
            });
        }
        if out.len() >= k {
            break;
        }
    }
    Ok(Json(SearchResp { results: out }))
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
