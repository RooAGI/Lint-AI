//! Bounded operational telemetry for long-running server processes.
//!
//! This module stores bounded query aggregates and sanitized provider lifecycle
//! metadata. Query text, prompts, tool payloads, user IDs, document IDs, and
//! memory content are either omitted or stored only as bounded redacted previews.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::path::Path;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const BUCKET_WIDTH_MS: u64 = 5_000;
const MAX_BUCKETS: usize = 120;
const QUERY_EVENT_QUEUE_CAPACITY: usize = 1_024;
const LATENCY_BOUNDARIES_MS: [u64; 12] = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1_000, 5_000, 30_000];
const PROVIDER_TELEMETRY_DIR: &str = "provider-telemetry";
const MAX_PROVIDER_TELEMETRY_EVENTS: usize = 500;
const QUERY_TELEMETRY_FILE: &str = "query-telemetry.json";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderLifecycleStatus {
    pub schema_version: u8,
    pub provider: String,
    pub last_event: Option<String>,
    pub last_seen_ms: u64,
    pub events_total: u64,
    pub sessions_started: u64,
    pub sessions_ended: u64,
    pub sessions_active: u64,
    pub retrieval_events: u64,
    pub capture_events: u64,
    #[serde(default)]
    pub events: Vec<ProviderLifecycleEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderLifecycleEvent {
    pub event_id: String,
    pub provider: String,
    pub session_key: String,
    pub event: String,
    pub category: String,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_arguments: Option<String>,
    #[serde(default)]
    pub prompt_preview: Option<String>,
    #[serde(default)]
    pub tool_response_preview: Option<String>,
    #[serde(default)]
    pub stop_response_preview: Option<String>,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub retrieval_query_preview: Option<String>,
    #[serde(default)]
    pub retrieval_latency_ms: Option<u64>,
    #[serde(default)]
    pub retrieved_memory_count: Option<u64>,
    #[serde(default)]
    pub retrieved_memory_details: Option<String>,
}

pub fn record_memory_retrieval(
    project_root: &Path,
    provider: &str,
    session_id: &str,
    query: &str,
    latency_ms: u64,
    memories: &[serde_json::Value],
) -> Result<()> {
    record_provider_lifecycle_event(
        project_root,
        provider,
        session_id,
        "MemoryRetrieved",
        &serde_json::json!({
            "query": query,
            "retrieval_latency_ms": latency_ms,
            "retrieved_memory_count": memories.len() as u64,
            "retrieved_memory_details": memories,
        }),
    )
}

pub fn record_provider_lifecycle_event(
    project_root: &Path,
    provider: &str,
    session_id: &str,
    event_name: &str,
    payload: &serde_json::Value,
) -> Result<()> {
    let root = project_root.join(".lint-ai").join(PROVIDER_TELEMETRY_DIR);
    fs::create_dir_all(&root)?;
    let lock = acquire_telemetry_lock(&root.join(format!("{provider}.lock")))?;
    let path = root.join(format!("{provider}.json"));
    let mut status = if path.exists() {
        serde_json::from_str::<ProviderLifecycleStatus>(&fs::read_to_string(&path)?)
            .unwrap_or_default()
    } else {
        ProviderLifecycleStatus {
            schema_version: 1,
            provider: provider.to_string(),
            ..ProviderLifecycleStatus::default()
        }
    };
    let category = lifecycle_category(event_name);
    let (
        tool_name,
        tool_arguments,
        prompt_preview,
        tool_response_preview,
        stop_response_preview,
        agent_id,
        agent_type,
        turn_id,
        input_tokens,
        output_tokens,
        total_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        retrieval_query_preview,
        retrieval_latency_ms,
        retrieved_memory_count,
        retrieved_memory_details,
    ) = event_details(provider, event_name, payload);
    status.provider = provider.to_string();
    status.last_event = Some(event_name.to_string());
    status.last_seen_ms = now_ms();
    status.events_total = status.events_total.saturating_add(1);
    match category {
        "session" if event_name.eq_ignore_ascii_case("SessionStart") => {
            status.sessions_started = status.sessions_started.saturating_add(1);
            status.sessions_active = status.sessions_active.saturating_add(1);
        }
        "session" if event_name.eq_ignore_ascii_case("SessionEnd") => {
            status.sessions_ended = status.sessions_ended.saturating_add(1);
            status.sessions_active = status.sessions_active.saturating_sub(1);
        }
        "retrieval" => status.retrieval_events = status.retrieval_events.saturating_add(1),
        "capture" => status.capture_events = status.capture_events.saturating_add(1),
        _ => {}
    }
    status.events.push(ProviderLifecycleEvent {
        event_id: format!("{}-{}", status.last_seen_ms, status.events_total),
        provider: provider.to_string(),
        session_key: opaque_session_key(session_id),
        event: event_name.to_string(),
        category: category.to_string(),
        timestamp_ms: status.last_seen_ms,
        agent_id,
        agent_type,
        turn_id,
        tool_name,
        tool_arguments,
        prompt_preview,
        tool_response_preview,
        stop_response_preview,
        input_tokens,
        output_tokens,
        total_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        retrieval_query_preview,
        retrieval_latency_ms,
        retrieved_memory_count,
        retrieved_memory_details,
    });
    if status.events.len() > MAX_PROVIDER_TELEMETRY_EVENTS {
        let keep_from = status.events.len() - MAX_PROVIDER_TELEMETRY_EVENTS;
        status.events.drain(..keep_from);
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&status)?)?;
    fs::rename(temporary, path)?;
    drop(lock);
    Ok(())
}

/// Extract and normalize provider usage into the telemetry schema.
///
/// Provider adapters may pass either a lifecycle payload or a transcript line;
/// keeping this here ensures both paths use identical accounting semantics.
pub fn normalize_usage(provider: &str, value: &serde_json::Value) -> Option<serde_json::Value> {
    fn visit(provider: &str, value: &serde_json::Value) -> Option<serde_json::Value> {
        match value {
            serde_json::Value::Object(object) => {
                for key in [
                    "usage",
                    "token_usage",
                    "tokenUsage",
                    "last_token_usage",
                    "total_token_usage",
                ] {
                    if let Some(candidate) = object.get(key) {
                        if let Some(normalized) = normalize_object(provider, candidate, object) {
                            return Some(normalized);
                        }
                    }
                }
                object.values().find_map(|value| visit(provider, value))
            }
            serde_json::Value::Array(values) => {
                values.iter().find_map(|value| visit(provider, value))
            }
            _ => None,
        }
    }

    visit(provider, value)
}

fn normalize_object(
    provider: &str,
    value: &serde_json::Value,
    parent: &serde_json::Map<String, serde_json::Value>,
) -> Option<serde_json::Value> {
    let object = value.as_object()?;
    let number = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| object.get(*key).and_then(serde_json::Value::as_u64))
            .or_else(|| {
                keys.iter()
                    .find_map(|key| parent.get(*key).and_then(serde_json::Value::as_u64))
            })
    };
    let raw_input = number(&["input_tokens", "inputTokens", "prompt_tokens"]);
    let output = number(&["output_tokens", "outputTokens", "completion_tokens"]);
    let cache_creation = number(&[
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
        "cache_write_input_tokens",
        "cacheWriteInputTokens",
    ]);
    let cache_read = number(&[
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cached_input_tokens",
    ]);
    let input_includes_cache = provider.eq_ignore_ascii_case("codex")
        && (object.contains_key("cached_input_tokens")
            || object.contains_key("cache_write_input_tokens")
            || parent.contains_key("cached_input_tokens")
            || parent.contains_key("cache_write_input_tokens"));
    let input = raw_input.map(|value| {
        if input_includes_cache {
            value.saturating_sub(
                cache_read
                    .unwrap_or(0)
                    .saturating_add(cache_creation.unwrap_or(0)),
            )
        } else {
            value
        }
    });
    let total = number(&["total_tokens", "totalTokens"])
        .or_else(|| raw_input.map(|value| value + output.unwrap_or(0)));
    if input.is_none()
        && output.is_none()
        && cache_creation.is_none()
        && cache_read.is_none()
        && total.is_none()
    {
        return None;
    }
    Some(serde_json::json!({
        "input_tokens": input,
        "output_tokens": output,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "total_tokens": total,
        "source": "hook-payload"
    }))
}

fn event_details(
    provider: &str,
    event_name: &str,
    payload: &serde_json::Value,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<String>,
    Option<u64>,
    Option<u64>,
    Option<String>,
) {
    let object = payload.as_object();
    let value_for =
        |keys: &[&str]| object.and_then(|object| keys.iter().find_map(|key| object.get(*key)));
    let string_for = |keys: &[&str]| value_for(keys).and_then(|value| value.as_str());
    let identity_for = |keys: &[&str]| string_for(keys).map(|value| truncate_text(value, 160));
    let tool_event = matches!(
        event_name,
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "BeforeTool"
    );
    let prompt_event = matches!(
        event_name,
        "UserPromptSubmit" | "UserPromptExpansion" | "BeforeAgent"
    );
    let response_event = matches!(
        event_name,
        "PostToolUse" | "PostToolUseFailure" | "AfterTool"
    );
    let stop_event = matches!(
        event_name,
        "Stop" | "AfterAgent" | "SessionEnd" | "SubagentStop"
    );
    let tool_name = tool_event
        .then(|| string_for(&["tool_name", "toolName", "name"]))
        .flatten()
        .map(|value| truncate_text(value, 160));
    let tool_arguments = tool_event
        .then(|| value_for(&["tool_input", "toolInput", "arguments", "args"]))
        .flatten()
        .and_then(preview_value);
    let prompt_preview = prompt_event
        .then(|| value_for(&["prompt", "user_prompt", "text"]))
        .flatten()
        .and_then(preview_value);
    let tool_response_preview = response_event
        .then(|| value_for(&["tool_response", "toolResponse", "tool_output", "response"]))
        .flatten()
        .and_then(preview_value);
    let stop_response_preview = stop_event
        .then(|| value_for(&["final_response", "response", "output", "reason"]))
        .flatten()
        .and_then(preview_value);
    let retrieval_query_preview = (event_name == "MemoryRetrieved")
        .then(|| string_for(&["query"]))
        .flatten()
        .map(|value| truncate_text(value, 512));
    let retrieval_latency_ms = (event_name == "MemoryRetrieved")
        .then(|| value_for(&["retrieval_latency_ms"]).and_then(serde_json::Value::as_u64))
        .flatten();
    let retrieved_memory_count = (event_name == "MemoryRetrieved")
        .then(|| value_for(&["retrieved_memory_count"]).and_then(serde_json::Value::as_u64))
        .flatten();
    let retrieved_memory_details = (event_name == "MemoryRetrieved")
        .then(|| value_for(&["retrieved_memory_details"]))
        .flatten()
        .and_then(preview_value);
    let usage = normalize_usage(provider, payload);
    let usage = usage.as_ref().and_then(serde_json::Value::as_object);
    let usage_number = |keys: &[&str]| {
        usage
            .and_then(|usage| {
                keys.iter()
                    .find_map(|key| usage.get(*key).and_then(serde_json::Value::as_u64))
            })
            .or_else(|| {
                object.and_then(|object| {
                    keys.iter()
                        .find_map(|key| object.get(*key).and_then(serde_json::Value::as_u64))
                })
            })
    };
    (
        tool_name,
        tool_arguments,
        prompt_preview,
        tool_response_preview,
        stop_response_preview,
        identity_for(&["agent_id", "agentId", "subagent_id", "subagentId"]),
        identity_for(&["agent_type", "agentType", "subagent_type", "subagentType"]),
        identity_for(&["turn_id", "turnId"]),
        usage_number(&["input_tokens", "inputTokens", "prompt_tokens"]),
        usage_number(&["output_tokens", "outputTokens", "completion_tokens"]),
        usage_number(&["total_tokens", "totalTokens"]),
        usage_number(&["cache_creation_input_tokens"]),
        usage_number(&["cache_read_input_tokens"]),
        retrieval_query_preview,
        retrieval_latency_ms,
        retrieved_memory_count,
        retrieved_memory_details,
    )
}

fn preview_value(value: &serde_json::Value) -> Option<String> {
    let sanitized = sanitize_tool_value(value, None, 0);
    let rendered = match sanitized {
        serde_json::Value::String(value) => value,
        value => serde_json::to_string(&value).ok()?,
    };
    Some(truncate_text(&rendered, 2_048))
}

fn sanitize_tool_value(
    value: &serde_json::Value,
    key: Option<&str>,
    depth: usize,
) -> serde_json::Value {
    if depth > 6 {
        return serde_json::Value::String("[MAX_DEPTH]".to_string());
    }
    if key.is_some_and(is_sensitive_key) {
        return serde_json::Value::String("[REDACTED]".to_string());
    }
    match value {
        serde_json::Value::Object(object) => serde_json::Value::Object(
            object
                .iter()
                .take(40)
                .map(|(key, value)| {
                    (
                        key.clone(),
                        sanitize_tool_value(value, Some(key), depth + 1),
                    )
                })
                .collect(),
        ),
        serde_json::Value::Array(values) => serde_json::Value::Array(
            values
                .iter()
                .take(20)
                .map(|value| sanitize_tool_value(value, None, depth + 1))
                .collect(),
        ),
        serde_json::Value::String(value) => {
            if looks_like_credential(value) {
                serde_json::Value::String("[REDACTED]".to_string())
            } else {
                serde_json::Value::String(truncate_text(value, 512))
            }
        }
        other => other.clone(),
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn looks_like_credential(value: &str) -> bool {
    ["sk-", "sk_live_", "ghp_", "gho_", "xoxb-", "Bearer "]
        .iter()
        .any(|prefix| value.contains(prefix))
}

fn truncate_text(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes.saturating_sub(3);
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}...", &value[..end])
}

pub fn provider_lifecycle_status(
    project_root: &Path,
    provider: &str,
) -> Result<Option<ProviderLifecycleStatus>> {
    let path = project_root
        .join(".lint-ai")
        .join(PROVIDER_TELEMETRY_DIR)
        .join(format!("{provider}.json"));
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs::read_to_string(path)?)?))
}

fn opaque_session_key(session_id: &str) -> String {
    if session_id.trim().is_empty() {
        return "unknown".to_string();
    }
    let digest = Sha256::digest(session_id.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn lifecycle_category(event_name: &str) -> &'static str {
    match event_name {
        "MemoryRetrieved" => "retrieval",
        "SessionStart" | "SessionEnd" | "SubagentStart" | "SubagentStop" => "session",
        "PreCompact" | "PostCompact" | "PreCompress" => "compaction",
        "Stop" | "AfterAgent" => "capture",
        "UserPromptSubmit"
        | "UserPromptExpansion"
        | "BeforeAgent"
        | "BeforeModel"
        | "BeforeToolSelection"
        | "PreToolUse"
        | "BeforeTool" => "retrieval",
        _ => "lifecycle",
    }
}

struct TelemetryLock {
    path: std::path::PathBuf,
}

fn acquire_telemetry_lock(path: &Path) -> Result<TelemetryLock> {
    for _ in 0..500 {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(_) => {
                return Ok(TelemetryLock {
                    path: path.to_path_buf(),
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("timed out waiting for provider telemetry lock")
}

impl Drop for TelemetryLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct QueryBucket {
    start_ms: u64,
    requests: u64,
    errors: u64,
    empty_results: u64,
    latency_counts: [u64; LATENCY_BOUNDARIES_MS.len() + 1],
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedQueryTelemetry {
    started_at_ms: u64,
    #[serde(default)]
    buckets: VecDeque<QueryBucket>,
}

#[derive(Default)]
struct TelemetryState {
    buckets: VecDeque<QueryBucket>,
}

#[derive(Clone, Copy)]
struct QueryEvent {
    duration_ms: u64,
    error: bool,
    empty_results: bool,
}

#[derive(Clone)]
pub struct OperationalTelemetry {
    started_at_ms: u64,
    state: std::sync::Arc<Mutex<TelemetryState>>,
    query_events: mpsc::SyncSender<QueryEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuerySeriesPoint {
    pub start_ms: u64,
    pub requests: u64,
    pub errors: u64,
    pub empty_results: u64,
    pub requests_per_second: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuerySummary {
    pub requests: u64,
    pub errors: u64,
    pub empty_results: u64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub empty_result_rate: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetrySnapshot {
    pub started_at_ms: u64,
    pub window_seconds: u64,
    pub query_summary: QuerySummary,
    pub query_series: Vec<QuerySeriesPoint>,
}

/// Record an aggregate for a query executed by an IndexStore API process.
/// Only bounded counters and latency buckets are persisted; query text is not.
pub fn record_project_query(
    project_root: &Path,
    duration_ms: u64,
    error: bool,
    empty_results: bool,
) -> Result<()> {
    let root = project_root.join(".lint-ai");
    fs::create_dir_all(&root)?;
    let lock = acquire_telemetry_lock(&root.join("query-telemetry.lock"))?;
    let path = root.join(QUERY_TELEMETRY_FILE);
    let mut ledger = if path.exists() {
        serde_json::from_str::<PersistedQueryTelemetry>(&fs::read_to_string(&path)?)
            .unwrap_or_default()
    } else {
        PersistedQueryTelemetry {
            started_at_ms: now_ms(),
            ..PersistedQueryTelemetry::default()
        }
    };
    let now = now_ms();
    let bucket_start = now - (now % BUCKET_WIDTH_MS);
    if ledger
        .buckets
        .back()
        .is_none_or(|bucket| bucket.start_ms != bucket_start)
    {
        ledger.buckets.push_back(QueryBucket {
            start_ms: bucket_start,
            ..QueryBucket::default()
        });
        while ledger.buckets.len() > MAX_BUCKETS {
            ledger.buckets.pop_front();
        }
    }
    if let Some(bucket) = ledger.buckets.back_mut() {
        bucket.requests = bucket.requests.saturating_add(1);
        bucket.errors = bucket.errors.saturating_add(u64::from(error));
        bucket.empty_results = bucket
            .empty_results
            .saturating_add(u64::from(empty_results));
        let latency_bucket = LATENCY_BOUNDARIES_MS
            .iter()
            .position(|boundary| duration_ms <= *boundary)
            .unwrap_or(LATENCY_BOUNDARIES_MS.len());
        bucket.latency_counts[latency_bucket] =
            bucket.latency_counts[latency_bucket].saturating_add(1);
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&ledger)?)?;
    fs::rename(temporary, path)?;
    drop(lock);
    Ok(())
}

pub fn project_query_snapshot(project_root: &Path) -> Result<Option<TelemetrySnapshot>> {
    let path = project_root.join(".lint-ai").join(QUERY_TELEMETRY_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let ledger = serde_json::from_str::<PersistedQueryTelemetry>(&fs::read_to_string(path)?)?;
    let buckets = ledger.buckets.into_iter().collect::<Vec<_>>();
    let query_series = buckets.iter().map(series_point).collect::<Vec<_>>();
    let mut aggregate = QueryBucket::default();
    for bucket in &buckets {
        aggregate.requests = aggregate.requests.saturating_add(bucket.requests);
        aggregate.errors = aggregate.errors.saturating_add(bucket.errors);
        aggregate.empty_results = aggregate.empty_results.saturating_add(bucket.empty_results);
        for (index, count) in bucket.latency_counts.iter().enumerate() {
            aggregate.latency_counts[index] =
                aggregate.latency_counts[index].saturating_add(*count);
        }
    }
    let window_seconds = (buckets.len() as u64 * BUCKET_WIDTH_MS) / 1_000;
    Ok(Some(TelemetrySnapshot {
        started_at_ms: ledger.started_at_ms,
        window_seconds,
        query_summary: summary(&aggregate, window_seconds),
        query_series,
    }))
}

impl OperationalTelemetry {
    pub fn new() -> Self {
        let state = std::sync::Arc::new(Mutex::new(TelemetryState::default()));
        let (query_events, receiver) = mpsc::sync_channel::<QueryEvent>(QUERY_EVENT_QUEUE_CAPACITY);
        let worker_state = std::sync::Arc::clone(&state);
        thread::Builder::new()
            .name("lint-ai-query-telemetry".to_string())
            .spawn(move || {
                while let Ok(event) = receiver.recv() {
                    record_query_in_state(
                        &worker_state,
                        event.duration_ms,
                        event.error,
                        event.empty_results,
                    );
                }
            })
            .expect("query telemetry worker should start");
        Self {
            started_at_ms: now_ms(),
            state,
            query_events,
        }
    }

    pub fn record_query(&self, duration_ms: u64, error: bool, empty_results: bool) {
        let _ = self.query_events.try_send(QueryEvent {
            duration_ms,
            error,
            empty_results,
        });
    }
}

fn record_query_in_state(
    state: &Mutex<TelemetryState>,
    duration_ms: u64,
    error: bool,
    empty_results: bool,
) {
    let now = now_ms();
    let bucket_start = now - (now % BUCKET_WIDTH_MS);
    let Ok(mut state) = state.lock() else {
        return;
    };
    if state
        .buckets
        .back()
        .is_none_or(|bucket| bucket.start_ms != bucket_start)
    {
        state.buckets.push_back(QueryBucket {
            start_ms: bucket_start,
            ..QueryBucket::default()
        });
        while state.buckets.len() > MAX_BUCKETS {
            state.buckets.pop_front();
        }
    }
    let Some(bucket) = state.buckets.back_mut() else {
        return;
    };
    bucket.requests = bucket.requests.saturating_add(1);
    bucket.errors = bucket.errors.saturating_add(u64::from(error));
    bucket.empty_results = bucket
        .empty_results
        .saturating_add(u64::from(empty_results));
    let latency_bucket = LATENCY_BOUNDARIES_MS
        .iter()
        .position(|boundary| duration_ms <= *boundary)
        .unwrap_or(LATENCY_BOUNDARIES_MS.len());
    bucket.latency_counts[latency_bucket] = bucket.latency_counts[latency_bucket].saturating_add(1);
}

impl OperationalTelemetry {
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let buckets = self
            .state
            .lock()
            .map(|state| state.buckets.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let query_series = buckets.iter().map(series_point).collect::<Vec<_>>();
        let mut aggregate = QueryBucket::default();
        for bucket in &buckets {
            aggregate.requests = aggregate.requests.saturating_add(bucket.requests);
            aggregate.errors = aggregate.errors.saturating_add(bucket.errors);
            aggregate.empty_results = aggregate.empty_results.saturating_add(bucket.empty_results);
            for (index, count) in bucket.latency_counts.iter().enumerate() {
                aggregate.latency_counts[index] =
                    aggregate.latency_counts[index].saturating_add(*count);
            }
        }
        let window_seconds = (buckets.len() as u64 * BUCKET_WIDTH_MS) / 1_000;
        TelemetrySnapshot {
            started_at_ms: self.started_at_ms,
            window_seconds,
            query_summary: summary(&aggregate, window_seconds),
            query_series,
        }
    }
}

impl Default for OperationalTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

fn series_point(bucket: &QueryBucket) -> QuerySeriesPoint {
    let summary = summary(bucket, BUCKET_WIDTH_MS / 1_000);
    QuerySeriesPoint {
        start_ms: bucket.start_ms,
        requests: bucket.requests,
        errors: bucket.errors,
        empty_results: bucket.empty_results,
        requests_per_second: summary.requests_per_second,
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
    }
}

fn summary(bucket: &QueryBucket, window_seconds: u64) -> QuerySummary {
    let requests = bucket.requests;
    let seconds = window_seconds.max(1) as f64;
    QuerySummary {
        requests,
        errors: bucket.errors,
        empty_results: bucket.empty_results,
        requests_per_second: requests as f64 / seconds,
        error_rate: ratio(bucket.errors, requests),
        empty_result_rate: ratio(bucket.empty_results, requests),
        p50_ms: percentile(&bucket.latency_counts, requests, 0.50),
        p95_ms: percentile(&bucket.latency_counts, requests, 0.95),
    }
}

fn percentile(counts: &[u64], total: u64, quantile: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let target = ((total as f64 * quantile).ceil() as u64).max(1);
    let mut seen: u64 = 0;
    for (index, count) in counts.iter().enumerate() {
        seen = seen.saturating_add(*count);
        if seen >= target {
            return if index < LATENCY_BOUNDARIES_MS.len() {
                LATENCY_BOUNDARIES_MS[index] as f64
            } else {
                (LATENCY_BOUNDARIES_MS[LATENCY_BOUNDARIES_MS.len() - 1] * 2) as f64
            };
        }
    }
    0.0
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_has_zero_rates() {
        let snapshot = OperationalTelemetry::new().snapshot();
        assert_eq!(snapshot.query_summary.requests, 0);
        assert_eq!(snapshot.query_summary.requests_per_second, 0.0);
    }

    #[test]
    fn records_query_counts_and_latency_buckets() {
        let telemetry = OperationalTelemetry::new();
        telemetry.record_query(8, false, true);
        telemetry.record_query(100, true, false);
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let snapshot = loop {
            let snapshot = telemetry.snapshot();
            if snapshot.query_summary.requests == 2 || std::time::Instant::now() >= deadline {
                break snapshot;
            }
            thread::yield_now();
        };
        assert_eq!(snapshot.query_summary.requests, 2);
        assert_eq!(snapshot.query_summary.errors, 1);
        assert_eq!(snapshot.query_summary.empty_results, 1);
        assert!(snapshot.query_summary.p50_ms > 0.0);
        assert!(!snapshot.query_series.is_empty());
    }

    #[test]
    fn reveals_bug_record_query_waits_for_the_telemetry_mutex() {
        let telemetry = OperationalTelemetry::new();
        let guard = telemetry.state.lock().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker_telemetry = telemetry.clone();
        thread::spawn(move || {
            worker_telemetry.record_query(8, false, false);
            done_tx.send(()).unwrap();
        });

        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_ok(),
            "record_query blocked behind the telemetry mutex"
        );
        drop(guard);
    }

    #[test]
    fn provider_lifecycle_ledger_is_bounded_and_opaque() {
        let root = std::env::temp_dir().join(format!("lint-ai-provider-telemetry-{}", now_ms()));
        record_provider_lifecycle_event(
            &root,
            "codex",
            "private-session-id",
            "SessionStart",
            &serde_json::json!({}),
        )
        .unwrap();
        record_provider_lifecycle_event(
            &root,
            "codex",
            "private-session-id",
            "PreToolUse",
            &serde_json::json!({
                "tool_name": "read_file",
                "tool_input": {"path": "README.md", "api_key": "do-not-store"}
            }),
        )
        .unwrap();
        record_provider_lifecycle_event(
            &root,
            "codex",
            "private-session-id",
            "UserPromptSubmit",
            &serde_json::json!({"prompt": "show the current index status"}),
        )
        .unwrap();
        record_provider_lifecycle_event(
            &root,
            "codex",
            "private-session-id",
            "PostToolUse",
            &serde_json::json!({
                "tool_name": "read_file",
                "tool_response": {"content": "index is ready", "token": "sk-live-secret"}
            }),
        )
        .unwrap();
        record_provider_lifecycle_event(
            &root,
            "codex",
            "private-session-id",
            "Stop",
            &serde_json::json!({"final_response": "The index is ready.", "reason": "completed"}),
        )
        .unwrap();
        let status = provider_lifecycle_status(&root, "codex").unwrap().unwrap();
        assert_eq!(status.events_total, 5);
        assert_eq!(status.sessions_started, 1);
        assert_eq!(status.retrieval_events, 2);
        assert_eq!(status.events.len(), 5);
        assert_eq!(status.events[1].tool_name.as_deref(), Some("read_file"));
        assert!(status.events[1]
            .tool_arguments
            .as_deref()
            .unwrap()
            .contains("REDACTED"));
        assert_eq!(
            status.events[2].prompt_preview.as_deref(),
            Some("show the current index status")
        );
        assert!(status.events[3]
            .tool_response_preview
            .as_deref()
            .unwrap()
            .contains("REDACTED"));
        assert_eq!(
            status.events[4].stop_response_preview.as_deref(),
            Some("The index is ready.")
        );
        assert_ne!(status.events[0].session_key, "private-session-id");
        assert!(!serde_json::to_string(&status)
            .unwrap()
            .contains("private-session-id"));
        fs::remove_dir_all(root).unwrap();
    }
}
