use axum::{
    error_handling::HandleErrorLayer,
    extract::{Extension, Json, Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use clap::Parser;
use jsonwebtoken::{decode, DecodingKey, Validation};
use lint_ai::memory_api::{
    AddRequest, DeleteRequest, GetRequest, ListRequest, MemorySearchService, MemoryService,
    SearchRequest, SupersedeRequest, UpdateRequest,
};
use lint_ai::segments::SegmentRoutingStrategy;
use lint_ai::telemetry::{
    project_query_snapshot, provider_lifecycle_status, OperationalTelemetry,
    ProviderLifecycleEvent, TelemetrySnapshot,
};
use lint_ai::{IndexStoreInspection, MemoryIndexLayout, PipelineOptions};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tower::{BoxError, ServiceBuilder};

const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_CONCURRENT_REQUESTS: usize = 128;

#[derive(Debug, Parser)]
#[command(name = "lint-ai-server", about = "Memory Add/Search server backed by Lint-AI", version = env!("CARGO_PKG_VERSION"))]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long)]
    server_token: Option<String>,
    #[arg(long)]
    tenant_id: Option<String>,
    #[arg(long)]
    allow_unauthenticated: bool,
    /// Enable adaptive segmented routing up to this many segments.
    #[arg(long)]
    adaptive_segment_max_n: Option<usize>,
    /// Use the global single-index layout for controlled comparisons.
    #[arg(long)]
    single_index: bool,
    /// Query every segmented shard (global segmented comparison mode).
    #[arg(long)]
    global_index: bool,
    /// Fuse the corpus-wide global arm with the routed arm
    /// (higher recall, higher latency; off by default).
    #[arg(long)]
    fuse_global: bool,
    /// Disable the two-stage conversational rerank for session follow-ups.
    #[arg(long)]
    no_conversational_rerank: bool,
    /// Project root containing provider hook telemetry under `.lint-ai`.
    #[arg(long)]
    project_root: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<MemoryService>>,
    published_search: Arc<RwLock<MemorySearchService>>,
    writer_gate: Arc<tokio::sync::Mutex<()>>,
    token: Option<Arc<str>>,
    jwt_secret: Option<Arc<str>>,
    tenant_id: Option<Arc<str>>,
    telemetry: OperationalTelemetry,
    project_root: PathBuf,
}

#[derive(serde::Serialize)]
struct DashboardStatus {
    status: &'static str,
    version: &'static str,
    uptime_seconds: u64,
    index: DashboardIndexStatus,
    telemetry: TelemetrySnapshot,
    integrations: Vec<DashboardIntegrationStatus>,
}

#[derive(serde::Serialize)]
struct DashboardIndexStatus {
    source_document_count: usize,
    record_count: usize,
    dirty: bool,
    store_revision: u64,
    snapshot_revision: u64,
    revision_lag: u64,
    snapshot: Option<DashboardSnapshotStatus>,
}

#[derive(serde::Serialize)]
struct DashboardSnapshotStatus {
    layout: String,
    segment_count: usize,
    global_document_count: usize,
    segments: Vec<DashboardSegmentStatus>,
}

#[derive(serde::Serialize)]
struct DashboardSegmentStatus {
    segment_id: String,
    document_count: usize,
    profile_term_count: usize,
    profile_entity_count: usize,
    profile_topic_count: usize,
    profile_local_memory_count: usize,
}

#[derive(serde::Serialize)]
struct DashboardIntegrationStatus {
    provider: &'static str,
    #[serde(skip_serializing)]
    recording_provider: &'static str,
    compiled: bool,
    state: &'static str,
    note: &'static str,
    last_event: Option<String>,
    last_seen_ms: Option<u64>,
    events_total: u64,
    sessions_started: u64,
    sessions_ended: u64,
    sessions_active: u64,
    retrieval_events: u64,
    capture_events: u64,
    indexes: Vec<DashboardProviderIndex>,
    events: Vec<ProviderLifecycleEvent>,
}

#[derive(serde::Serialize)]
struct DashboardProviderIndex {
    name: String,
    role: String,
    source_document_count: usize,
    record_count: usize,
    dirty: bool,
    store_revision: u64,
    snapshot_revision: u64,
    snapshot: Option<DashboardSnapshotStatus>,
}

#[derive(serde::Serialize)]
struct DashboardSession {
    provider: String,
    session_key: String,
    event_count: usize,
    first_seen_ms: u64,
    last_seen_ms: u64,
    last_event: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    args.server_token = args
        .server_token
        .or_else(|| std::env::var("SERVER_TOKEN").ok());
    args.tenant_id = args
        .tenant_id
        .or_else(|| std::env::var("SERVER_TENANT_ID").ok());
    if args.adaptive_segment_max_n.is_none() {
        args.adaptive_segment_max_n = std::env::var("ADAPTIVE_SEGMENT_MAX_N")
            .ok()
            .map(|value| {
                value.parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("ADAPTIVE_SEGMENT_MAX_N must be a positive integer")
                })
            })
            .transpose()?;
    }
    let jwt_secret = normalize_secret(std::env::var("JWT_SECRET").ok());
    ensure_loopback_bind(&args.bind)?;
    let options = memory_pipeline_options(
        args.adaptive_segment_max_n,
        args.single_index,
        args.global_index,
        args.fuse_global,
        !args.no_conversational_rerank,
    );
    let project_root = args
        .project_root
        .or_else(|| std::env::var_os("LINT_AI_PROJECT_ROOT").map(PathBuf::from))
        .unwrap_or(std::env::current_dir()?)
        .canonicalize()?;
    let discovered_indexes = discover_index_paths(&project_root);
    let service = match args.index {
        Some(path) => MemoryService::at_path(&path, options)?,
        None => discovered_indexes
            .iter()
            .find(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("workspace-memory")
            })
            .cloned()
            // Keep existing installations usable during their migration. New
            // provider MCP servers create workspace-memory instead.
            .or_else(|| {
                discovered_indexes
                    .iter()
                    .find(|path| {
                        path.file_name().and_then(|name| name.to_str()) == Some("codex-mcp-index")
                    })
                    .cloned()
            })
            .or_else(|| discovered_indexes.first().cloned())
            .map(|path| MemoryService::at_path(&path, options.clone()))
            .transpose()?
            .unwrap_or_else(|| MemoryService::in_memory(options)),
    };
    let published_search = service.published_search();
    let state = AppState {
        service: Arc::new(RwLock::new(service)),
        published_search: Arc::new(RwLock::new(published_search)),
        writer_gate: Arc::new(tokio::sync::Mutex::new(())),
        token: args
            .server_token
            .map(|s| Arc::<str>::from(s.trim().to_owned())),
        jwt_secret: jwt_secret.map(Arc::<str>::from),
        tenant_id: args
            .tenant_id
            .map(|s| Arc::<str>::from(s.trim().to_owned())),
        telemetry: OperationalTelemetry::new(),
        project_root,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/dashboard", get(dashboard))
        .route("/dashboard/app.js", get(dashboard_app))
        .route("/dashboard/event-detail.js", get(dashboard_event_detail))
        .route("/dashboard/styles.css", get(dashboard_styles))
        .route("/dashboard/rooagi-logo.png", get(dashboard_logo))
        .route("/api/status", get(api_status))
        .route("/api/timeseries", get(api_timeseries))
        .route("/api/integrations", get(api_integrations))
        .route("/api/sessions", get(api_sessions))
        .route("/api/sessions/:session_key/events", get(api_session_events))
        .route("/api/events", get(api_events))
        .route("/api/metrics", get(api_metrics))
        .route("/metrics", get(metrics))
        .route("/add", post(add))
        .route("/add/batch", post(add_batch))
        .route("/search", post(search))
        .route("/delete", post(delete))
        .route("/supersede", post(supersede))
        .route("/expire", post(expire))
        .route("/v1/memories", get(list_memories).post(add))
        .route(
            "/v1/memories/:memory_id",
            get(get_memory).patch(update_memory).delete(delete_memory),
        )
        .route("/v1/memories/search", post(search))
        .route("/v1/memories/refresh", post(refresh_memories))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|_: BoxError| async {
                    (
                        StatusCode::REQUEST_TIMEOUT,
                        Json(json!({"detail": "request timed out"})),
                    )
                }))
                .layer(tower::timeout::TimeoutLayer::new(Duration::from_secs(30)))
                .layer(tower::limit::ConcurrencyLimitLayer::new(
                    MAX_CONCURRENT_REQUESTS,
                )),
        )
        .layer(middleware::from_fn_with_state(state.clone(), authorize))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    eprintln!("Lint-AI server listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}

fn memory_pipeline_options(
    adaptive_segment_max_n: Option<usize>,
    single_index: bool,
    global_index: bool,
    fuse_global: bool,
    conversational_rerank: bool,
) -> PipelineOptions {
    if single_index {
        return PipelineOptions {
            memory_index_layout: MemoryIndexLayout::Single,
            ..PipelineOptions::default()
        };
    }
    let layout = if global_index {
        MemoryIndexLayout::Segmented {
            query_top_n: usize::MAX,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        }
    } else {
        adaptive_segment_max_n
            .filter(|max_n| *max_n > 3)
            .map(|max_n| MemoryIndexLayout::AdaptiveSegmented {
                query_top_n: 3,
                max_query_n: max_n,
                routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
            })
            .unwrap_or(MemoryIndexLayout::Segmented {
                query_top_n: 3,
                routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
            })
    };
    PipelineOptions {
        memory_index_layout: layout,
        fuse_global_arm: fuse_global,
        conversational_rerank,
        ..PipelineOptions::default()
    }
}

fn normalize_secret(value: Option<String>) -> Option<String> {
    value
        .map(|secret| secret.trim().to_owned())
        .filter(|secret| !secret.is_empty())
}

async fn authorize(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    let path = request.uri().path();
    if path == "/health" || path == "/dashboard" || path.starts_with("/dashboard/") {
        return next.run(request).await;
    }
    if let Some(secret) = state.jwt_secret.as_deref() {
        let supplied = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = supplied.strip_prefix("Bearer ").unwrap_or("");
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.validate_exp = true;
        let claims = match decode::<Claims>(
            token,
            &DecodingKey::from_secret(secret.as_bytes()),
            &validation,
        ) {
            Ok(data) if !data.claims.sub.trim().is_empty() => data.claims,
            _ => {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"detail":"unauthorized"})),
                )
                    .into_response()
            }
        };
        request.extensions_mut().insert(AuthContext {
            user_id: claims.sub,
        });
        return next.run(request).await;
    }
    if let Some(expected) = state.token.as_deref() {
        let supplied = headers
            .get("authorization")
            .or_else(|| headers.get("x-api-key"))
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !token_is_valid(supplied, expected) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"detail":"unauthorized"})),
            )
                .into_response();
        }
    }
    next.run(request).await
}

#[derive(Debug, serde::Deserialize)]
struct Claims {
    sub: String,
}

#[derive(Clone, Debug)]
struct AuthContext {
    user_id: String,
}

async fn health() -> impl IntoResponse {
    Json(json!({"status":"ok", "version": env!("CARGO_PKG_VERSION")}))
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("../../dashboard/index.html"))
}

async fn dashboard_app() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../../dashboard/app.js"),
    )
}

async fn dashboard_event_detail() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../../dashboard/event-detail.js"),
    )
}

async fn dashboard_styles() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../../dashboard/styles.css"),
    )
}

async fn dashboard_logo() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static [u8],
) {
    (
        [(axum::http::header::CONTENT_TYPE, "image/png")],
        include_bytes!("../../dashboard/rooagi-logo.png"),
    )
}

async fn api_status(State(state): State<AppState>) -> impl IntoResponse {
    let inspection = match state.service.read() {
        Ok(service) => service.inspection(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail":"status unavailable"})),
            )
                .into_response()
        }
    };
    let telemetry = project_query_snapshot(&state.project_root)
        .ok()
        .flatten()
        .unwrap_or_else(|| state.telemetry.snapshot());
    Json(DashboardStatus {
        status: if inspection.dirty {
            "degraded"
        } else {
            "healthy"
        },
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: now_unix_ms()
            .saturating_sub(telemetry.started_at_ms)
            .checked_div(1_000)
            .unwrap_or_default(),
        index: dashboard_index_status(inspection),
        telemetry,
        integrations: dashboard_integrations(&state.project_root),
    })
    .into_response()
}

async fn api_timeseries(State(state): State<AppState>) -> impl IntoResponse {
    Json(
        project_query_snapshot(&state.project_root)
            .ok()
            .flatten()
            .unwrap_or_else(|| state.telemetry.snapshot()),
    )
}

async fn api_integrations(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "integrations": dashboard_integrations(&state.project_root),
        "note": "Provider hooks and MCP processes report independently; this server does not infer agent connectivity from index health."
    }))
}

async fn api_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let mut sessions = BTreeMap::<(String, String), Vec<ProviderLifecycleEvent>>::new();
    for integration in dashboard_integrations(&state.project_root) {
        for event in integration.events {
            sessions
                .entry((event.provider.clone(), event.session_key.clone()))
                .or_default()
                .push(event);
        }
    }
    let sessions = sessions
        .into_iter()
        .map(|((provider, session_key), mut events)| {
            events.sort_by_key(|event| event.timestamp_ms);
            let first_seen_ms = events
                .first()
                .map(|event| event.timestamp_ms)
                .unwrap_or_default();
            let last = events.last();
            DashboardSession {
                provider,
                session_key,
                event_count: events.len(),
                first_seen_ms,
                last_seen_ms: last.map(|event| event.timestamp_ms).unwrap_or_default(),
                last_event: last.map(|event| event.event.clone()).unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    Json(json!({"sessions": sessions}))
}

async fn api_events(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({"events": recent_provider_events(&state.project_root)}))
}

async fn api_session_events(
    State(state): State<AppState>,
    Path(session_key): Path<String>,
) -> impl IntoResponse {
    let events = recent_provider_events(&state.project_root)
        .into_iter()
        .filter(|event| event.session_key == session_key)
        .collect::<Vec<_>>();
    Json(json!({"session_key": session_key, "events": events}))
}

async fn api_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let integrations = dashboard_integrations(&state.project_root);
    let query = project_query_snapshot(&state.project_root)
        .ok()
        .flatten()
        .unwrap_or_else(|| state.telemetry.snapshot())
        .query_summary;
    Json(json!({
        "query": query,
        "providers": integrations.into_iter().map(|item| json!({
            "provider": item.provider,
            "compiled": item.compiled,
            "state": item.state,
            "events_total": item.events_total,
            "sessions_started": item.sessions_started,
            "sessions_ended": item.sessions_ended,
            "sessions_active": item.sessions_active,
            "retrieval_events": item.retrieval_events,
            "capture_events": item.capture_events,
        })).collect::<Vec<_>>(),
    }))
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let query = project_query_snapshot(&state.project_root)
        .ok()
        .flatten()
        .unwrap_or_else(|| state.telemetry.snapshot())
        .query_summary;
    let integrations = dashboard_integrations(&state.project_root);
    let mut body = String::from("# Lint-AI operational metrics\n");
    body.push_str("# TYPE lint_ai_query_requests_total counter\n");
    body.push_str(&format!(
        "lint_ai_query_requests_total {}\n",
        query.requests
    ));
    body.push_str("# TYPE lint_ai_query_errors_total counter\n");
    body.push_str(&format!("lint_ai_query_errors_total {}\n", query.errors));
    body.push_str("# TYPE lint_ai_query_requests_per_second gauge\n");
    body.push_str(&format!(
        "lint_ai_query_requests_per_second {}\n",
        query.requests_per_second
    ));
    body.push_str("# TYPE lint_ai_provider_events_total counter\n");
    for integration in integrations {
        body.push_str(&format!(
            "lint_ai_provider_events_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.events_total
        ));
        body.push_str(&format!(
            "lint_ai_provider_sessions_active{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.sessions_active
        ));
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
}

fn recent_provider_events(root: &std::path::Path) -> Vec<ProviderLifecycleEvent> {
    let mut events = dashboard_integrations(root)
        .into_iter()
        .flat_map(|integration| integration.events)
        .collect::<Vec<_>>();
    events.sort_by_key(|event| std::cmp::Reverse(event.timestamp_ms));
    events.truncate(100);
    events
}

fn dashboard_index_status(inspection: IndexStoreInspection) -> DashboardIndexStatus {
    let snapshot = inspection.snapshot.map(|snapshot| DashboardSnapshotStatus {
        layout: snapshot.layout,
        segment_count: snapshot.segment_count,
        global_document_count: snapshot.global_document_count,
        segments: snapshot
            .segments
            .into_iter()
            .map(|segment| DashboardSegmentStatus {
                segment_id: segment.segment_id,
                document_count: segment.document_count,
                profile_term_count: segment.profile_term_count,
                profile_entity_count: segment.profile_entity_count,
                profile_topic_count: segment.profile_topic_count,
                profile_local_memory_count: segment.profile_local_memory_count,
            })
            .collect(),
    });
    DashboardIndexStatus {
        source_document_count: inspection.source_document_count,
        record_count: inspection.record_count,
        dirty: inspection.dirty,
        store_revision: inspection.store_revision,
        snapshot_revision: inspection.snapshot_revision,
        revision_lag: inspection
            .store_revision
            .saturating_sub(inspection.snapshot_revision),
        snapshot,
    }
}

fn dashboard_integrations(root: &std::path::Path) -> Vec<DashboardIntegrationStatus> {
    vec![
        dashboard_integration("Claude Code", cfg!(feature = "claude-code"), "claude", root),
        dashboard_integration("Codex", cfg!(feature = "codex"), "codex", root),
        dashboard_integration(
            "Gemini CLI",
            cfg!(feature = "gemini-cli"),
            "gemini-cli",
            root,
        ),
        dashboard_integration("AGY", cfg!(feature = "agy"), "agy", root),
    ]
}

fn dashboard_integration(
    provider: &'static str,
    compiled: bool,
    recording_provider: &'static str,
    root: &std::path::Path,
) -> DashboardIntegrationStatus {
    let observed = provider_lifecycle_status(root, recording_provider)
        .ok()
        .flatten();
    let (state, note) = match observed.as_ref() {
        Some(status) if now_unix_ms().saturating_sub(status.last_seen_ms) <= 60_000 => (
            "active",
            "Lifecycle events have been received in the last minute.",
        ),
        Some(_) => (
            "idle",
            "Lifecycle events were received, but none arrived in the last minute.",
        ),
        None => (
            "not_observed",
            "No provider lifecycle events have been received yet.",
        ),
    };
    let status = observed.unwrap_or_default();
    DashboardIntegrationStatus {
        provider,
        recording_provider,
        compiled,
        state,
        note,
        last_event: status.last_event,
        last_seen_ms: (status.last_seen_ms > 0).then_some(status.last_seen_ms),
        events_total: status.events_total,
        sessions_started: status.sessions_started,
        sessions_ended: status.sessions_ended,
        sessions_active: status.sessions_active,
        retrieval_events: status.retrieval_events,
        capture_events: status.capture_events,
        indexes: dashboard_provider_indexes(root, recording_provider),
        events: status.events,
    }
}

fn discover_index_paths(root: &std::path::Path) -> Vec<PathBuf> {
    let directory = root.join(".lint-ai");
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path.join("metadata.json").is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name == "workspace-memory"
                            || name.ends_with("-mcp-index")
                            || name.ends_with("-memory")
                    })
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn dashboard_provider_indexes(
    root: &std::path::Path,
    provider: &str,
) -> Vec<DashboardProviderIndex> {
    discover_index_paths(root)
        .into_iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?.to_string();
            let prefix = name
                .strip_suffix("-mcp-index")
                .or_else(|| name.strip_suffix("-memory"))?;
            let normalized = if provider == "gemini-cli" {
                "gemini"
            } else {
                provider
            };
            if prefix != normalized {
                return None;
            }
            let inspection = MemoryService::at_path(
                &path,
                memory_pipeline_options(None, false, false, false, true),
            )
            .ok()?
            .inspection();
            let snapshot = inspection.snapshot.map(|snapshot| DashboardSnapshotStatus {
                layout: snapshot.layout,
                segment_count: snapshot.segment_count,
                global_document_count: snapshot.global_document_count,
                segments: snapshot
                    .segments
                    .into_iter()
                    .map(|segment| DashboardSegmentStatus {
                        segment_id: segment.segment_id,
                        document_count: segment.document_count,
                        profile_term_count: segment.profile_term_count,
                        profile_entity_count: segment.profile_entity_count,
                        profile_topic_count: segment.profile_topic_count,
                        profile_local_memory_count: segment.profile_local_memory_count,
                    })
                    .collect(),
            });
            Some(DashboardProviderIndex {
                name,
                role: if path.file_name()?.to_str()?.ends_with("-memory") {
                    "memory".to_string()
                } else {
                    "mcp".to_string()
                },
                source_document_count: inspection.source_document_count,
                record_count: inspection.record_count,
                dirty: inspection.dirty,
                store_revision: inspection.store_revision,
                snapshot_revision: inspection.snapshot_revision,
                snapshot,
            })
        })
        .collect()
}

fn tenant_check(
    state: &AppState,
    value: &Value,
    auth: Option<&AuthContext>,
) -> Result<(), (StatusCode, Json<Value>)> {
    if let Some(auth) = auth {
        if value.get("user_id").and_then(Value::as_str) != Some(auth.user_id.as_str()) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(json!({"detail":"user is not authorized"})),
            ));
        }
    }
    if let Some(expected) = state.tenant_id.as_deref() {
        if value.get("user_id").and_then(Value::as_str) != Some(expected) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(json!({"detail":"tenant is not authorized"})),
            ));
        }
    }
    Ok(())
}

async fn add(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let request = match serde_json::from_value::<AddRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            publish_mutation(&state, |service| service.add(request))
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(result) => (StatusCode::OK, Json(serde_json::to_value(result).unwrap())).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"mutation failed"})),
        )
            .into_response(),
    }
}

async fn add_batch(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(requests): Json<Vec<AddRequest>>,
) -> impl IntoResponse {
    if requests.is_empty() || requests.len() > 128 {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"detail":"batch must contain between 1 and 128 requests"})),
        )
            .into_response();
    }
    for request in &requests {
        let value = match serde_json::to_value(request) {
            Ok(value) => value,
            Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
        };
        if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
            return error.into_response();
        }
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || publish_mutation(&state, |service| service.add_batch(requests))
    })
    .await
    .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")));
    match result {
        Ok(result) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"batch mutation failed"})),
        )
            .into_response(),
    }
}

async fn search(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let request = match serde_json::from_value::<SearchRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    // Published snapshots are immutable and search is read-only. Execute it
    // directly so concurrent requests do not queue behind the blocking-pool
    // handoff; mutation/index rebuild work remains on spawn_blocking paths.
    let started = Instant::now();
    let result = match state.published_search.read() {
        Ok(service) => service.search(request),
        Err(_) => Err(anyhow::anyhow!("published search lock poisoned")),
    };
    match result {
        Ok(result) => {
            state.telemetry.record_query(
                started.elapsed().as_millis() as u64,
                false,
                result.data.is_empty(),
            );
            Json(serde_json::to_value(result).unwrap()).into_response()
        }
        Err(_) => {
            state
                .telemetry
                .record_query(started.elapsed().as_millis() as u64, true, false);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail":"search failed"})),
            )
                .into_response()
        }
    }
}

async fn get_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Path(memory_id): Path<String>,
    Query(query): Query<GetRequest>,
) -> impl IntoResponse {
    let request = GetRequest { memory_id, ..query };
    let value = match serde_json::to_value(&request) {
        Ok(value) => value,
        Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
    };
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let result = match state.service.read() {
        Ok(service) => service.get(request),
        Err(_) => Err(anyhow::anyhow!("memory reader lock poisoned")),
    };
    match result {
        Ok(Some(memory)) => Json(memory).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"memory lookup failed"})),
        )
            .into_response(),
    }
}

async fn list_memories(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Query(request): Query<ListRequest>,
) -> impl IntoResponse {
    let value = match serde_json::to_value(&request) {
        Ok(value) => value,
        Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
    };
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let result = match state.service.read() {
        Ok(service) => service.list(request),
        Err(_) => Err(anyhow::anyhow!("memory reader lock poisoned")),
    };
    match result {
        Ok(result) => Json(result).into_response(),
        Err(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"detail":"invalid list request"})),
        )
            .into_response(),
    }
}

async fn update_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Path(memory_id): Path<String>,
    Json(mut request): Json<UpdateRequest>,
) -> impl IntoResponse {
    request.memory_id = memory_id;
    let value = match serde_json::to_value(&request) {
        Ok(value) => value,
        Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
    };
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            publish_mutation(&state, |service| service.update(request))
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(Some(memory)) => Json(memory).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"detail":"memory update failed"})),
        )
            .into_response(),
    }
}

async fn delete_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Path(memory_id): Path<String>,
    Query(query): Query<DeleteRequest>,
) -> impl IntoResponse {
    let request = DeleteRequest {
        doc_id: memory_id,
        ..query
    };
    let value = match serde_json::to_value(&request) {
        Ok(value) => value,
        Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
    };
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            publish_mutation(&state, |service| {
                service.delete(&request.user_id, &request.doc_id)
            })
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"memory deletion failed"})),
        )
            .into_response(),
    }
}

async fn refresh_memories(State(state): State<AppState>) -> impl IntoResponse {
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || publish_mutation(&state, |service| service.refresh()))
            .await
            .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"memory refresh failed"})),
        )
            .into_response(),
    }
}

async fn delete(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let request = match serde_json::from_value::<DeleteRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            publish_mutation(&state, |s| s.delete(&request.user_id, &request.doc_id))
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(affected) => Json(json!({"success":true,"affected":affected as usize})).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"mutation failed"})),
        )
            .into_response(),
    }
}

async fn supersede(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let request = match serde_json::from_value::<SupersedeRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            publish_mutation(&state, |s| {
                s.supersede(&request.user_id, &request.replacement_id, &request.old_id)
            })
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")))
    };
    match result {
        Ok(affected) => Json(json!({"success":true,"affected":affected as usize})).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"mutation failed"})),
        )
            .into_response(),
    }
}

async fn expire(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let Ok(_writer) = state.writer_gate.try_lock() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let user_id = value
        .get("user_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let state_for_mutation = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        publish_mutation(&state_for_mutation, |service| service.expire(&user_id))
    })
    .await
    .unwrap_or_else(|error| Err(anyhow::anyhow!("mutation task failed: {error}")));
    match result {
        Ok(affected) => Json(json!({"success":true,"affected":affected})).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"mutation failed"})),
        )
            .into_response(),
    }
}

fn publish_mutation<T>(
    state: &AppState,
    mutation: impl FnOnce(&mut MemoryService) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let mut service = state
        .service
        .write()
        .map_err(|_| anyhow::anyhow!("memory writer lock poisoned"))?;
    let result = mutation(&mut service)?;
    let next = service.published_search();
    drop(service);
    *state
        .published_search
        .write()
        .map_err(|_| anyhow::anyhow!("published search lock poisoned"))? = next;
    Ok(result)
}

fn ensure_loopback_bind(bind: &str) -> anyhow::Result<()> {
    use std::net::ToSocketAddrs;
    let mut resolved = false;
    for address in bind.to_socket_addrs()? {
        resolved = true;
        if !address.ip().is_loopback() {
            anyhow::bail!(
                "refusing non-localhost bind address {}; lint-ai server supports localhost only",
                address
            );
        }
    }
    anyhow::ensure!(resolved, "bind address resolved to no addresses: {bind}");
    Ok(())
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    let mut difference = (left.len() ^ right.len()) as u8;
    for i in 0..left.len().max(right.len()) {
        difference |= left.get(i).copied().unwrap_or(0) ^ right.get(i).copied().unwrap_or(0);
    }
    difference == 0
}

fn token_is_valid(supplied: &str, expected: &str) -> bool {
    constant_time_eq(supplied, expected)
        || constant_time_eq(supplied, &format!("Bearer {expected}"))
        || constant_time_eq(supplied, &format!("Token {expected}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_uses_segmented_memory_index() {
        assert!(matches!(
            memory_pipeline_options(None, false, false, false, true).memory_index_layout,
            MemoryIndexLayout::Segmented { .. }
        ));
    }

    #[test]
    fn server_adaptive_mode_is_opt_in() {
        assert!(matches!(
            memory_pipeline_options(Some(8), false, false, false, true).memory_index_layout,
            MemoryIndexLayout::AdaptiveSegmented {
                query_top_n: 3,
                max_query_n: 8,
                ..
            }
        ));
    }

    #[test]
    fn server_adaptive_limit_at_or_below_base_keeps_fixed_mode() {
        for limit in [0, 2, 3] {
            assert!(matches!(
                memory_pipeline_options(Some(limit), false, false, false, true).memory_index_layout,
                MemoryIndexLayout::Segmented { .. }
            ));
        }
    }

    #[test]
    fn empty_jwt_secret_is_not_considered_configured() {
        assert!(normalize_secret(Some("   ".to_string())).is_none());
    }

    #[test]
    fn published_search_lock_is_independent_from_writer_lock() {
        let service =
            MemoryService::in_memory(memory_pipeline_options(None, false, false, false, true));
        let published = service.published_search();
        let state = AppState {
            service: Arc::new(RwLock::new(service)),
            published_search: Arc::new(RwLock::new(published)),
            writer_gate: Arc::new(tokio::sync::Mutex::new(())),
            token: None,
            jwt_secret: None,
            tenant_id: None,
            telemetry: OperationalTelemetry::new(),
            project_root: std::env::current_dir().unwrap(),
        };
        let _writer = state.service.write().unwrap();
        assert!(state.published_search.try_read().is_ok());
    }

    #[test]
    fn token_accepts_supported_forms_and_rejects_wrong_values() {
        assert!(token_is_valid("secret", "secret"));
        assert!(token_is_valid("Bearer secret", "secret"));
        assert!(token_is_valid("Token secret", "secret"));
        assert!(!token_is_valid("Bearer other", "secret"));
    }

    #[test]
    fn server_accepts_only_loopback_bind_addresses() {
        assert!(ensure_loopback_bind("127.0.0.1:8080").is_ok());
        assert!(ensure_loopback_bind("[::1]:8080").is_ok());
        assert!(ensure_loopback_bind("0.0.0.0:8080").is_err());
        assert!(ensure_loopback_bind("192.168.1.10:8080").is_err());
    }

    #[tokio::test]
    async fn writer_gate_rejects_a_second_mutation() {
        let gate = tokio::sync::Mutex::new(());
        let _held = gate.lock().await;
        assert!(gate.try_lock().is_err());
    }

    #[test]
    fn jwt_subject_can_be_decoded_with_expiry_validation() {
        let key = b"test-secret";
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &serde_json::json!({"sub":"user-a","exp":4_102_444_800u64}),
            &jsonwebtoken::EncodingKey::from_secret(key),
        )
        .unwrap();
        let claims = jsonwebtoken::decode::<Claims>(
            &token,
            &jsonwebtoken::DecodingKey::from_secret(key),
            &Validation::new(jsonwebtoken::Algorithm::HS256),
        )
        .unwrap()
        .claims;
        assert_eq!(claims.sub, "user-a");
    }
}
