#[path = "server_writer.rs"]
mod server_writer;
use server_writer::StagedWriter;

use crate::memory_api::{
    AddRequest, DeleteRequest, GetRequest, ListRequest, MemoryService, SearchRequest,
    SupersedeRequest, UpdateRequest,
};
use crate::segments::SegmentRoutingStrategy;
use crate::telemetry::{
    project_query_snapshot, provider_lifecycle_status, OperationalTelemetry,
    ProviderLifecycleEvent, TelemetrySnapshot,
};
use crate::{
    default_production_pipeline_options, lang::Lang, IndexStoreInspection, MemoryIndexLayout,
    PipelineOptions, DEFAULT_SEGMENT_QUERY_TOP_N,
};
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
    /// Publish staged writes after this interval from the first pending add.
    #[arg(long, default_value_t = 250)]
    refresh_interval_ms: u64,
    /// Publish early when this many documents are pending.
    #[arg(long, default_value_t = 512)]
    refresh_batch_size: usize,
    /// Persist a full checkpoint and reclaim the durable journal on this cadence.
    #[arg(long, default_value_t = 30)]
    checkpoint_interval_seconds: u64,
    #[arg(long)]
    server_token: Option<String>,
    #[arg(long)]
    tenant_id: Option<String>,
    #[arg(long)]
    allow_unauthenticated: bool,
    /// Allow binding to non-loopback interfaces. Requires token or JWT authentication.
    #[arg(long)]
    allow_non_loopback: bool,
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
    /// Enable optional Bekind linguistic enrichment (disabled by default).
    #[arg(long)]
    bekind: bool,
    /// Segment routing strategy for the segmented layout.
    /// Names match docs/benchmark-results.md; the default is the measured
    /// best recall-per-latency trade-off (gated coverage-local).
    #[arg(long, value_enum, default_value_t = SegmentRoutingArg::GatedCoverageLocal)]
    segment_routing: SegmentRoutingArg,
    /// Content language. `auto` (default) detects per text from script
    /// statistics (plus Spanish signals for Latin text); pass `zh`/`ko`/`es`/`en` to force it.
    #[arg(long, value_enum, default_value = "auto")]
    lang: Lang,
    /// Project root containing provider hook telemetry under `.lint-ai`.
    #[arg(long)]
    project_root: Option<PathBuf>,
}

/// CLI-selectable segment routing strategies. Variant names map to the router
/// names in docs/benchmark-results.md; see SegmentRoutingStrategy for the
/// scoring each one applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SegmentRoutingArg {
    Sparse,
    LocalDistinctiveness,
    CoverageLocal,
    CoverageTeam,
    TeamCoverageLocal,
    TypedEvidenceAdditive,
    GatedCoverageLocal,
    GatedCoverageTeam,
}

impl SegmentRoutingArg {
    fn strategy(self) -> SegmentRoutingStrategy {
        match self {
            SegmentRoutingArg::Sparse => SegmentRoutingStrategy::SparseOverlap,
            SegmentRoutingArg::LocalDistinctiveness => SegmentRoutingStrategy::LocalDistinctiveness,
            SegmentRoutingArg::CoverageLocal => {
                SegmentRoutingStrategy::CoverageLocalDistinctiveness
            }
            SegmentRoutingArg::CoverageTeam => SegmentRoutingStrategy::CoverageTeamSelection,
            SegmentRoutingArg::TeamCoverageLocal => {
                SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness
            }
            SegmentRoutingArg::TypedEvidenceAdditive => SegmentRoutingStrategy::TypedEvidence,
            SegmentRoutingArg::GatedCoverageLocal => {
                SegmentRoutingStrategy::TypedEvidenceMultiplicative
            }
            SegmentRoutingArg::GatedCoverageTeam => {
                SegmentRoutingStrategy::CoverageTeamTypedMultiplicative
            }
        }
    }
}

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<MemoryService>>,
    writer_gate: Arc<tokio::sync::Mutex<()>>,
    staged_writer: Option<StagedWriter>,
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
pub(crate) async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    let bekind_enabled = args.bekind;
    crate::behood_query::set_enabled(bekind_enabled);
    args.server_token = normalize_secret(
        args.server_token
            .or_else(|| std::env::var("SERVER_TOKEN").ok()),
    );
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
    ensure_bind_allowed(
        &args.bind,
        args.allow_non_loopback,
        args.server_token.is_some() || jwt_secret.is_some(),
        args.allow_unauthenticated,
    )?;
    let mut options = memory_pipeline_options(
        args.adaptive_segment_max_n,
        args.single_index,
        args.global_index,
        args.fuse_global,
        !args.no_conversational_rerank,
        args.segment_routing.strategy(),
    );
    options.lang = args.lang;
    let project_root = args
        .project_root
        .or_else(|| std::env::var_os("LINT_AI_PROJECT_ROOT").map(PathBuf::from))
        .unwrap_or(std::env::current_dir()?)
        .canonicalize()?;
    let discovered_indexes = discover_index_paths(&project_root);
    // behood (bekind) owns the full tag→chunk→judge pipeline (Luyi
    // 2026-09-30): the query path sends raw texts to `bekind --serve` and
    // gets verdicts back. No parse backend to select, no spaCy involved.
    // Note: the structured-relations extractor (ExtractorDaemon,
    // scripts/spacy_relations.py) is a separate spaCy component.
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
    let mut state = AppState {
        service: Arc::new(RwLock::new(service)),
        writer_gate: Arc::new(tokio::sync::Mutex::new(())),
        staged_writer: None,
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
    // Warm the Python daemon children in the background: the first query
    // that needs key-phrase backfill or structured relations then pays
    // inference only (~100ms) instead of interpreter+model load (~2-3s).
    // Best-effort — extraction/analysis falls back to one-shot subprocesses
    // if a daemon cannot start.
    //
    // The NER daemon is deliberately NOT prewarmed: it starts lazily on the
    // first NER request. Prewarming would force a third Python+spaCy child
    // (~145MB RSS) on every server start, even when the heuristic NER
    // provider is configured and spaCy is never used. (Luyi 2026-09-29 P2.)
    std::thread::Builder::new()
        .name("python-daemon-prewarm".to_string())
        .spawn(move || {
            crate::segments::extractor_daemon::ExtractorDaemon::global().prewarm();
            // The judge daemon is optional and prewarms only when --bekind
            // enabled it.
            if bekind_enabled {
                crate::behood_query::BekindDaemon::global().prewarm();
            }
        })
        .ok();
    state.staged_writer = Some(StagedWriter::start_with_schedule(
        state.service.clone(),
        server_writer::WriteSchedule {
            refresh_interval: Duration::from_millis(args.refresh_interval_ms),
            batch_size: args.refresh_batch_size,
            checkpoint_interval: Duration::from_secs(args.checkpoint_interval_seconds),
        },
    )?);
    let app = app_router(state.clone());
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    eprintln!("Lint-AI server listening on {}", args.bind);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("install SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await?;
    if let Some(worker) = &state.staged_writer {
        worker
            .flush()
            .await
            .map_err(|status| anyhow::anyhow!("shutdown flush failed: {status}"))?;
    }
    Ok(())
}

fn app_router(state: AppState) -> Router {
    let router = Router::new()
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
        .route("/flush", post(refresh_memories))
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
        .route("/v1/memories/refresh", post(refresh_memories));
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code",
        feature = "openclaw",
        feature = "hermes",
        feature = "roo-runtime"
    ))]
    let router = router
        .route(
            "/provider-memory/add/batch",
            post(provider_memory_add_batch),
        )
        .route("/provider-memory/search", post(provider_memory_search));
    #[cfg(feature = "openclaw")]
    let router = router.route("/integrations/openclaw/hooks/:kind", post(openclaw_hook));
    router
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
        .with_state(state)
}

#[cfg(feature = "openclaw")]
async fn openclaw_hook(
    State(state): State<AppState>,
    Path(kind): Path<String>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let Some(kind) = crate::integrations::openclaw::hooks::OpenClawHookKind::from_str(&kind) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail":"unknown OpenClaw hook"})),
        )
            .into_response();
    };
    // These hooks read/write workspace-wide provider data, not tenant data.
    if state.tenant_id.is_some() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail":"workspace hooks are unavailable in tenant mode"})),
        )
            .into_response();
    }
    let root = match state.project_root.canonicalize() {
        Ok(root) => root,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    for pointer in [
        "/ctx/workspaceDir",
        "/event/context/workspaceDir",
        "/event/workspaceDir",
    ] {
        if let Some(candidate) = payload
            .pointer(pointer)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            if std::fs::canonicalize(candidate.trim()).ok().as_ref() != Some(&root) {
                return (
                    StatusCode::FORBIDDEN,
                    Json(json!({"detail":"hook workspace does not match the server workspace"})),
                )
                    .into_response();
            }
        }
    }
    // OpenClaw captures and Hermes provider-memory writes share the same
    // server-level mutation admission gate. The shared-store file lock in
    // `MemoryService::with_shared_memory` remains the cross-process guard.
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = match tokio::task::spawn_blocking(move || {
        // Keep admission held if the HTTP timeout cancels the awaiting task.
        let _writer = writer;
        crate::integrations::openclaw::hooks::process_http_hook(kind, payload, &root)
    })
    .await
    {
        Ok(result) => result,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    (StatusCode::OK, Json(result)).into_response()
}

/// Production pipeline options with the server's CLI flags applied as
/// overrides on top of [`default_production_pipeline_options`].
fn memory_pipeline_options(
    adaptive_segment_max_n: Option<usize>,
    single_index: bool,
    global_index: bool,
    fuse_global: bool,
    conversational_rerank: bool,
    routing_strategy: SegmentRoutingStrategy,
) -> PipelineOptions {
    if single_index {
        return PipelineOptions {
            memory_index_layout: MemoryIndexLayout::Single,

            ..default_production_pipeline_options()
        };
    }
    let layout = if global_index {
        MemoryIndexLayout::Segmented {
            query_top_n: usize::MAX,
            routing_strategy,
        }
    } else {
        adaptive_segment_max_n
            .filter(|max_n| *max_n > DEFAULT_SEGMENT_QUERY_TOP_N)
            .map(|max_n| MemoryIndexLayout::AdaptiveSegmented {
                query_top_n: DEFAULT_SEGMENT_QUERY_TOP_N,
                max_query_n: max_n,
                routing_strategy,
            })
            .unwrap_or(MemoryIndexLayout::Segmented {
                query_top_n: DEFAULT_SEGMENT_QUERY_TOP_N,
                routing_strategy,
            })
    };
    PipelineOptions {
        memory_index_layout: layout,
        fuse_global_arm: fuse_global,
        conversational_rerank,

        ..default_production_pipeline_options()
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
    if is_workspace_operation_path(path) {
        if let Some(expected) = state.token.as_deref() {
            let supplied = headers
                .get("authorization")
                .or_else(|| headers.get("x-api-key"))
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if token_is_valid(supplied, expected) {
                return next.run(request).await;
            }
        }
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
        // JWT currently establishes a user identity only. Until the server
        // defines and validates an administrative role, JWT identities must
        // not read workspace-wide operational telemetry.
        if is_workspace_operation_path(path) {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"detail":"workspace operations require server-token authentication"})),
            )
                .into_response();
        }
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

fn is_workspace_operation_path(path: &str) -> bool {
    // Provider hooks access the entire workspace and do not implement
    // per-user attribution or retrieval. A user JWT is insufficient.
    is_workspace_telemetry_path(path) || path.starts_with("/integrations/openclaw/hooks/")
}

fn is_workspace_telemetry_path(path: &str) -> bool {
    matches!(
        path,
        "/api/events"
            | "/api/integrations"
            | "/api/sessions"
            | "/api/status"
            | "/api/timeseries"
            | "/api/metrics"
            | "/metrics"
    ) || path.starts_with("/api/sessions/")
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
    body.push_str("# HELP lint_ai_query_requests_window Search requests in the retained telemetry window.\n# TYPE lint_ai_query_requests_window gauge\n");
    body.push_str(&format!(
        "lint_ai_query_requests_window {}\n",
        query.requests
    ));
    body.push_str("# HELP lint_ai_query_errors_window Search errors in the retained telemetry window.\n# TYPE lint_ai_query_errors_window gauge\n");
    body.push_str(&format!("lint_ai_query_errors_window {}\n", query.errors));
    body.push_str("# HELP lint_ai_query_empty_results_window Searches with no results in the retained telemetry window.\n# TYPE lint_ai_query_empty_results_window gauge\n");
    body.push_str(&format!(
        "lint_ai_query_empty_results_window {}\n",
        query.empty_results
    ));
    body.push_str("# HELP lint_ai_query_requests_per_second Search request rate over the retained telemetry window.\n# TYPE lint_ai_query_requests_per_second gauge\n");
    body.push_str(&format!(
        "lint_ai_query_requests_per_second {}\n",
        query.requests_per_second
    ));
    body.push_str("# HELP lint_ai_query_error_rate Fraction of searches that returned errors in the retained telemetry window.\n# TYPE lint_ai_query_error_rate gauge\n");
    body.push_str(&format!("lint_ai_query_error_rate {}\n", query.error_rate));
    body.push_str("# HELP lint_ai_query_empty_result_rate Fraction of searches with no results in the retained telemetry window.\n# TYPE lint_ai_query_empty_result_rate gauge\n");
    body.push_str(&format!(
        "lint_ai_query_empty_result_rate {}\n",
        query.empty_result_rate
    ));
    body.push_str("# HELP lint_ai_query_latency_ms Estimated query latency quantile in milliseconds over the retained telemetry window.\n# TYPE lint_ai_query_latency_ms gauge\n");
    body.push_str(&format!(
        "lint_ai_query_latency_ms{{quantile=\"0.5\"}} {}\n",
        query.p50_ms
    ));
    body.push_str(&format!(
        "lint_ai_query_latency_ms{{quantile=\"0.95\"}} {}\n",
        query.p95_ms
    ));
    body.push_str("# HELP lint_ai_provider_compiled Whether this server build includes the provider integration.\n# TYPE lint_ai_provider_compiled gauge\n");
    body.push_str("# HELP lint_ai_provider_observed Whether Lint-AI has ever received lifecycle telemetry for the provider.\n# TYPE lint_ai_provider_observed gauge\n");
    body.push_str("# HELP lint_ai_provider_events_total Lifecycle events recorded for the provider.\n# TYPE lint_ai_provider_events_total counter\n");
    body.push_str("# HELP lint_ai_provider_sessions_started_total Provider sessions started.\n# TYPE lint_ai_provider_sessions_started_total counter\n");
    body.push_str("# HELP lint_ai_provider_sessions_ended_total Provider sessions ended.\n# TYPE lint_ai_provider_sessions_ended_total counter\n");
    body.push_str("# HELP lint_ai_provider_retrieval_events_total Provider lifecycle events categorized as retrieval.\n# TYPE lint_ai_provider_retrieval_events_total counter\n");
    body.push_str("# HELP lint_ai_provider_capture_events_total Provider lifecycle events categorized as capture.\n# TYPE lint_ai_provider_capture_events_total counter\n");
    body.push_str("# HELP lint_ai_provider_sessions_active Active provider sessions observed by Lint-AI.\n# TYPE lint_ai_provider_sessions_active gauge\n");
    body.push_str("# HELP lint_ai_provider_last_seen_timestamp_seconds Unix timestamp of the last received provider event, or zero if none.\n# TYPE lint_ai_provider_last_seen_timestamp_seconds gauge\n");
    body.push_str("# HELP lint_ai_provider_recent_events Events by bounded lifecycle category in the retained provider event ledger.\n# TYPE lint_ai_provider_recent_events gauge\n");
    body.push_str("# HELP lint_ai_provider_token_usage_recent Tokens reported in the retained provider event ledger.\n# TYPE lint_ai_provider_token_usage_recent gauge\n");
    for integration in integrations {
        body.push_str(&format!(
            "lint_ai_provider_events_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.events_total
        ));
        body.push_str(&format!(
            "lint_ai_provider_compiled{{provider=\"{}\"}} {}\n",
            integration.recording_provider,
            u8::from(integration.compiled)
        ));
        body.push_str(&format!(
            "lint_ai_provider_observed{{provider=\"{}\"}} {}\n",
            integration.recording_provider,
            u8::from(integration.last_seen_ms.is_some())
        ));
        body.push_str(&format!(
            "lint_ai_provider_sessions_started_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.sessions_started
        ));
        body.push_str(&format!(
            "lint_ai_provider_sessions_ended_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.sessions_ended
        ));
        body.push_str(&format!(
            "lint_ai_provider_retrieval_events_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.retrieval_events
        ));
        body.push_str(&format!(
            "lint_ai_provider_capture_events_total{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.capture_events
        ));
        body.push_str(&format!(
            "lint_ai_provider_sessions_active{{provider=\"{}\"}} {}\n",
            integration.recording_provider, integration.sessions_active
        ));
        body.push_str(&format!(
            "lint_ai_provider_last_seen_timestamp_seconds{{provider=\"{}\"}} {}\n",
            integration.recording_provider,
            integration.last_seen_ms.unwrap_or_default() as f64 / 1_000.0
        ));

        let mut recent_categories = BTreeMap::<&str, u64>::new();
        let mut recent_tokens = BTreeMap::<&str, u64>::new();
        for event in &integration.events {
            if matches!(
                event.category.as_str(),
                "session" | "retrieval" | "compaction" | "capture" | "lifecycle"
            ) {
                *recent_categories
                    .entry(event.category.as_str())
                    .or_default() += 1;
            }
            for (kind, value) in [
                ("input", event.input_tokens),
                ("output", event.output_tokens),
                ("cache_creation_input", event.cache_creation_input_tokens),
                ("cache_read_input", event.cache_read_input_tokens),
            ] {
                if let Some(value) = value {
                    let total = recent_tokens.entry(kind).or_default();
                    *total = total.saturating_add(value);
                }
            }
        }
        for category in ["session", "retrieval", "compaction", "capture", "lifecycle"] {
            body.push_str(&format!(
                "lint_ai_provider_recent_events{{provider=\"{}\",category=\"{}\"}} {}\n",
                integration.recording_provider,
                category,
                recent_categories.get(category).copied().unwrap_or_default()
            ));
        }
        for kind in [
            "input",
            "output",
            "cache_creation_input",
            "cache_read_input",
        ] {
            body.push_str(&format!(
                "lint_ai_provider_token_usage_recent{{provider=\"{}\",kind=\"{}\"}} {}\n",
                integration.recording_provider,
                kind,
                recent_tokens.get(kind).copied().unwrap_or_default()
            ));
        }
    }

    body.push_str("# HELP lint_ai_index_source_documents Source documents in the project memory index.\n# TYPE lint_ai_index_source_documents gauge\n");
    body.push_str("# HELP lint_ai_index_records Records in the project memory index.\n# TYPE lint_ai_index_records gauge\n");
    body.push_str("# HELP lint_ai_index_dirty Whether the memory index has unpublished changes.\n# TYPE lint_ai_index_dirty gauge\n");
    body.push_str("# HELP lint_ai_index_store_revision Current mutable store revision.\n# TYPE lint_ai_index_store_revision gauge\n");
    body.push_str("# HELP lint_ai_index_snapshot_revision Published search snapshot revision.\n# TYPE lint_ai_index_snapshot_revision gauge\n");
    body.push_str("# HELP lint_ai_index_revision_lag Difference between store and published snapshot revisions.\n# TYPE lint_ai_index_revision_lag gauge\n");
    body.push_str("# HELP lint_ai_index_segments Number of segments in the published search snapshot.\n# TYPE lint_ai_index_segments gauge\n");
    if let Ok(service) = state.service.read() {
        let inspection = service.inspection();
        body.push_str(&format!(
            "lint_ai_index_source_documents {}\nlint_ai_index_records {}\nlint_ai_index_dirty {}\nlint_ai_index_store_revision {}\nlint_ai_index_snapshot_revision {}\nlint_ai_index_revision_lag {}\nlint_ai_index_segments {}\n",
            inspection.source_document_count,
            inspection.record_count,
            u8::from(inspection.dirty),
            inspection.store_revision,
            inspection.snapshot_revision,
            inspection.store_revision.saturating_sub(inspection.snapshot_revision),
            inspection.snapshot.map(|snapshot| snapshot.segment_count).unwrap_or_default()
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
        dashboard_integration("OpenClaw", cfg!(feature = "openclaw"), "openclaw", root),
        dashboard_integration("Hermes", cfg!(feature = "hermes"), "hermes", root),
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
                memory_pipeline_options(
                    None,
                    false,
                    false,
                    false,
                    true,
                    SegmentRoutingStrategy::TypedEvidenceMultiplicative,
                ),
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

#[derive(Default, serde::Deserialize)]
struct WriteOptions {
    #[serde(default)]
    wait_for_visibility: bool,
}

async fn add(
    State(state): State<AppState>,
    Query(options): Query<WriteOptions>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
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
    if let Some(worker) = &state.staged_writer {
        return match worker.add(vec![request], options.wait_for_visibility).await {
            Ok(mut responses) => Json(responses.as_array_mut().unwrap().remove(0)).into_response(),
            Err(status) => (
                status,
                Json(json!({"detail":"mutation failed or writer queue full"})),
            )
                .into_response(),
        };
    }
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            let _writer = writer;
            write_mutation(&state, |service| service.add(request))
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
    Query(options): Query<WriteOptions>,
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
    if let Some(worker) = &state.staged_writer {
        return match worker.add(requests, options.wait_for_visibility).await {
            Ok(responses) => Json(responses).into_response(),
            Err(status) => (
                status,
                Json(json!({"detail":"mutation failed or writer queue full"})),
            )
                .into_response(),
        };
    }
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let _writer = writer;
            write_mutation(&state, |service| service.add_batch(requests))
        }
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

/// Provider hooks share `.lint-ai/memory` regardless of the server's primary
/// `--index`. Search composes that store with the workspace-memory index.
#[cfg(any(
    feature = "claude-code",
    feature = "codex",
    feature = "gemini-cli",
    feature = "agy",
    feature = "muse-code",
    feature = "openclaw",
    feature = "hermes",
    feature = "roo-runtime"
))]
async fn provider_memory_add_batch(
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let root = state.project_root.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _writer = writer;
        MemoryService::with_shared_memory(&root, |service| service.add_batch(requests))
    })
    .await
    .unwrap_or_else(|error| {
        Err(anyhow::anyhow!(
            "provider memory write task failed: {error}"
        ))
    });
    match result {
        Ok(result) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"provider memory batch mutation failed"})),
        )
            .into_response(),
    }
}

#[cfg(any(
    feature = "claude-code",
    feature = "codex",
    feature = "gemini-cli",
    feature = "agy",
    feature = "muse-code",
    feature = "openclaw",
    feature = "hermes",
    feature = "roo-runtime"
))]
async fn provider_memory_search(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    if let Err(error) = tenant_check(&state, &value, auth.as_ref().map(|a| &a.0)) {
        return error.into_response();
    }
    let mut request = match serde_json::from_value::<SearchRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    if let Some(auth) = auth.as_ref() {
        request.scope = Some(auth.0.user_id.clone());
    } else {
        // Server-token provider adapters share one workspace store across
        // integrations. Keep conversation state scoped to the caller's
        // configured identity, while allowing retrieval across provider-owned
        // memories and ordinary workspace documents. JWT callers remain
        // isolated to their authenticated user below.
        let caller_scope = request.user_id.clone();
        request.user_id.clear();
        if request.scope.as_deref().is_none_or(str::is_empty) {
            request.scope = Some(caller_scope);
        }
    }
    let root = state.project_root.clone();
    let result = tokio::task::spawn_blocking(move || {
        let ignore_paths = vec![
            "node_modules".to_string(),
            "target".to_string(),
            "dist".to_string(),
            "build".to_string(),
            "vendor".to_string(),
            "coverage".to_string(),
            ".git".to_string(),
        ];
        let input = crate::adapters::AdapterInput {
            root: &root,
            max_bytes: 5_000_000,
            max_files: 50_000,
            max_depth: 20,
            max_total_bytes: 100_000_000,
        };
        let mut service = MemoryService::open_workspace(
            &root,
            crate::integrations::mcp_index::SHARED_MEMORY_DIR,
            &ignore_paths,
            || {
                let graph = crate::adapters::build_project_graph(&input)?;
                let graph = crate::adapters::apply_ignore_paths(graph, &ignore_paths);
                Ok(crate::adapters::graph_to_source_documents(&graph))
            },
        )?;
        service.search(request)
    })
    .await
    .unwrap_or_else(|error| {
        Err(anyhow::anyhow!(
            "provider memory search task failed: {error}"
        ))
    });
    match result {
        Ok(result) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"provider memory search failed"})),
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
    let mut request = match serde_json::from_value::<SearchRequest>(value) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail":"invalid request"})),
            )
                .into_response()
        }
    };
    if let Some(auth) = auth.as_ref() {
        // The JWT subject owns both the documents and conversation context.
        // Ignore caller-provided scope so one user cannot mutate another
        // user's follow-up state.
        request.scope = Some(auth.0.user_id.clone());
    }
    // Search uses the already-published immutable index snapshot. Its small
    // mutable side state is internally synchronized, so requests can share
    // the service read lock and run retrieval concurrently.
    //
    // Query-time key-phrase backfill: documents written by provider hooks
    // (separate short-lived processes) never see this process's background
    // enrichment worker, so the first query that needs their phrases
    // extracts them synchronously here. The read-lock check is cheap, so
    // the steady state costs nothing; the extractor (the slow part) runs
    // with no service lock held, and only the fast apply step takes the
    // write lock. A concurrent write landing mid-backfill is safe:
    // application re-validates content hashes and skips stale results.
    // Bounded and fail-open.
    let backfill_needed = if let Some(worker) = &state.staged_writer {
        worker
            .read_view()
            .map(|view| view.key_phrase_backfill_needed())
            .unwrap_or(false)
    } else {
        state
            .service
            .read()
            .map(|service| service.key_phrase_backfill_needed())
            .unwrap_or(false)
    };
    if backfill_needed {
        let owned_state = state.clone();
        let backfill = tokio::task::spawn_blocking(move || {
            let (docs, script) = owned_state
                .service
                .read()
                .map(|service| service.key_phrase_backfill_snapshot())
                .map_err(|_| anyhow::anyhow!("memory service lock poisoned"))?;
            if docs.is_empty() {
                return Ok(0);
            }
            let raw = MemoryService::extract_key_phrases_for_docs(&docs, script.as_deref());
            write_mutation(&owned_state, |service| {
                Ok(service.apply_key_phrase_backfill(&docs, raw))
            })
        })
        .await
        .map_err(|join| anyhow::anyhow!("backfill task failed: {join}"))
        .and_then(|inner| inner);
        if let Err(error) = backfill {
            eprintln!("key-phrase backfill failed (fail-open): {error:#}");
        }
    }
    if let Some(worker) = &state.staged_writer {
        let started = Instant::now();
        let result = worker
            .read_view()
            .and_then(|view| view.search_cached(request));
        state.telemetry.record_query(
            started.elapsed().as_millis() as u64,
            result.is_err(),
            result.as_ref().map(|r| r.data.is_empty()).unwrap_or(false),
        );
        return match result {
            Ok(response) => Json(response).into_response(),
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
    }
    let started = Instant::now();
    let result = match state.service.read() {
        Ok(service) => service.search_cached(request),
        Err(_) => Err(anyhow::anyhow!("memory service lock poisoned")),
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            let _writer = writer;
            write_mutation(&state, |service| service.update(request))
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            let _writer = writer;
            write_mutation(&state, |service| {
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
    if let Some(worker) = &state.staged_writer {
        return match worker.flush().await {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(status) => status.into_response(),
        };
    }
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"detail":"writer busy"})),
        )
            .into_response();
    };
    let result = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            let _writer = writer;
            write_mutation(&state, |service| service.refresh())
        })
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
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
            let _writer = writer;
            write_mutation(&state, |s| s.delete(&request.user_id, &request.doc_id))
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
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
            let _writer = writer;
            write_mutation(&state, |s| {
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
    let Ok(writer) = state.writer_gate.clone().try_lock_owned() else {
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
        let _writer = writer;
        write_mutation(&state_for_mutation, |service| service.expire(&user_id))
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

fn write_mutation<T>(
    state: &AppState,
    mutation: impl FnOnce(&mut MemoryService) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    // Synchronous mutations checkpoint staged work and replace the detached
    // read view before returning. The mutable owner serializes all mutations.
    let mut service = state
        .service
        .write()
        .map_err(|_| anyhow::anyhow!("memory writer lock poisoned"))?;
    let result = mutation(&mut service)?;
    if let Some(worker) = &state.staged_writer {
        if service.pending_add_count() == 0 {
            worker.publish_owner(&service)?;
        }
    }
    drop(service);
    Ok(result)
}

fn ensure_bind_allowed(
    bind: &str,
    allow_non_loopback: bool,
    authentication_configured: bool,
    allow_unauthenticated: bool,
) -> anyhow::Result<()> {
    use std::net::ToSocketAddrs;
    let mut resolved = false;
    for address in bind.to_socket_addrs()? {
        resolved = true;
        if !address.ip().is_loopback() && !allow_non_loopback {
            anyhow::bail!(
                "refusing non-localhost bind address {}; pass --allow-non-loopback to opt in",
                address
            );
        }
    }
    anyhow::ensure!(resolved, "bind address resolved to no addresses: {bind}");
    if allow_non_loopback {
        anyhow::ensure!(
            authentication_configured && !allow_unauthenticated,
            "--allow-non-loopback requires SERVER_TOKEN or JWT_SECRET authentication and cannot be combined with --allow-unauthenticated"
        );
    }
    Ok(())
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    let mut difference = left.len() ^ right.len();
    for i in 0..left.len().max(right.len()) {
        difference |=
            usize::from(left.get(i).copied().unwrap_or(0) ^ right.get(i).copied().unwrap_or(0));
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

    fn security_test_root() -> PathBuf {
        static NEXT_ROOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lint-ai-http-security-{}-{}-{}",
            std::process::id(),
            now_unix_ms(),
            NEXT_ROOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::canonicalize(path).unwrap()
    }

    fn security_test_state(root: &std::path::Path) -> AppState {
        AppState {
            service: Arc::new(RwLock::new(
                MemoryService::at_path(
                    root,
                    PipelineOptions {
                        key_phrase_enrichment: false,
                        ..PipelineOptions::default()
                    },
                )
                .unwrap(),
            )),
            writer_gate: Arc::new(tokio::sync::Mutex::new(())),
            staged_writer: None,
            token: None,
            jwt_secret: Some("test-secret".into()),
            tenant_id: None,
            telemetry: OperationalTelemetry::new(),
            project_root: root.to_path_buf(),
        }
    }

    fn provider_memory_test_state(root: &std::path::Path) -> AppState {
        let mut state = security_test_state(root);
        state.jwt_secret = None;
        state
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn staged_http_add_flush_wait_and_user_isolation() {
        use tower::ServiceExt;
        let root = security_test_root();
        let mut state = provider_memory_test_state(&root);
        state.staged_writer = Some(
            StagedWriter::start_with_schedule(
                state.service.clone(),
                server_writer::WriteSchedule {
                    refresh_interval: Duration::from_secs(60),
                    batch_size: 10000,
                    checkpoint_interval: Duration::from_secs(60),
                },
            )
            .unwrap(),
        );
        let app = app_router(state.clone());
        async fn post(app: &Router, uri: &str, payload: Value) -> (StatusCode, Value) {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            (
                status,
                if bytes.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes).unwrap()
                },
            )
        }
        let payload = json!({"request_id":"http-staged","session_id":"session","user_id":"alice","messages":[{"role":"user","content":"quartz staged HTTP memory"}]});
        let (status, ack) = post(&app, "/add", payload.clone()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ack["published"], false);
        let (_, before) = post(
            &app,
            "/search",
            json!({"query":"quartz","user_id":"alice","top_k":10}),
        )
        .await;
        assert!(before["data"].as_array().unwrap().is_empty());
        assert_eq!(
            post(&app, "/flush", json!({})).await.0,
            StatusCode::NO_CONTENT
        );
        let (_, after) = post(
            &app,
            "/search",
            json!({"query":"quartz","user_id":"alice","top_k":10}),
        )
        .await;
        assert_eq!(after["data"].as_array().unwrap().len(), 1);
        let (_, foreign) = post(
            &app,
            "/search",
            json!({"query":"quartz","user_id":"bob","top_k":10}),
        )
        .await;
        assert!(foreign["data"].as_array().unwrap().is_empty());
        let (_, receipt) = post(&app, "/add?wait_for_visibility=true", payload).await;
        assert!(receipt["adjudication"].is_object());
        assert_eq!(
            state
                .service
                .read()
                .unwrap()
                .inspection()
                .source_document_count,
            1
        );
        drop(app);
        drop(state);
        // The worker's final checkpoint may still be completing, so avoid
        // deleting its root here; the OS temp directory is test-owned.
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_write_keeps_admission_until_worker_finishes() {
        use tower::ServiceExt;
        let root = security_test_root();
        let state = provider_memory_test_state(&root);
        let read_guard = state.service.read().unwrap();
        let app = app_router(state.clone());
        let request = tokio::spawn(async move {
            app.oneshot(axum::http::Request::builder().method("POST").uri("/add")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(json!({"request_id":"cancel-test","session_id":"cancel-session","user_id":"alice","messages":[{"role":"user","timestamp":null,"content":"cancellation regression"}]}).to_string())).unwrap()).await
        });
        let acquired = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                if state.writer_gate.try_lock().is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_ok();
        request.abort();
        let _ = request.await;
        let remains_locked = state.writer_gate.try_lock().is_err();
        drop(read_guard);
        // Let the unabortable blocking worker finish before asserting or cleaning up.
        tokio::task::spawn_blocking({
            let state = state.clone();
            move || {
                let _guard = state.service.write().unwrap();
            }
        })
        .await
        .unwrap();
        assert!(acquired, "request never acquired writer admission");
        assert!(
            remains_locked,
            "request cancellation released admission while its writer was still running"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    fn security_test_token(claims: Value) -> String {
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap()
    }

    #[cfg(feature = "openclaw")]
    #[tokio::test]
    async fn openclaw_http_hook_rejects_other_workspaces() {
        use tower::ServiceExt;
        let root = security_test_root();
        let other = security_test_root();
        let app = app_router(provider_memory_test_state(&root));
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/integrations/openclaw/hooks/session-start")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({"event":{},"ctx":{"workspaceDir":other,"sessionId":"outside"}})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(
            !other.join(".lint-ai").exists(),
            "rejected hook must not write outside the server workspace"
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(other);
    }

    #[cfg(feature = "openclaw")]
    #[tokio::test]
    async fn openclaw_http_hook_rejects_jwt_workspace_access() {
        use tower::ServiceExt;
        let root = security_test_root();
        let app = app_router(security_test_state(&root));
        let token = security_test_token(json!({"sub":"alice", "exp":4_102_444_800u64}));
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/integrations/openclaw/hooks/bootstrap")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(axum::body::Body::from(
                        json!({"event":{"context":{"workspaceDir":root}},"query":"private memory"})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn jwt_is_identity_only_and_cannot_read_workspace_telemetry() {
        use tower::ServiceExt;
        let root = security_test_root();
        let app = app_router(security_test_state(&root));
        let token = security_test_token(json!({"sub":"alice", "exp":4_102_444_800u64}));
        // A valid JWT can still use user-scoped memory APIs.
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/memories?user_id=alice")
                    .header("authorization", format!("Bearer {token}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // JWT has no role model; an undocumented `admin` claim grants nothing.
        for claims in [
            json!({"sub":"alice", "exp":4_102_444_800u64}),
            json!({"sub":"alice", "admin":true, "exp":4_102_444_800u64}),
        ] {
            let token = security_test_token(claims);
            for path in [
                "/api/events",
                "/api/sessions/victim/events",
                "/api/integrations",
                "/api/sessions",
                "/api/status",
                "/api/timeseries",
                "/api/metrics",
                "/metrics",
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        axum::http::Request::builder()
                            .uri(path)
                            .header("authorization", format!("Bearer {token}"))
                            .body(axum::body::Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::FORBIDDEN,
                    "JWT telemetry exposed at {path}"
                );
            }
        }
        // The privilege claim is accepted only after signature validation.
        let forged = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &json!({"sub":"operator", "admin":true, "exp":4_102_444_800u64}),
            &jsonwebtoken::EncodingKey::from_secret(b"wrong-secret"),
        )
        .unwrap();
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/events")
                    .header("authorization", format!("Bearer {forged}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn server_token_can_read_telemetry_when_jwt_auth_is_configured() {
        use tower::ServiceExt;
        let root = security_test_root();
        let mut state = security_test_state(&root);
        state.token = Some("operator-token".into());
        let app = app_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/events")
                    .header("authorization", "Bearer operator-token")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn jwt_search_scope_is_bound_to_subject() {
        use tower::ServiceExt;
        let root = security_test_root();
        let app = app_router(security_test_state(&root));
        let token = security_test_token(json!({"sub":"alice", "exp":4_102_444_800u64}));
        let body = json!({
            "query":"some unusual query for session state",
            "user_id":"alice",
            "session_id":"shared-session",
            "scope":"victim-scope",
            "top_k":5
        });
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/search")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut states = crate::conversation_state::ConversationStateStore::open_under(&root);
        assert!(
            states
                .get("victim-scope", "shared-session", now_unix_ms())
                .is_none(),
            "request-provided scope contaminated another user's conversation state"
        );
        assert!(states
            .get("alice", "shared-session", now_unix_ms())
            .is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn provider_memory_batch_writes_to_shared_store_and_search_sees_it() {
        use tower::ServiceExt;

        let root = security_test_root();
        let app = app_router(provider_memory_test_state(&root));
        let add = json!([{
            "request_id":"hermes:test-turn",
            "user_id":"hermes",
            "session_id":"hermes:test-session",
            "messages":[{"role":"assistant","content":"Shared provider memory stores the cobalt telescope calibration procedure."}]
        }]);
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/provider-memory/add/batch")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(add.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let shared_root = crate::integrations::mcp_index::shared_memory_root(&root);
        assert!(
            shared_root.exists(),
            "provider writes must use .lint-ai/memory"
        );

        // Use another integration's identity: server-token provider recall is
        // intentionally shared across provider-owned memories.
        let query = json!({
            "query":"cobalt telescope calibration",
            "user_id":"openclaw",
            "top_k":5
        });
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/provider-memory/search")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(query.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        let hits = body["data"].as_array().unwrap();
        assert!(hits.iter().any(|hit| hit["content"]
            .as_str()
            .unwrap_or_default()
            .contains("cobalt telescope")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn provider_memory_search_includes_workspace_documents() {
        use tower::ServiceExt;

        let root = security_test_root();
        std::fs::write(
            root.join("README.md"),
            "The vermilion observatory uses a brass meridian alignment checklist.",
        )
        .unwrap();
        let app = app_router(provider_memory_test_state(&root));
        let query = json!({
            "query":"vermilion observatory meridian alignment checklist",
            "user_id":"hermes",
            "top_k":5
        });
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/provider-memory/search")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(query.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        let hits = body["data"].as_array().unwrap();
        assert!(hits.iter().any(|hit| hit["content"]
            .as_str()
            .unwrap_or_default()
            .contains("vermilion observatory")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn jwt_provider_memory_search_remains_user_isolated() {
        use tower::ServiceExt;

        let root = security_test_root();
        MemoryService::with_shared_memory(&root, |service| {
            service.add_batch(vec![AddRequest {
                request_id: "bob-private-memory".to_string(),
                user_id: "bob".to_string(),
                session_id: "bob-session".to_string(),
                messages: vec![crate::memory_api::Message {
                    role: "assistant".to_string(),
                    timestamp: None,
                    content: "Bob's private obsidian sundial calibration details.".to_string(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
            }])?;
            Ok(())
        })
        .unwrap();

        let app = app_router(security_test_state(&root));
        let token = security_test_token(json!({"sub":"alice", "exp":4_102_444_800u64}));
        let query = json!({
            "query":"obsidian sundial calibration",
            "user_id":"alice",
            "top_k":5
        });
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/provider-memory/search")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(axum::body::Body::from(query.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert!(body["data"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn server_uses_segmented_memory_index() {
        assert!(matches!(
            memory_pipeline_options(
                None,
                false,
                false,
                false,
                true,
                SegmentRoutingStrategy::TypedEvidenceMultiplicative
            )
            .memory_index_layout,
            MemoryIndexLayout::Segmented { .. }
        ));
    }

    #[test]
    fn server_adaptive_mode_is_opt_in() {
        assert!(matches!(
            memory_pipeline_options(
                Some(8),
                false,
                false,
                false,
                true,
                SegmentRoutingStrategy::TypedEvidenceMultiplicative
            )
            .memory_index_layout,
            MemoryIndexLayout::AdaptiveSegmented {
                query_top_n: 5,
                max_query_n: 8,
                ..
            }
        ));
    }

    #[test]
    fn server_adaptive_limit_at_or_below_base_keeps_fixed_mode() {
        for limit in [0, 2, 3, 4, 5] {
            assert!(matches!(
                memory_pipeline_options(
                    Some(limit),
                    false,
                    false,
                    false,
                    true,
                    SegmentRoutingStrategy::TypedEvidenceMultiplicative
                )
                .memory_index_layout,
                MemoryIndexLayout::Segmented { .. }
            ));
        }
    }

    #[test]
    fn server_default_routing_is_the_chosen_gated_coverage_local() {
        // The benchmark comparison in docs/benchmark-results.md settled on
        // gated coverage-local as the best recall-per-latency trade-off; the
        // server default must track that decision, not a stale hardcoded
        // variant.
        assert_eq!(
            SegmentRoutingArg::GatedCoverageLocal.strategy(),
            SegmentRoutingStrategy::TypedEvidenceMultiplicative
        );
        let options = memory_pipeline_options(
            None,
            false,
            false,
            false,
            true,
            SegmentRoutingArg::GatedCoverageLocal.strategy(),
        );
        assert!(matches!(
            options.memory_index_layout,
            MemoryIndexLayout::Segmented {
                routing_strategy: SegmentRoutingStrategy::TypedEvidenceMultiplicative,
                ..
            }
        ));
    }

    #[test]
    fn segment_routing_flag_selects_each_strategy() {
        for (arg, expected) in [
            (
                SegmentRoutingArg::Sparse,
                SegmentRoutingStrategy::SparseOverlap,
            ),
            (
                SegmentRoutingArg::LocalDistinctiveness,
                SegmentRoutingStrategy::LocalDistinctiveness,
            ),
            (
                SegmentRoutingArg::CoverageLocal,
                SegmentRoutingStrategy::CoverageLocalDistinctiveness,
            ),
            (
                SegmentRoutingArg::CoverageTeam,
                SegmentRoutingStrategy::CoverageTeamSelection,
            ),
            (
                SegmentRoutingArg::TeamCoverageLocal,
                SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness,
            ),
            (
                SegmentRoutingArg::TypedEvidenceAdditive,
                SegmentRoutingStrategy::TypedEvidence,
            ),
            (
                SegmentRoutingArg::GatedCoverageLocal,
                SegmentRoutingStrategy::TypedEvidenceMultiplicative,
            ),
            (
                SegmentRoutingArg::GatedCoverageTeam,
                SegmentRoutingStrategy::CoverageTeamTypedMultiplicative,
            ),
        ] {
            assert_eq!(arg.strategy(), expected);
            let options = memory_pipeline_options(None, false, false, false, true, arg.strategy());
            assert!(matches!(
                options.memory_index_layout,
                MemoryIndexLayout::Segmented {
                    routing_strategy,
                    ..
                } if routing_strategy == expected
            ));
        }
    }

    #[test]
    fn empty_jwt_secret_is_not_considered_configured() {
        assert!(normalize_secret(Some("   ".to_string())).is_none());
    }

    #[test]
    fn search_reads_see_writes_through_the_single_service_lock() {
        let mut service = MemoryService::in_memory(memory_pipeline_options(
            None,
            false,
            false,
            false,
            true,
            SegmentRoutingStrategy::TypedEvidenceMultiplicative,
        ));
        service
            .add(AddRequest {
                request_id: "r1".into(),
                messages: vec![crate::memory_api::Message {
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
        let state = AppState {
            service: Arc::new(RwLock::new(service)),
            writer_gate: Arc::new(tokio::sync::Mutex::new(())),
            staged_writer: None,
            token: None,
            jwt_secret: None,
            tenant_id: None,
            telemetry: OperationalTelemetry::new(),
            project_root: std::env::current_dir().unwrap(),
        };
        // Reads take the shared lock and see the write with no re-publish step.
        let response = state
            .service
            .write()
            .unwrap()
            .search(SearchRequest {
                query: "codename zephyr".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
                session_id: None,
                scope: None,
                filters: None,
            })
            .unwrap();
        assert!(response.data.iter().any(|m| m.content.contains("zephyr")));
    }

    #[test]
    fn token_accepts_supported_forms_and_rejects_wrong_values() {
        assert!(token_is_valid("secret", "secret"));
        assert!(token_is_valid("Bearer secret", "secret"));
        assert!(token_is_valid("Token secret", "secret"));
        assert!(!token_is_valid("Bearer other", "secret"));
        assert!(!token_is_valid("", &"\0".repeat(256)));
    }

    #[test]
    fn non_loopback_bind_requires_opt_in_and_authentication() {
        assert!(ensure_bind_allowed("127.0.0.1:8080", false, false, false).is_ok());
        assert!(ensure_bind_allowed("[::1]:8080", false, false, false).is_ok());
        assert!(ensure_bind_allowed("0.0.0.0:8080", false, true, false).is_err());
        assert!(ensure_bind_allowed("192.168.1.10:8080", true, false, false).is_err());
        assert!(ensure_bind_allowed("0.0.0.0:8080", true, true, false).is_ok());
        assert!(ensure_bind_allowed("0.0.0.0:8080", true, true, true).is_err());
        let blank_token_is_configured = normalize_secret(Some("   ".to_string())).is_some();
        assert!(
            ensure_bind_allowed("0.0.0.0:8080", true, blank_token_is_configured, false).is_err()
        );
    }

    #[tokio::test]
    async fn writer_gate_rejects_a_second_mutation() {
        let gate = tokio::sync::Mutex::new(());
        let _held = gate.lock().await;
        assert!(gate.try_lock().is_err());
    }

    #[cfg(feature = "openclaw")]
    #[tokio::test]
    async fn provider_memory_and_openclaw_http_writers_share_the_same_gate() {
        use tower::ServiceExt;

        let root = security_test_root();
        let state = provider_memory_test_state(&root);
        let gate = state.writer_gate.clone();
        let app = app_router(state);
        let _held = gate.lock().await;

        let writes = [
            (
                "/provider-memory/add/batch",
                json!([{
                    "request_id": "hermes-race-test",
                    "user_id": "hermes",
                    "session_id": "race-test",
                    "messages": [{"role": "user", "content": "race test"}]
                }]),
            ),
            (
                "/integrations/openclaw/hooks/agent-end",
                json!({"event": {"runId": "openclaw-race-test"}, "ctx": {}}),
            ),
        ];

        for (uri, payload) in writes {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "{uri} must be rejected while the shared writer gate is held"
            );
        }
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
