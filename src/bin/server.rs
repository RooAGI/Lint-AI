use axum::{
    error_handling::HandleErrorLayer,
    extract::{Extension, Json, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use jsonwebtoken::{decode, DecodingKey, Validation};
use lint_ai::memory_api::{
    AddRequest, DeleteRequest, MemorySearchService, MemoryService, SearchRequest, SupersedeRequest,
};
use lint_ai::segments::SegmentRoutingStrategy;
use lint_ai::{IndexStore, MemoryIndexLayout, PipelineOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;
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
}

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<MemoryService>>,
    published_search: Arc<RwLock<MemorySearchService>>,
    writer_gate: Arc<tokio::sync::Mutex<()>>,
    token: Option<Arc<str>>,
    jwt_secret: Option<Arc<str>>,
    tenant_id: Option<Arc<str>>,
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
    if args
        .server_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .is_none()
        && jwt_secret.is_none()
        && !args.allow_unauthenticated
        && binds_beyond_loopback(&args.bind)?
    {
        anyhow::bail!("refusing to serve {} without a token", args.bind);
    }
    let options = memory_pipeline_options(
        args.adaptive_segment_max_n,
        args.single_index,
        args.global_index,
    );
    let store = match args.index.as_deref() {
        Some(path) => IndexStore::at_path(path, options)?,
        None => IndexStore::in_memory(options),
    };
    let service = MemoryService::new(store);
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
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/add", post(add))
        .route("/add/batch", post(add_batch))
        .route("/search", post(search))
        .route("/delete", post(delete))
        .route("/supersede", post(supersede))
        .route("/expire", post(expire))
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
    if request.uri().path() == "/health" {
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
    let result = match state.published_search.read() {
        Ok(service) => service.search(request),
        Err(_) => Err(anyhow::anyhow!("published search lock poisoned")),
    };
    match result {
        Ok(result) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail":"search failed"})),
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

fn binds_beyond_loopback(bind: &str) -> anyhow::Result<bool> {
    use std::net::ToSocketAddrs;
    let mut resolved = false;
    for address in bind.to_socket_addrs()? {
        resolved = true;
        if !address.ip().is_loopback() {
            return Ok(true);
        }
    }
    anyhow::ensure!(resolved, "bind address resolved to no addresses: {bind}");
    Ok(false)
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
            memory_pipeline_options(None, false, false).memory_index_layout,
            MemoryIndexLayout::Segmented { .. }
        ));
    }

    #[test]
    fn server_adaptive_mode_is_opt_in() {
        assert!(matches!(
            memory_pipeline_options(Some(8), false, false).memory_index_layout,
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
                memory_pipeline_options(Some(limit), false, false).memory_index_layout,
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
        let service = MemoryService::new(IndexStore::in_memory(memory_pipeline_options(
            None, false, false,
        )));
        let published = service.published_search();
        let state = AppState {
            service: Arc::new(RwLock::new(service)),
            published_search: Arc::new(RwLock::new(published)),
            writer_gate: Arc::new(tokio::sync::Mutex::new(())),
            token: None,
            jwt_secret: None,
            tenant_id: None,
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
