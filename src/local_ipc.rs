//! Local transport for the resident workspace service.
//!
//! Requests use HTTP framing over an OS local stream, so the same Axum routes
//! and JSON contract serve both IPC and the optional TCP listener.

use anyhow::{Context, Result};
use axum::{body::Body, Router};
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus};
use tower::ServiceExt;

#[derive(Clone, Copy)]
pub(crate) struct LocalIpcRequest;

fn runtime_dir(root: &Path) -> Result<PathBuf> {
    let runtime = root.join(".membrick").join("runtime");
    fs::create_dir_all(&runtime).with_context(|| {
        format!(
            "could not create project runtime directory {}",
            runtime.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&runtime, fs::Permissions::from_mode(0o700))?;
    }
    Ok(runtime)
}

#[cfg(unix)]
pub(crate) fn socket_path(root: &Path) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let identity = std::env::var("USER").unwrap_or_default();
    let mut hash = Sha256::new();
    hash.update(identity.as_bytes());
    hash.update([0]);
    hash.update(root.as_os_str().as_encoded_bytes());
    let digest = hash.finalize();
    let name = format!(
        "mb-{}.sock",
        digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    Ok(runtime_dir(root)?.join(name))
}

pub(crate) fn endpoint(root: &Path) -> Result<String> {
    #[cfg(unix)]
    {
        Ok(socket_path(root)?.display().to_string())
    }
    #[cfg(windows)]
    {
        pipe_name(root)
    }
}

#[cfg(windows)]
fn pipe_name(root: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(std::env::var("USERNAME").unwrap_or_default().as_bytes());
    hash.update([0]);
    hash.update(root.as_os_str().as_encoded_bytes());
    let digest = hash.finalize();
    Ok(format!(
        r"\\.\pipe\membrick-{}",
        digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn starting_path(root: &Path) -> Result<PathBuf> {
    #[cfg(unix)]
    {
        return Ok(socket_path(root)?.with_extension("starting"));
    }
    #[cfg(windows)]
    {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(std::env::var("USERNAME").unwrap_or_default().as_bytes());
        hash.update([0]);
        hash.update(root.as_os_str().as_encoded_bytes());
        let digest = hash.finalize();
        let name = digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(runtime_dir(root)?.join(format!("mb-{name}.starting")))
    }
}

fn starting_lock_path(marker: &Path) -> PathBuf {
    marker.with_extension("startup.lock")
}

fn resident_log_path(root: &Path) -> Result<PathBuf> {
    Ok(runtime_dir(root)?.join("resident.log"))
}

pub(crate) fn post_json_with_transport(
    root: &Path,
    path: &str,
    payload: &Value,
) -> Result<(StatusCode, Value, &'static str)> {
    let body = serde_json::to_vec(payload)?;
    let path = path.to_owned();
    let root = root.to_owned();
    let call = || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                match request(&root, &path, body.clone()).await {
                    Ok((status, body)) => Ok((status, body, "ipc")),
                    Err(ipc_error) => match configured_http_bind(&root) {
                        Some(bind) => request_http(&bind, &path, body.clone(), &root)
                            .await
                            .map(|(status, body)| (status, body, "http"))
                            .with_context(|| {
                                format!("resident HTTP transport failed after IPC: {ipc_error}")
                            }),
                        None => Err(ipc_error),
                    },
                }
            })
    };
    match call() {
        Ok(response) => Ok(response),
        Err(first_error) => {
            let marker = starting_path(&root)?;
            let mut startup_owner = start_server(&root).unwrap_or(None);
            // IPC sockets and loopback HTTP can both be unavailable in a
            // sandbox. The resident also consumes atomic file requests from
            // the project runtime directory, so keep the same MemoryService
            // owner without requiring another socket.
            if let Ok((status, body)) = queue_post_json(&root, &path, payload) {
                return Ok((status, body, "file-queue"));
            }
            // A cold resident start also opens and indexes the workspace. On
            // slower machines that can take longer than the former 2 second
            // window, which made the first hook call fail even though the
            // service was still starting.
            // Keep waiting on the process that owns the startup marker. A
            // caller must never remove another process's marker: doing so
            // lets a later hook launch a second writer while the first one
            // is still opening the workspace.
            if let Some(response) = wait_for_startup(
                &marker,
                &mut startup_owner,
                || call().ok(),
                1200,
                std::time::Duration::from_millis(50),
            )? {
                return Ok(response);
            }
            Err(first_error
                .context("could not connect to the automatically started resident service"))
        }
    }
}

fn configured_http_bind(root: &Path) -> Option<String> {
    let settings = fs::read_to_string(root.join(".membrick").join("server.toml")).ok()?;
    settings
        .parse::<toml::Value>()
        .ok()?
        .get("resident_http_bind")?
        .as_str()
        .map(str::to_owned)
}

pub(crate) fn transport_endpoint(root: &Path, transport: &str) -> Option<String> {
    match transport {
        "ipc" => endpoint(root).ok(),
        "http" => configured_http_bind(root),
        "file-queue" => Some(
            root.join(".membrick/runtime/requests")
                .display()
                .to_string(),
        ),
        _ => None,
    }
}

fn http_bind_is_loopback(bind: Option<&str>) -> bool {
    use std::net::ToSocketAddrs;
    bind.and_then(|bind| bind.to_socket_addrs().ok())
        .is_some_and(|addresses| {
            addresses
                .into_iter()
                .all(|address| address.ip().is_loopback())
        })
}

async fn request_http(
    bind: &str,
    path: &str,
    body: Vec<u8>,
    root: &Path,
) -> Result<(StatusCode, Value)> {
    anyhow::ensure!(
        http_bind_is_loopback(Some(bind)),
        "resident HTTP fallback only connects to loopback addresses"
    );
    let identity_stream = tokio::net::TcpStream::connect(bind)
        .await
        .context("resident Membrick HTTP listener unavailable")?;
    let (identity_status, identity) =
        request_over_stream(identity_stream, "/health", Vec::new(), hyper::Method::GET)
            .await
            .context("could not verify resident HTTP workspace")?;
    anyhow::ensure!(
        identity_status.is_success(),
        "resident HTTP workspace identity check returned {identity_status}"
    );
    anyhow::ensure!(
        identity.get("workspace_id").and_then(Value::as_str)
            == Some(workspace_identity(root).as_str()),
        "resident HTTP listener belongs to a different workspace"
    );
    let stream = tokio::net::TcpStream::connect(bind)
        .await
        .context("resident Membrick HTTP listener unavailable")?;
    request_over_stream(stream, path, body, hyper::Method::POST).await
}

fn workspace_identity(root: &Path) -> String {
    use sha2::{Digest, Sha256};
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

async fn request_over_stream<S>(
    stream: S,
    path: &str,
    body: Vec<u8>,
    method: hyper::Method,
) -> Result<(StatusCode, Value)>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let io = TokioIo::new(stream);
    let (mut sender, connection) = hyper::client::conn::http1::handshake(io).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(hyper::header::HOST, "membrick.local")
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::new(hyper::body::Bytes::from(body)))?;
    let response = sender.send_request(request).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    let value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| serde_json::json!({"detail": error.to_string()}));
    Ok((status, value))
}

#[derive(Serialize, Deserialize)]
struct QueueRequest {
    id: String,
    path: String,
    payload: Value,
}

#[derive(Serialize, Deserialize)]
struct QueueResponse {
    status: u16,
    body: Value,
}

const QUEUE_ARTIFACT_RETENTION: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

fn queue_dirs(root: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let base = root.join(".membrick").join("runtime").join("requests");
    let pending = base.join("pending");
    let claimed = base.join("claimed");
    let responses = base.join("responses");
    for directory in [&pending, &claimed, &responses] {
        fs::create_dir_all(directory)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&base, fs::Permissions::from_mode(0o700))?;
        for directory in [&pending, &claimed, &responses] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok((pending, claimed, responses))
}

/// Recover claims left by a resident that exited unexpectedly. The request
/// might already have mutated MemoryService before the crash, so never replay
/// it. Return an explicit uncertain result to a still-waiting client instead.
fn recover_interrupted_claims(root: &Path) -> Result<usize> {
    let (_, claimed, responses) = queue_dirs(root)?;
    let mut recovered = 0;
    for entry in fs::read_dir(&claimed)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let request = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<QueueRequest>(&bytes).ok());
        if let Some(request) = request {
            if path.file_stem().and_then(|value| value.to_str()) != Some(request.id.as_str()) {
                eprintln!(
                    "removing abandoned queue claim with mismatched id {}",
                    path.display()
                );
                fs::remove_file(path)?;
                recovered += 1;
                continue;
            }
            let response_path = responses.join(format!("{}.json", request.id));
            if !response_path.exists() {
                let response = QueueResponse {
                    status: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
                    body: serde_json::json!({
                        "detail": "resident stopped while processing this request; outcome is unknown and the request was not replayed"
                    }),
                };
                let temporary = responses.join(format!("{}.recovery.tmp", request.id));
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&temporary)?;
                file.write_all(&serde_json::to_vec(&response)?)?;
                fs::rename(temporary, response_path)?;
            }
        } else {
            eprintln!(
                "removing malformed abandoned queue claim {}",
                path.display()
            );
        }
        fs::remove_file(path)?;
        recovered += 1;
    }
    Ok(recovered)
}

/// Remove old pending work, orphaned responses, and incomplete temporary
/// files. Claimed requests are handled separately at resident startup, when
/// the workspace owner lock proves the previous resident is gone.
fn cleanup_expired_queue_files(root: &Path, now: std::time::SystemTime) -> Result<usize> {
    let (pending, claimed, responses) = queue_dirs(root)?;
    let mut removed = 0;
    for directory in [&pending, &claimed, &responses] {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let extension = path.extension().and_then(|value| value.to_str());
            if extension != Some("json") && extension != Some("tmp") {
                continue;
            }
            if directory == &claimed && extension == Some("json") {
                continue;
            }
            let modified = match entry.metadata()?.modified() {
                Ok(modified) => modified,
                Err(_) => continue,
            };
            if now.duration_since(modified).unwrap_or_default() >= QUEUE_ARTIFACT_RETENTION {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn watch_directory(
    directory: &Path,
    wake: std::sync::Arc<tokio::sync::Notify>,
) -> Result<RecommendedWatcher> {
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if event.is_ok() {
            wake.notify_one();
        }
    })?;
    watcher.watch(directory, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

pub(crate) fn queue_post_json(
    root: &Path,
    path: &str,
    payload: &Value,
) -> Result<(StatusCode, Value)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let (pending, _, responses) = queue_dirs(root)?;
    let (events_tx, events_rx) = std::sync::mpsc::channel();
    let mut response_watcher =
        notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
            if event.is_ok() {
                let _ = events_tx.send(());
            }
        })
        .ok();
    if response_watcher.as_mut().is_some_and(|watcher| {
        watcher
            .watch(&responses, RecursiveMode::NonRecursive)
            .is_err()
    }) {
        response_watcher = None;
    }
    let id = format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let request = QueueRequest {
        id: id.clone(),
        path: path.to_owned(),
        payload: payload.clone(),
    };
    let temporary = pending.join(format!("{id}.tmp"));
    let target = pending.join(format!("{id}.json"));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&serde_json::to_vec(&request)?)?;
    fs::rename(temporary, target)?;

    let response = responses.join(format!("{id}.json"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        match fs::read(&response) {
            Ok(bytes) => {
                let value: QueueResponse = serde_json::from_slice(&bytes)?;
                let _ = fs::remove_file(response);
                let status =
                    StatusCode::from_u16(value.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Ok((status, value.body));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if response_watcher.is_some() {
                    let _ = events_rx.recv_timeout(std::time::Duration::from_millis(250));
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("resident file-queue request timed out; outcome may be unknown")
}

fn wait_for_startup<T>(
    marker: &Path,
    startup_owner: &mut Option<StartupOwner>,
    mut try_request: impl FnMut() -> Option<T>,
    attempts: usize,
    delay: std::time::Duration,
) -> Result<Option<T>> {
    for _ in 0..attempts {
        std::thread::sleep(delay);
        if let Some(response) = try_request() {
            return Ok(Some(response));
        }
        if let Some(owner) = startup_owner.as_mut() {
            if let Some(status) = owner.child.try_wait()? {
                remove_startup_marker_if_owned(&owner.marker, &owner.token);
                return Err(startup_exit_error(status));
            }
        }
        // The startup owner removes the marker after binding, or removes its
        // own marker if launch fails. If it vanished while the endpoint is
        // still unavailable, fail promptly and let the next request retry.
        if !marker.exists() {
            return Ok(None);
        }
    }
    // A timed-out waiter must leave another starter's marker untouched.
    Ok(None)
}

struct StartupOwner {
    child: Child,
    marker: PathBuf,
    token: String,
    _startup_lock: fs::File,
}

fn start_server(root: &Path) -> Result<Option<StartupOwner>> {
    let marker = starting_path(root)?;
    let Some(claim) = claim_startup(&marker)? else {
        return Ok(None);
    };
    let token = claim.token.clone();

    let launch = (|| -> Result<Child> {
        let executable =
            std::env::current_exe().context("could not locate the Membrick executable")?;
        let log_path = resident_log_path(root)?;
        if std::fs::metadata(&log_path).is_ok_and(|metadata| metadata.len() > 1_048_576) {
            let previous = log_path.with_extension("log.1");
            let _ = std::fs::remove_file(&previous);
            std::fs::rename(&log_path, previous)?;
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        std::process::Command::new(executable)
            .arg("--resident-server")
            .current_dir(root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::from(log))
            .spawn()
            .context("could not start the resident Membrick service")
    })();

    match launch {
        Ok(child) => Ok(Some(StartupOwner {
            child,
            marker,
            token,
            _startup_lock: claim.lock,
        })),
        Err(error) => {
            remove_startup_marker_if_owned(&marker, &token);
            Err(error)
        }
    }
}

struct StartupClaim {
    token: String,
    lock: fs::File,
}

/// The kernel lock establishes ownership; the `.starting` file is only a
/// status record. A new owner may safely replace stale status after acquiring
/// the lock because locks are released by the OS if the previous starter dies.
fn claim_startup(marker: &Path) -> Result<Option<StartupClaim>> {
    use std::io::Write;
    static NEXT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let lock_path = starting_lock_path(marker);
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("could not open startup lock {}", lock_path.display()))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(fs::TryLockError::WouldBlock) => return Ok(None),
        Err(fs::TryLockError::Error(error)) => {
            return Err(error).with_context(|| {
                format!("could not acquire startup lock {}", lock_path.display())
            });
        }
    }

    let token = format!(
        "{}:{}:{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    // We own the kernel lock, so any existing marker can only be stale.
    match fs::remove_file(marker) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker)
    {
        Ok(mut file) => {
            if let Err(error) = file.write_all(token.as_bytes()) {
                drop(file);
                let _ = std::fs::remove_file(marker);
                return Err(error.into());
            }
            Ok(Some(StartupClaim { token, lock }))
        }
        Err(error) => Err(error.into()),
    }
}

fn remove_startup_marker_if_owned(marker: &Path, token: &str) {
    if std::fs::read_to_string(marker).is_ok_and(|contents| contents == token) {
        let _ = std::fs::remove_file(marker);
    }
}

fn startup_exit_error(status: ExitStatus) -> anyhow::Error {
    anyhow::anyhow!("resident Membrick service exited before becoming available ({status})")
}

async fn request(root: &Path, path: &str, body: Vec<u8>) -> Result<(StatusCode, Value)> {
    #[cfg(unix)]
    let stream = {
        let endpoint = socket_path(root)?;
        tokio::net::UnixStream::connect(endpoint)
            .await
            .context("resident Membrick IPC server unavailable")?
    };
    #[cfg(windows)]
    let stream = {
        let endpoint = pipe_name(root)?;
        tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint)?
    };

    let io = TokioIo::new(stream);
    let (mut sender, connection) = hyper::client::conn::http1::handshake(io).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let request = Request::post(path)
        .header(hyper::header::HOST, "membrick.local")
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::new(hyper::body::Bytes::from(body)))?;
    let response = sender.send_request(request).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    let value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| serde_json::json!({"detail": error.to_string()}));
    Ok((status, value))
}

pub(crate) async fn process_queue_once(router: &Router, root: &Path) -> Result<()> {
    const MAX_IN_FLIGHT: usize = 32;
    let root = root.to_owned();
    let claims =
        tokio::task::spawn_blocking(move || claim_pending_requests(&root, MAX_IN_FLIGHT)).await??;
    let mut tasks = tokio::task::JoinSet::new();
    for (claim, responses) in claims {
        let router = router.clone();
        tasks.spawn(async move {
            if let Err(error) = process_claimed_request(router, claim.clone(), responses).await {
                eprintln!("resident file-queue request failed: {error:#}");
            }
            let claim_to_remove = claim;
            let _ = tokio::task::spawn_blocking(move || fs::remove_file(claim_to_remove)).await;
        });
    }
    while tasks.join_next().await.is_some() {}
    Ok(())
}

/// Atomically claim a bounded batch. Filesystem enumeration, rename, and file
/// reads stay off the resident's async executor threads.
fn claim_pending_requests(root: &Path, limit: usize) -> Result<Vec<(PathBuf, PathBuf)>> {
    let (pending, claimed, responses) = queue_dirs(root)?;
    let mut entries = fs::read_dir(&pending)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    let mut claims = Vec::with_capacity(limit.min(entries.len()));
    for entry in entries.into_iter().take(limit) {
        let source = entry.path();
        let claim = claimed.join(entry.file_name());
        // Atomic rename gives exactly one resident worker ownership of a
        // request, even if multiple workers are accidentally started.
        if fs::rename(&source, &claim).is_ok() {
            claims.push((claim, responses.clone()));
        }
    }
    Ok(claims)
}

async fn process_claimed_request(router: Router, claim: PathBuf, responses: PathBuf) -> Result<()> {
    let claim_for_read = claim.clone();
    let request: QueueRequest = tokio::task::spawn_blocking(move || {
        serde_json::from_slice(&fs::read(claim_for_read)?).map_err(anyhow::Error::from)
    })
    .await??;
    anyhow::ensure!(
        request.path.starts_with('/') && !request.path.starts_with("//"),
        "invalid queued route"
    );
    let mut http_request = Request::post(&request.path)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&request.payload)?))?;
    http_request.extensions_mut().insert(LocalIpcRequest);
    let response = router
        .oneshot(http_request)
        .await
        .unwrap_or_else(|never| match never {});
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| serde_json::json!({"detail": error.to_string()}));
    let output = QueueResponse {
        status: status.as_u16(),
        body,
    };
    let response_path = responses.join(format!("{}.json", request.id));
    let temporary = responses.join(format!("{}.tmp", request.id));
    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec(&output)?)?;
        fs::rename(temporary, response_path)?;
        Ok(())
    })
    .await??;
    Ok(())
}

/// Serve the workspace endpoint until `shutdown` completes. On Unix the
/// socket file is removed on shutdown, so a stopped owner leaves no stale
/// endpoint behind.
pub(crate) async fn serve(
    router: Router,
    root: &Path,
    shutdown: impl std::future::Future<Output = ()>,
    http_bind: Option<&str>,
    auto_http_port: bool,
    settings_path: Option<&Path>,
) -> Result<()> {
    tokio::pin!(shutdown);
    let cleanup_root = root.to_owned();
    match tokio::task::spawn_blocking(move || {
        let recovered = recover_interrupted_claims(&cleanup_root)?;
        let expired = cleanup_expired_queue_files(&cleanup_root, std::time::SystemTime::now())?;
        Ok::<_, anyhow::Error>((recovered, expired))
    })
    .await
    {
        Ok(Ok((recovered, expired))) if recovered > 0 || expired > 0 => eprintln!(
            "recovered {recovered} interrupted file-queue requests and removed {expired} expired queue files"
        ),
        Ok(Ok(_)) => {}
        Ok(Err(error)) => eprintln!("file-queue startup cleanup failed: {error:#}"),
        Err(error) => eprintln!("file-queue startup cleanup worker failed: {error}"),
    }
    let mut http_listener = if let Some(bind) = http_bind {
        match bind_resident_http(bind, auto_http_port, settings_path).await {
            Ok(listener) => {
                eprintln!(
                    "Membrick resident HTTP listening on {}",
                    listener.local_addr()?
                );
                Some(listener)
            }
            Err(error) => {
                eprintln!(
                    "resident HTTP listener unavailable; continuing with local transports: {error}"
                );
                None
            }
        }
    } else {
        None
    };
    let queue_wakeup = std::sync::Arc::new(tokio::sync::Notify::new());
    let _queue_watcher = match queue_dirs(root) {
        Ok((pending, _, _)) => match watch_directory(&pending, queue_wakeup.clone()) {
            Ok(watcher) => Some(watcher),
            Err(error) => {
                eprintln!("resident queue watcher unavailable; using recovery scans: {error}");
                None
            }
        },
        Err(error) => {
            eprintln!("resident queue setup failed; using recovery scans: {error:#}");
            None
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use tokio::net::UnixListener;
        let endpoint = socket_path(root)?;
        let mut listener = if endpoint.exists() {
            // Do not unlink a live owner's socket. Remove only a stale socket
            // path left behind after its process exited; bind below remains
            // the final arbiter if multiple starters race here.
            if tokio::net::UnixStream::connect(&endpoint).await.is_ok() {
                anyhow::bail!("a resident Membrick service already owns this workspace");
            }
            std::fs::remove_file(&endpoint)?;
            match UnixListener::bind(&endpoint) {
                Ok(listener) => Some(listener),
                Err(error) => {
                    eprintln!("resident IPC socket unavailable; using file queue: {error}");
                    None
                }
            }
        } else {
            match UnixListener::bind(&endpoint) {
                Ok(listener) => Some(listener),
                Err(error) => {
                    eprintln!("resident IPC socket unavailable; using file queue: {error}");
                    None
                }
            }
        };
        if listener.is_some() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&endpoint, std::fs::Permissions::from_mode(0o600))?;
        }
        // Remember which socket file this process created, so shutdown never
        // removes a newer owner's endpoint at the same path.
        let bound_inode = listener
            .as_ref()
            .and_then(|_| std::fs::metadata(&endpoint).ok())
            .map(|metadata| metadata.ino());
        let _ = std::fs::remove_file(starting_path(root)?);
        if listener.is_some() {
            eprintln!("Membrick resident IPC listening on {}", endpoint.display());
        }
        let recovery_period = if _queue_watcher.is_some() {
            std::time::Duration::from_secs(1)
        } else {
            std::time::Duration::from_millis(20)
        };
        let mut recovery = tokio::time::interval(recovery_period);
        let mut queue_cleanup = tokio::time::interval_at(
            tokio::time::Instant::now() + std::time::Duration::from_secs(15 * 60),
            std::time::Duration::from_secs(15 * 60),
        );
        let result = loop {
            tokio::select! {
                accepted = async {
                    match listener.as_ref() {
                        Some(listener) => listener.accept().await.map(Some),
                        None => std::future::pending().await,
                    }
                } => match accepted {
                    Ok(Some((stream, _))) => serve_stream(router.clone(), stream, true),
                    Ok(None) => unreachable!(),
                    Err(error) => {
                        eprintln!("resident IPC listener failed; continuing with file queue: {error}");
                        listener = None;
                    },
                },
                accepted = async {
                    match http_listener.as_ref() {
                        Some(listener) => listener.accept().await.map(Some),
                        None => std::future::pending().await,
                    }
                } => match accepted {
                    Ok(Some((stream, _))) => serve_stream(router.clone(), stream, http_bind_is_loopback(http_bind)),
                    Ok(None) => unreachable!(),
                    Err(error) => {
                        eprintln!("resident HTTP listener stopped: {error}");
                        http_listener = None;
                    },
                },
                _ = queue_wakeup.notified() => if let Err(error) = process_queue_once(&router, root).await {
                    eprintln!("resident file-queue poll failed: {error:#}");
                },
                _ = recovery.tick() => if let Err(error) = process_queue_once(&router, root).await {
                    eprintln!("resident file-queue recovery scan failed: {error:#}");
                },
                _ = queue_cleanup.tick() => {
                    let cleanup_root = root.to_owned();
                    match tokio::task::spawn_blocking(move || {
                        cleanup_expired_queue_files(&cleanup_root, std::time::SystemTime::now())
                    }).await {
                        Ok(Ok(removed)) if removed > 0 => eprintln!("removed {removed} expired file-queue artifacts"),
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => eprintln!("file-queue cleanup failed: {error:#}"),
                        Err(error) => eprintln!("file-queue cleanup worker failed: {error}"),
                    }
                }
                () = &mut shutdown => break Ok(()),
            }
        };
        drop(listener);
        if bound_inode.is_some()
            && std::fs::metadata(&endpoint)
                .is_ok_and(|metadata| Some(metadata.ino()) == bound_inode)
        {
            let _ = std::fs::remove_file(&endpoint);
        }
        eprintln!("Membrick resident stopped on {}", endpoint.display());
        result
    }
    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ServerOptions;
        let endpoint = pipe_name(root)?;
        eprintln!("Membrick resident IPC listening on {endpoint}");
        let recovery_period = if _queue_watcher.is_some() {
            std::time::Duration::from_secs(1)
        } else {
            std::time::Duration::from_millis(20)
        };
        let mut recovery = tokio::time::interval(recovery_period);
        let mut queue_cleanup = tokio::time::interval_at(
            tokio::time::Instant::now() + std::time::Duration::from_secs(15 * 60),
            std::time::Duration::from_secs(15 * 60),
        );
        loop {
            let server = ServerOptions::new().create(&endpoint)?;
            let _ = std::fs::remove_file(starting_path(root)?);
            tokio::select! {
                connected = server.connect() => {
                    connected?;
                    serve_stream(router.clone(), server, true);
                }
                accepted = async {
                    match http_listener.as_ref() {
                        Some(listener) => listener.accept().await.map(Some),
                        None => std::future::pending().await,
                    }
                } => match accepted {
                    Ok(Some((stream, _))) => {
                        serve_stream(router.clone(), stream, http_bind_is_loopback(http_bind))
                    }
                    Ok(None) => unreachable!(),
                    Err(error) => {
                        eprintln!("resident HTTP listener stopped: {error}");
                        http_listener = None;
                    },
                },
                _ = queue_wakeup.notified() => if let Err(error) = process_queue_once(&router, root).await {
                    eprintln!("resident file-queue poll failed: {error:#}");
                },
                _ = recovery.tick() => if let Err(error) = process_queue_once(&router, root).await {
                    eprintln!("resident file-queue recovery scan failed: {error:#}");
                },
                _ = queue_cleanup.tick() => {
                    let cleanup_root = root.to_owned();
                    match tokio::task::spawn_blocking(move || {
                        cleanup_expired_queue_files(&cleanup_root, std::time::SystemTime::now())
                    }).await {
                        Ok(Ok(removed)) if removed > 0 => eprintln!("removed {removed} expired file-queue artifacts"),
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => eprintln!("file-queue cleanup failed: {error:#}"),
                        Err(error) => eprintln!("file-queue cleanup worker failed: {error}"),
                    }
                }
                () = &mut shutdown => return Ok(()),
            }
        }
    }
}

async fn bind_resident_http(
    bind: &str,
    auto_port: bool,
    settings_path: Option<&Path>,
) -> Result<tokio::net::TcpListener> {
    match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => Ok(listener),
        Err(error) if auto_port && error.kind() == std::io::ErrorKind::AddrInUse => {
            let requested: std::net::SocketAddr = bind.parse().with_context(|| {
                format!(
                    "automatic resident HTTP port selection requires an IP:port bind, got {bind}"
                )
            })?;
            const FIRST_PORT: u16 = 8465;
            const LAST_PORT: u16 = 9999;
            let count = u32::from(LAST_PORT - FIRST_PORT) + 1;
            let preferred_offset = requested.port().saturating_sub(FIRST_PORT) as u32;
            for offset in 1..count {
                let port = FIRST_PORT + ((preferred_offset + offset) % count) as u16;
                let candidate = std::net::SocketAddr::new(requested.ip(), port);
                match tokio::net::TcpListener::bind(candidate).await {
                    Ok(listener) => {
                        let actual = listener.local_addr()?;
                        let Some(settings_path) = settings_path else {
                            drop(listener);
                            anyhow::bail!(
                                "automatic resident HTTP port selection requires a settings file"
                            );
                        };
                        if let Err(error) = persist_resident_http_bind(settings_path, actual) {
                            drop(listener);
                            return Err(error).context(
                                "could not persist the resident HTTP port selected after a bind conflict",
                            );
                        }
                        eprintln!(
                            "resident HTTP port {} was occupied; using {}",
                            requested.port(),
                            actual.port()
                        );
                        return Ok(listener);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error).context(format!(
                "no available resident HTTP port in {FIRST_PORT}..={LAST_PORT}"
            ))
        }
        Err(error) => Err(error.into()),
    }
}

fn persist_resident_http_bind(settings_path: &Path, bind: std::net::SocketAddr) -> Result<()> {
    let contents = fs::read_to_string(&settings_path)
        .with_context(|| format!("could not read {}", settings_path.display()))?;
    let mut settings = contents
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("could not parse {}", settings_path.display()))?;
    let table = settings.as_table_mut();
    table.insert("resident_http_bind", toml_edit::value(bind.to_string()));
    table.insert(
        "dashboard_url",
        toml_edit::value(format!("http://{bind}/dashboard")),
    );

    let temporary = settings_path.with_file_name(format!(
        ".server.toml.{}.{}.tmp",
        std::process::id(),
        bind.port()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("could not create {}", temporary.display()))?;
    if let Err(error) = file.write_all(settings.to_string().as_bytes()) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    drop(file);
    #[cfg(windows)]
    if settings_path.exists() {
        fs::remove_file(&settings_path)?;
    }
    if let Err(error) = fs::rename(&temporary, &settings_path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn serve_stream<S>(router: Router, stream: S, trusted_local: bool)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let service = hyper::service::service_fn(move |request: Request<Incoming>| {
            let router = router.clone();
            async move {
                let mut request = request.map(Body::new);
                if trusted_local {
                    request.extensions_mut().insert(LocalIpcRequest);
                }
                let response = router
                    .oneshot(request)
                    .await
                    .unwrap_or_else(|never| match never {});
                Ok::<Response<Body>, std::convert::Infallible>(response)
            }
        });
        let io = TokioIo::new(stream);
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(io, service)
            .await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::State,
        routing::{get, post},
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "membrick-local-ipc-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resident_runtime_paths_are_project_local() {
        let directory = TempDir::new();
        let root = directory.0.join("project");
        std::fs::create_dir_all(&root).unwrap();

        let runtime = root.join(".membrick").join("runtime");
        assert_eq!(runtime_dir(&root).unwrap(), runtime);
        assert_eq!(
            starting_path(&root).unwrap().parent(),
            Some(runtime.as_path())
        );
        assert_eq!(
            starting_lock_path(&starting_path(&root).unwrap()).parent(),
            Some(runtime.as_path())
        );
        assert_eq!(
            resident_log_path(&root).unwrap(),
            runtime.join("resident.log")
        );
        #[cfg(unix)]
        assert_eq!(
            socket_path(&root).unwrap().parent(),
            Some(runtime.as_path())
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&runtime).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn failed_startup_wait_does_not_remove_the_owner_marker() {
        let directory = TempDir::new();
        let marker = directory.0.join("resident.starting");
        std::fs::write(&marker, "owner").unwrap();

        let result =
            wait_for_startup::<()>(&marker, &mut None, || None, 1, std::time::Duration::ZERO)
                .unwrap();

        assert!(result.is_none());
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "owner");
    }

    #[test]
    fn stale_startup_marker_can_be_recovered_by_a_new_owner() {
        let directory = TempDir::new();
        let marker = directory.0.join("resident.starting");
        std::fs::write(&marker, "dead-process-token").unwrap();

        let claim = claim_startup(&marker).unwrap();

        assert!(claim.is_some(), "a stale marker must not block startup");
        assert_ne!(
            std::fs::read_to_string(marker).unwrap(),
            "dead-process-token",
            "the acquired startup owner should replace stale marker status"
        );
    }

    #[test]
    fn startup_lock_blocks_another_starter_until_the_owner_releases_it() {
        let directory = TempDir::new();
        let marker = directory.0.join("resident.starting");
        let first = claim_startup(&marker).unwrap().unwrap();

        assert!(claim_startup(&marker).unwrap().is_none());
        drop(first);
        assert!(claim_startup(&marker).unwrap().is_some());
    }

    #[test]
    fn concurrent_startup_claims_have_one_owner() {
        let directory = TempDir::new();
        let marker = Arc::new(directory.0.join("resident.starting"));
        let winners = Arc::new(Mutex::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let marker = Arc::clone(&marker);
                let winners = Arc::clone(&winners);
                std::thread::spawn(move || {
                    if let Some(_claim) = claim_startup(marker.as_ref()).unwrap() {
                        *winners.lock().unwrap() += 1;
                        std::thread::sleep(std::time::Duration::from_millis(30));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*winners.lock().unwrap(), 1);
    }

    #[test]
    fn interrupted_queue_claim_returns_uncertain_result_without_replaying_request() {
        let directory = TempDir::new();
        let (_, claimed, responses) = queue_dirs(&directory.0).unwrap();
        let id = "interrupted-write";
        let request = QueueRequest {
            id: id.to_owned(),
            path: "/memory/add".to_owned(),
            payload: serde_json::json!({"text": "may already have been written"}),
        };
        fs::write(
            claimed.join(format!("{id}.json")),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();

        recover_interrupted_claims(&directory.0).unwrap();

        assert!(!claimed.join(format!("{id}.json")).exists());
        let (pending, _, _) = queue_dirs(&directory.0).unwrap();
        assert!(!pending.join(format!("{id}.json")).exists());
        let response: QueueResponse =
            serde_json::from_slice(&fs::read(responses.join(format!("{id}.json"))).unwrap())
                .unwrap();
        assert_eq!(response.status, StatusCode::SERVICE_UNAVAILABLE.as_u16());
        assert!(response.body["detail"]
            .as_str()
            .unwrap()
            .contains("outcome is unknown"));
    }

    #[test]
    fn expired_pending_and_orphan_response_files_are_removed_but_recent_files_remain() {
        let directory = TempDir::new();
        let (pending, _, responses) = queue_dirs(&directory.0).unwrap();
        let expired_pending = pending.join("expired.json");
        let recent_pending = pending.join("recent.json");
        let expired_response = responses.join("orphan.json");
        let recent_response = responses.join("active.json");
        for path in [
            &expired_pending,
            &recent_pending,
            &expired_response,
            &recent_response,
        ] {
            fs::write(path, b"stale queue artifact").unwrap();
        }
        let now = std::time::SystemTime::now();
        let old = now - QUEUE_ARTIFACT_RETENTION - std::time::Duration::from_secs(1);
        for path in [&expired_pending, &expired_response] {
            fs::File::open(path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(old))
                .unwrap();
        }

        cleanup_expired_queue_files(&directory.0, now).unwrap();

        assert!(!expired_pending.exists());
        assert!(!expired_response.exists());
        assert!(recent_pending.exists());
        assert!(recent_response.exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_file_queue_requests_are_claimed_once_and_get_their_own_responses() {
        let directory = TempDir::new();
        let handled = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/echo",
                post(
                    |State(handled): State<Arc<AtomicUsize>>,
                     axum::Json(payload): axum::Json<Value>| async move {
                        let sequence = handled.fetch_add(1, Ordering::SeqCst) + 1;
                        axum::Json(
                            serde_json::json!({"echo": payload["value"], "sequence": sequence}),
                        )
                    },
                ),
            )
            .with_state(handled.clone());

        let workers: Vec<_> = (0..2)
            .map(|_| {
                let app = app.clone();
                let root = directory.0.clone();
                let handled = handled.clone();
                tokio::spawn(async move {
                    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
                    while handled.load(Ordering::SeqCst) < 24
                        && tokio::time::Instant::now() < deadline
                    {
                        let _ = process_queue_once(&app, &root).await;
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    }
                })
            })
            .collect();

        let clients: Vec<_> = (0..24)
            .map(|value| {
                let root = directory.0.clone();
                tokio::task::spawn_blocking(move || {
                    let response =
                        queue_post_json(&root, "/echo", &serde_json::json!({"value": value}))
                            .unwrap();
                    (value, response)
                })
            })
            .collect();
        for client in clients {
            let (expected, (status, response)) = client.await.unwrap();
            assert_eq!(status, StatusCode::OK);
            assert!(response["sequence"].as_u64().unwrap() >= 1);
            assert_eq!(response["echo"].as_u64().unwrap(), expected);
        }
        for worker in workers {
            worker.abort();
        }
        assert_eq!(handled.load(Ordering::SeqCst), 24);
    }

    #[tokio::test]
    async fn one_queue_scan_dispatches_multiple_requests_concurrently() {
        let directory = TempDir::new();
        let rendezvous = Arc::new(tokio::sync::Barrier::new(4));
        let app = Router::new()
            .route(
                "/parallel",
                post(
                    |State(rendezvous): State<Arc<tokio::sync::Barrier>>,
                     axum::Json(payload): axum::Json<Value>| async move {
                        rendezvous.wait().await;
                        axum::Json(payload)
                    },
                ),
            )
            .with_state(rendezvous);
        let (pending, _, _) = queue_dirs(&directory.0).unwrap();
        for index in 0..4 {
            let id = format!("parallel-{index}");
            let request = QueueRequest {
                id: id.clone(),
                path: "/parallel".into(),
                payload: serde_json::json!({"index": index}),
            };
            fs::write(
                pending.join(format!("{id}.json")),
                serde_json::to_vec(&request).unwrap(),
            )
            .unwrap();
        }

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            process_queue_once(&app, &directory.0),
        )
        .await
        .expect("queue requests should be dispatched concurrently")
        .unwrap();
        let (_, _, responses) = queue_dirs(&directory.0).unwrap();
        for index in 0..4 {
            let response: QueueResponse = serde_json::from_slice(
                &fs::read(responses.join(format!("parallel-{index}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(response.status, StatusCode::OK.as_u16());
            assert_eq!(response.body["index"], index);
        }
    }

    #[tokio::test]
    async fn queue_directory_notifications_wake_waiters() {
        let directory = TempDir::new();
        let (pending, _, _) = queue_dirs(&directory.0).unwrap();
        let wake = Arc::new(tokio::sync::Notify::new());
        let _watcher = watch_directory(&pending, wake.clone()).unwrap();
        let write_dir = pending.clone();
        tokio::task::spawn_blocking(move || {
            fs::write(write_dir.join("request.tmp"), b"request").unwrap();
            fs::rename(
                write_dir.join("request.tmp"),
                write_dir.join("request.json"),
            )
            .unwrap();
        })
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), wake.notified())
            .await
            .expect("filesystem change should wake the resident queue");
    }

    #[tokio::test]
    async fn loopback_http_listener_uses_the_same_router_contract() {
        let directory = TempDir::new();
        let identity = super::workspace_identity(&directory.0);
        let handled = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let identity = identity.clone();
                    async move { axum::Json(serde_json::json!({"workspace_id": identity})) }
                }),
            )
            .route(
                "/echo",
                post(
                    |State(handled): State<Arc<AtomicUsize>>,
                     axum::Json(payload): axum::Json<Value>| async move {
                        handled.fetch_add(1, Ordering::SeqCst);
                        axum::Json(payload)
                    },
                ),
            )
            .with_state(handled.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bind = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                serve_stream(app.clone(), stream, true);
            }
        });
        let (status, result) = request_http(
            &bind,
            "/echo",
            serde_json::to_vec(&serde_json::json!({"value": "resident-http"})).unwrap(),
            &directory.0,
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result["value"], "resident-http");
        assert_eq!(handled.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn http_fallback_refuses_a_resident_from_another_workspace() {
        let directory = TempDir::new();
        let other_workspace_id = super::workspace_identity(&directory.0.join("other"));
        let handled = Arc::new(AtomicUsize::new(0));
        let app =
            Router::new()
                .route(
                    "/health",
                    get(move || {
                        let other_workspace_id = other_workspace_id.clone();
                        async move {
                            axum::Json(serde_json::json!({"workspace_id": other_workspace_id}))
                        }
                    }),
                )
                .route(
                    "/echo",
                    post(
                        |State(handled): State<Arc<AtomicUsize>>,
                         axum::Json(payload): axum::Json<Value>| async move {
                            handled.fetch_add(1, Ordering::SeqCst);
                            axum::Json(payload)
                        },
                    ),
                )
                .with_state(handled.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bind = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            serve_stream(app, stream, true);
        });

        let result = request_http(&bind, "/echo", Vec::new(), &directory.0).await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("different workspace"));
        assert_eq!(handled.load(Ordering::SeqCst), 0);
        server.await.unwrap();
    }
}

#[cfg(test)]
mod resident_http_port_tests {
    use super::bind_resident_http;
    use std::fs;
    use std::path::{Path, PathBuf};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "membrick-resident-http-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_projects_claim_distinct_http_ports_and_persist_them() {
        let directory = TempDir::new();
        let first_settings = directory.0.join("first/.membrick/server.toml");
        let second_settings = directory.0.join("second/.membrick/server.toml");
        fs::create_dir_all(first_settings.parent().unwrap()).unwrap();
        fs::create_dir_all(second_settings.parent().unwrap()).unwrap();

        let (port, blocker) = (8465..=9999)
            .find_map(|port| {
                std::net::TcpListener::bind(("127.0.0.1", port))
                    .ok()
                    .map(|listener| (port, listener))
            })
            .expect("test port range should have a free port");
        let config = format!("resident_http_bind = \"127.0.0.1:{port}\"\n");
        fs::write(&first_settings, &config).unwrap();
        fs::write(&second_settings, &config).unwrap();

        let bind = format!("127.0.0.1:{port}");
        let (first, second) = tokio::join!(
            bind_resident_http(&bind, true, Some(&first_settings)),
            bind_resident_http(&bind, true, Some(&second_settings))
        );
        let first = first.unwrap();
        let second = second.unwrap();
        let first_addr = first.local_addr().unwrap();
        let second_addr = second.local_addr().unwrap();
        assert_ne!(first_addr, second_addr);

        for (settings_path, address) in [
            (&first_settings, first_addr),
            (&second_settings, second_addr),
        ] {
            let parsed: toml::Value = fs::read_to_string(settings_path).unwrap().parse().unwrap();
            let expected_bind = address.to_string();
            let expected_dashboard = format!("http://{address}/dashboard");
            assert_eq!(
                parsed["resident_http_bind"].as_str(),
                Some(expected_bind.as_str())
            );
            assert_eq!(
                parsed["dashboard_url"].as_str(),
                Some(expected_dashboard.as_str())
            );
        }
        drop((first, second, blocker));
    }

    #[test]
    fn workspace_identity_is_stable_and_project_specific() {
        let first = Path::new("/tmp/membrick-project-a");
        let second = Path::new("/tmp/membrick-project-b");
        assert_eq!(
            super::workspace_identity(first),
            super::workspace_identity(first)
        );
        assert_ne!(
            super::workspace_identity(first),
            super::workspace_identity(second)
        );
    }
}
