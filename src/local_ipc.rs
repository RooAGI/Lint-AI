//! Local transport for the resident workspace service.
//!
//! Requests use HTTP framing over an OS local stream, so the same Axum routes
//! and JSON contract serve both IPC and the optional TCP listener.

use anyhow::{Context, Result};
use axum::{body::Body, Router};
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tower::ServiceExt;

#[derive(Clone, Copy)]
pub(crate) struct LocalIpcRequest;

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
    let home = std::env::var_os("HOME").context("HOME is required for the resident socket")?;
    let base = PathBuf::from(home).join(".membrick").join("runtime");
    std::fs::create_dir_all(&base)?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(base.join(name))
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
        let base = PathBuf::from(
            std::env::var_os("USERPROFILE")
                .context("USERPROFILE is required for the resident service")?,
        )
        .join(".membrick")
        .join("runtime");
        std::fs::create_dir_all(&base)?;
        Ok(base.join(format!("mb-{name}.starting")))
    }
}

fn resident_log_path() -> Result<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .context("home directory is required for resident service logs")?;
    let base = PathBuf::from(home).join(".membrick").join("runtime");
    std::fs::create_dir_all(&base)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(base.join("resident.log"))
}

pub(crate) fn post_json(root: &Path, path: &str, payload: &Value) -> Result<(StatusCode, Value)> {
    let body = serde_json::to_vec(payload)?;
    let path = path.to_owned();
    let root = root.to_owned();
    let call = || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async { request(&root, &path, body.clone()).await })
    };
    match call() {
        Ok(response) => Ok(response),
        Err(first_error) => {
            start_server(&root)?;
            // A cold resident start also opens and indexes the workspace. On
            // slower machines that can take longer than the former 2 second
            // window, which made the first hook call fail even though the
            // service was still starting.
            for _ in 0..200 {
                std::thread::sleep(std::time::Duration::from_millis(50));
                if let Ok(response) = call() {
                    return Ok(response);
                }
            }
            let _ = std::fs::remove_file(starting_path(&root)?);
            Err(first_error
                .context("could not connect to the automatically started resident service"))
        }
    }
}

fn start_server(root: &Path) -> Result<()> {
    #[cfg(unix)]
    let endpoint = socket_path(root)?;
    #[cfg(windows)]
    let endpoint = PathBuf::from(pipe_name(root)?);
    let lock = starting_path(root)?;
    let lock_file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    drop(lock_file);
    #[cfg(unix)]
    if endpoint.exists() {
        let _ = std::fs::remove_file(&endpoint);
    }
    let executable = std::env::current_exe().context("could not locate the Membrick executable")?;
    let log_path = resident_log_path()?;
    if std::fs::metadata(&log_path).is_ok_and(|metadata| metadata.len() > 1_048_576) {
        let previous = log_path.with_extension("log.1");
        let _ = std::fs::remove_file(&previous);
        std::fs::rename(&log_path, previous)?;
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    if let Err(error) = std::process::Command::new(executable)
        .arg("--resident-server")
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log))
        .spawn()
        .context("could not start the resident Membrick service")
    {
        let _ = std::fs::remove_file(lock);
        return Err(error);
    }
    Ok(())
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

pub(crate) async fn serve(router: Router, root: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::net::UnixListener;
        let endpoint = socket_path(root)?;
        if endpoint.exists() {
            let _ = std::fs::remove_file(&endpoint);
        }
        let listener = UnixListener::bind(&endpoint)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&endpoint, std::fs::Permissions::from_mode(0o600))?;
        }
        let _ = std::fs::remove_file(starting_path(root)?);
        eprintln!("Membrick resident IPC listening on {}", endpoint.display());
        loop {
            let (stream, _) = listener.accept().await?;
            serve_stream(router.clone(), stream);
        }
    }
    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ServerOptions;
        let endpoint = pipe_name(root)?;
        let _ = std::fs::remove_file(starting_path(root)?);
        eprintln!("Membrick resident IPC listening on {endpoint}");
        loop {
            let server = ServerOptions::new().create(&endpoint)?;
            server.connect().await?;
            serve_stream(router.clone(), server);
        }
    }
}

fn serve_stream<S>(router: Router, stream: S)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let service = hyper::service::service_fn(move |request: Request<Incoming>| {
            let router = router.clone();
            async move {
                let mut request = request.map(Body::new);
                request.extensions_mut().insert(LocalIpcRequest);
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
