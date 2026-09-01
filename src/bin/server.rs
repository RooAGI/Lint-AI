use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::memory_api::{
    AddRequest, DeleteRequest, MemoryService, SearchRequest, SupersedeRequest,
};
use lint_ai::{IndexStore, PipelineOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Largest request body accepted, after which the server answers 413.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Largest single request or header line accepted.
const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
/// Total budget for the request line plus all headers.
const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Upper bound on header lines, so a client cannot stream headers forever.
const MAX_HEADER_COUNT: usize = 100;
/// Socket timeouts, so a stalled client cannot hold a worker thread open.
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Connections handled concurrently; further connections are refused with 503.
const MAX_CONCURRENT_CONNECTIONS: usize = 128;

#[derive(Debug, Parser)]
#[command(
    name = "lint-ai-server",
    about = "Memory Add/Search server backed by Lint-AI",
    version = env!("CARGO_PKG_VERSION")
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long)]
    server_token: Option<String>,
    /// Serve without a token on a non-loopback address. Only for closed networks.
    #[arg(long)]
    allow_unauthenticated: bool,
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if args.server_token.is_none() {
        args.server_token = std::env::var("SERVER_TOKEN")
            .ok()
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty());
    }
    if args.server_token.is_none()
        && !args.allow_unauthenticated
        && binds_beyond_loopback(&args.bind)?
    {
        anyhow::bail!(
            "refusing to serve {} without a token: pass --server-token, set SERVER_TOKEN, \
             bind to a loopback address, or pass --allow-unauthenticated to override",
            args.bind
        );
    }
    if args.server_token.is_none() {
        eprintln!("Warning: no server token configured; every request is accepted.");
    }
    let store = match args.index.as_deref() {
        Some(path) => IndexStore::at_path(path, PipelineOptions::default())?,
        None => IndexStore::in_memory(PipelineOptions::default()),
    };
    let service = Arc::new(Mutex::new(MemoryService::new(store)));
    let listener = TcpListener::bind(&args.bind)?;
    let active = Arc::new(AtomicUsize::new(0));
    eprintln!("Lint-AI server listening on {}", args.bind);
    for connection in listener.incoming() {
        match connection {
            Ok(mut stream) => {
                if active.load(Ordering::Acquire) >= MAX_CONCURRENT_CONNECTIONS {
                    let _ = write_json(
                        &mut stream,
                        503,
                        &serde_json::json!({"detail": "server busy"}),
                    );
                    continue;
                }
                let service = Arc::clone(&service);
                let server_token = args.server_token.clone();
                let active = Arc::clone(&active);
                active.fetch_add(1, Ordering::AcqRel);
                std::thread::spawn(move || {
                    let _guard = ConnectionGuard(active);
                    if let Err(error) = handle_connection(stream, service, server_token.as_deref())
                    {
                        eprintln!("Request failed: {error:#}");
                    }
                });
            }
            Err(error) => eprintln!("Accept failed: {error}"),
        }
    }
    Ok(())
}

/// Decrements the live-connection counter even if the handler panics.
struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// True when `bind` resolves to any address reachable from outside this host.
fn binds_beyond_loopback(bind: &str) -> Result<bool> {
    let addresses = bind
        .to_socket_addrs()
        .with_context(|| format!("invalid bind address: {bind}"))?;
    let mut resolved = false;
    for address in addresses {
        resolved = true;
        if !address.ip().is_loopback() {
            return Ok(true);
        }
    }
    if !resolved {
        anyhow::bail!("bind address resolved to no addresses: {bind}");
    }
    Ok(false)
}

/// Compares two secrets without leaking their contents through timing.
fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut difference = (left.len() ^ right.len()) as u8;
    for index in 0..left.len().max(right.len()) {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= a ^ b;
    }
    difference == 0
}

fn token_is_valid(supplied: &str, expected: &str) -> bool {
    // Bitwise `|` so every candidate is compared regardless of earlier matches.
    constant_time_eq(supplied, expected)
        | constant_time_eq(supplied, &format!("Bearer {expected}"))
        | constant_time_eq(supplied, &format!("Token {expected}"))
}

/// Reads one CRLF-terminated line, charging it against the shared header budget.
fn read_limited_line(reader: &mut impl BufRead, budget: &mut usize) -> Result<String> {
    let limit = (*budget).min(MAX_HEADER_LINE_BYTES);
    let mut line = String::new();
    let read = reader.take(limit as u64).read_line(&mut line)?;
    *budget -= read;
    if read == 0 {
        anyhow::bail!("client closed the connection before the request was complete");
    }
    if !line.ends_with('\n') {
        anyhow::bail!("request line or header exceeds {MAX_HEADER_LINE_BYTES} bytes");
    }
    Ok(line)
}

fn handle_connection(
    mut stream: TcpStream,
    service: Arc<Mutex<MemoryService>>,
    server_token: Option<&str>,
) -> Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut header_budget = MAX_HEADER_BYTES;
    let request_line = read_limited_line(&mut reader, &mut header_budget)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    let mut content_length = 0usize;
    let mut authorization = None;
    let mut header_count = 0usize;
    loop {
        let line = read_limited_line(&mut reader, &mut header_budget)?;
        if line == "\r\n" || line == "\n" {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            return write_json(
                &mut stream,
                431,
                &serde_json::json!({"detail": "too many headers"}),
            );
        }
        if let Some((name, value)) = line.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => {
                    content_length = match value.trim().parse() {
                        Ok(length) => length,
                        Err(_) => {
                            return write_json(
                                &mut stream,
                                400,
                                &serde_json::json!({"detail": "invalid content-length"}),
                            )
                        }
                    }
                }
                "authorization" | "x-api-key" => authorization = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }
    if method == "GET" && path == "/health" {
        return write_json(
            &mut stream,
            200,
            &serde_json::json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}),
        );
    }
    if let Some(expected) = server_token {
        let supplied = authorization.as_deref().unwrap_or("");
        if !token_is_valid(supplied, expected) {
            return write_json(
                &mut stream,
                401,
                &serde_json::json!({"detail": "unauthorized"}),
            );
        }
    }

    if method != "POST"
        || !matches!(
            path,
            "/add" | "/search" | "/delete" | "/supersede" | "/expire"
        )
    {
        return write_json(
            &mut stream,
            404,
            &serde_json::json!({"detail": "not found"}),
        );
    }
    if content_length > MAX_BODY_BYTES {
        return write_json(
            &mut stream,
            413,
            &serde_json::json!({"detail": "request too large"}),
        );
    }
    // Grow into the body rather than trusting Content-Length with an allocation.
    let mut body = Vec::new();
    let read = reader
        .take(content_length as u64)
        .read_to_end(&mut body)
        .context("failed to read request body")?;
    if read != content_length {
        anyhow::bail!("request body ended after {read} of {content_length} bytes");
    }
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return write_json(
                &mut stream,
                422,
                &serde_json::json!({"detail": error.to_string()}),
            )
        }
    };
    let mut service = service
        .lock()
        .map_err(|_| anyhow::anyhow!("service lock poisoned"))?;
    let response = match path {
        "/add" => match serde_json::from_value::<AddRequest>(value)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .and_then(|request| service.add(request))
        {
            Ok(response) => serde_json::to_value(response)?,
            Err(error) => {
                return write_json(
                    &mut stream,
                    422,
                    &serde_json::json!({"detail": error.to_string()}),
                )
            }
        },
        "/search" => match serde_json::from_value::<SearchRequest>(value)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .and_then(|request| service.search(request))
        {
            Ok(response) => serde_json::to_value(response)?,
            Err(error) => {
                return write_json(
                    &mut stream,
                    422,
                    &serde_json::json!({"detail": error.to_string()}),
                )
            }
        },
        "/delete" => match serde_json::from_value::<DeleteRequest>(value)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .and_then(|request| service.delete(&request.user_id, &request.doc_id))
        {
            Ok(affected) => serde_json::json!({"success": true, "affected": affected as usize}),
            Err(error) => {
                return write_json(
                    &mut stream,
                    422,
                    &serde_json::json!({"detail": error.to_string()}),
                )
            }
        },
        "/supersede" => match serde_json::from_value::<SupersedeRequest>(value)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .and_then(|request| {
                service.supersede(&request.user_id, &request.replacement_id, &request.old_id)
            }) {
            Ok(affected) => serde_json::json!({"success": true, "affected": affected as usize}),
            Err(error) => {
                return write_json(
                    &mut stream,
                    422,
                    &serde_json::json!({"detail": error.to_string()}),
                )
            }
        },
        "/expire" => {
            let user_id = value
                .get("user_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match service.expire(user_id) {
                Ok(affected) => serde_json::json!({"success": true, "affected": affected}),
                Err(error) => {
                    return write_json(
                        &mut stream,
                        422,
                        &serde_json::json!({"detail": error.to_string()}),
                    )
                }
            }
        }
        _ => unreachable!(),
    };
    write_json(&mut stream, 200, &response)
}

fn write_json(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Error",
    };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len())?;
    stream.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_binds_do_not_require_a_token() {
        assert!(!binds_beyond_loopback("127.0.0.1:8080").unwrap());
        assert!(binds_beyond_loopback("0.0.0.0:8080").unwrap());
    }

    #[test]
    fn token_accepts_the_documented_schemes() {
        assert!(token_is_valid("secret", "secret"));
        assert!(token_is_valid("Bearer secret", "secret"));
        assert!(token_is_valid("Token secret", "secret"));
        assert!(!token_is_valid("", "secret"));
        assert!(!token_is_valid("secre", "secret"));
        assert!(!token_is_valid("Bearer other", "secret"));
    }

    #[test]
    fn header_lines_are_bounded() {
        let mut budget = MAX_HEADER_BYTES;
        let oversized = format!("X-Long: {}\r\n", "a".repeat(MAX_HEADER_LINE_BYTES));
        let error = read_limited_line(&mut oversized.as_bytes(), &mut budget).unwrap_err();
        assert!(error.to_string().contains("exceeds"));
    }
}
