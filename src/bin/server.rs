use anyhow::Result;
use clap::Parser;
use lint_ai::memory_api::{AddRequest, MemoryService, SearchRequest};
use lint_ai::{IndexStore, PipelineOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Parser)]
#[command(
    name = "lint-ai-server",
    about = "Memory Add/Search server backed by Lint-AI"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long)]
    server_token: Option<String>,
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if args.server_token.is_none() {
        args.server_token = std::env::var("SERVER_TOKEN").ok();
    }
    let store = match args.index.as_deref() {
        Some(path) => IndexStore::at_path(path, PipelineOptions::default())?,
        None => IndexStore::in_memory(PipelineOptions::default()),
    };
    let service = Arc::new(Mutex::new(MemoryService::new(store)));
    let listener = TcpListener::bind(&args.bind)?;
    eprintln!("Lint-AI server listening on {}", args.bind);
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let service = Arc::clone(&service);
                let server_token = args.server_token.clone();
                std::thread::spawn(move || {
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

fn handle_connection(
    mut stream: TcpStream,
    service: Arc<Mutex<MemoryService>>,
    server_token: Option<&str>,
) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    let mut content_length = 0usize;
    let mut authorization = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => content_length = value.trim().parse()?,
                "authorization" | "x-api-key" => authorization = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }
    if method == "GET" && path == "/health" {
        return write_json(&mut stream, 200, &serde_json::json!({"status": "ok"}));
    }
    if let Some(expected) = server_token {
        let supplied = authorization.as_deref().unwrap_or("");
        let valid = supplied == expected
            || supplied == format!("Bearer {expected}")
            || supplied == format!("Token {expected}");
        if !valid {
            return write_json(
                &mut stream,
                401,
                &serde_json::json!({"detail": "unauthorized"}),
            );
        }
    }

    if method != "POST" || (path != "/add" && path != "/search") {
        return write_json(
            &mut stream,
            404,
            &serde_json::json!({"detail": "not found"}),
        );
    }
    if content_length > 16 * 1024 * 1024 {
        return write_json(
            &mut stream,
            413,
            &serde_json::json!({"detail": "request too large"}),
        );
    }
    let mut body = vec![0; content_length];
    reader.read_exact(&mut body)?;
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
        _ => unreachable!(),
    };
    write_json(&mut stream, 200, &response)
}

fn write_json(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        _ => "Error",
    };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len())?;
    stream.write_all(&bytes)?;
    Ok(())
}
