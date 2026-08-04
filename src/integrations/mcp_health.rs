use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;

pub fn verify(root: &Path, serve_flag: &str, timeout_ms: Option<u64>) -> Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let timeout_ms = timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
    let started = Instant::now();
    let mut child =
        Command::new(env::current_exe().context("failed to locate lint-ai executable")?)
            .arg(serve_flag)
            .arg(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to launch MCP server with {serve_flag}"))?;

    let mut stdin = child.stdin.take().context("MCP server stdin unavailable")?;
    let stdout = child
        .stdout
        .take()
        .context("MCP server stdout unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("MCP server stderr unavailable")?;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            match read_frame(&mut reader) {
                Ok(None) => break,
                Ok(Some(body)) => {
                    let result =
                        serde_json::from_slice::<Value>(&body).map_err(std::io::Error::other);
                    if tx.send(result).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                    break;
                }
            }
        }
    });
    let (stderr_tx, stderr_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut stderr = stderr;
        let mut output = String::new();
        let _ = stderr.read_to_string(&mut output);
        let _ = stderr_tx.send(output);
    });

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "lint-ai-health-check", "version": env!("CARGO_PKG_VERSION")}
        }
    });
    let request_started = Instant::now();
    write_frame(&mut stdin, &request)?;
    stdin.flush()?;

    let response = match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(response) => match response {
            Ok(response) => response,
            Err(error) => {
                terminate_child(&mut child);
                anyhow::bail!("MCP server returned invalid JSON: {error}")
            }
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            terminate_child(&mut child);
            let stderr = stderr_rx.recv_timeout(Duration::from_secs(1)).ok();
            let report = report(
                "failed",
                started,
                request_started,
                None,
                Some("initialize timed out"),
                stderr.as_deref(),
                None,
                None,
            );
            write_report(&report)?;
            anyhow::bail!("MCP initialize timed out after {timeout_ms} ms")
        }
        Err(error) => {
            terminate_child(&mut child);
            anyhow::bail!("MCP server exited before initialize response: {error}")
        }
    };

    let initialized = response.get("result").is_some()
        && response.get("id") == Some(&json!(1))
        && response["result"]["serverInfo"]["name"] == "lint-ai";
    if !initialized {
        terminate_child(&mut child);
        let stderr = stderr_rx.recv_timeout(Duration::from_secs(1)).ok();
        let report = report(
            "failed",
            started,
            request_started,
            Some(&response),
            Some("invalid initialize response"),
            stderr.as_deref(),
            None,
            None,
        );
        write_report(&report)?;
        anyhow::bail!("MCP server returned an invalid initialize response")
    }

    let tools_started = Instant::now();
    let tools_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    write_frame(&mut stdin, &tools_request)?;
    stdin.flush()?;
    let tools_response = match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(response) => response,
        Err(error) => {
            terminate_child(&mut child);
            anyhow::bail!("MCP server did not respond to tools/list: {error}")
        }
    };
    let tools_response = match tools_response {
        Ok(response) => response,
        Err(error) => {
            terminate_child(&mut child);
            anyhow::bail!("MCP server returned invalid tools/list JSON: {error}")
        }
    };
    let tools_valid =
        tools_response.get("id") == Some(&json!(2)) && tools_response["result"]["tools"].is_array();
    if !tools_valid {
        terminate_child(&mut child);
        anyhow::bail!("MCP server returned an invalid tools/list response")
    }
    let tools_count = tools_response["result"]["tools"]
        .as_array()
        .map(Vec::len)
        .unwrap_or_default();

    terminate_child(&mut child);
    let stderr = stderr_rx.recv_timeout(Duration::from_secs(1)).ok();
    let report = report(
        "healthy",
        started,
        request_started,
        Some(&response),
        None,
        stderr.as_deref(),
        Some(tools_started.elapsed().as_secs_f64() * 1000.0),
        Some(tools_count),
    );
    write_report(&report)?;
    Ok(())
}

fn terminate_child(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn write_frame(writer: &mut impl Write, request: &Value) -> Result<()> {
    let body = serde_json::to_vec(request)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    Ok(())
}

fn read_frame(reader: &mut impl BufRead) -> std::io::Result<Option<Vec<u8>>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                rest.trim()
                    .parse::<usize>()
                    .map_err(std::io::Error::other)?,
            );
        }
    }
    let length = content_length.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing Content-Length")
    })?;
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(body))
}

fn report(
    status: &str,
    started: Instant,
    request_started: Instant,
    response: Option<&Value>,
    error: Option<&str>,
    stderr: Option<&str>,
    tools_list_ms: Option<f64>,
    tools_count: Option<usize>,
) -> Value {
    json!({
        "status": status,
        "startup_ms": started.elapsed().as_secs_f64() * 1000.0,
        "initialize_ms": request_started.elapsed().as_secs_f64() * 1000.0,
        "protocol_version": response.and_then(|value| value["result"]["protocolVersion"].as_str()),
        "server_info": response.and_then(|value| value["result"]["serverInfo"].clone().as_object().cloned()),
        "error": error,
        "stderr": stderr.filter(|value| !value.trim().is_empty()),
        "tools_list_ms": tools_list_ms,
        "tools_count": tools_count,
    })
}

fn write_report(report: &Value) -> Result<()> {
    let output = serde_json::to_string_pretty(report)?;
    println!("{output}");
    if let Some(path) = env::var_os("LINT_AI_MCP_HEALTH_PATH") {
        fs::write(path, format!("{output}\n"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_contains_protocol_and_timings() {
        let value = report(
            "healthy",
            Instant::now(),
            Instant::now(),
            Some(&json!({
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "lint-ai"}
                }
            })),
            None,
            None,
            None,
            None,
        );
        assert_eq!(value["status"], "healthy");
        assert_eq!(value["protocol_version"], "2024-11-05");
        assert!(value["startup_ms"].is_number());
    }
}
