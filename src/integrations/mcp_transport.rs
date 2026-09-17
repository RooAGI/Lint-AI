use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, Write};

pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_HEADER_COUNT: usize = 100;

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

pub fn read_request(reader: &mut impl BufRead) -> Result<Option<(JsonRpcRequest, bool)>> {
    let mut content_length = None;
    let mut header_bytes = 0usize;
    let mut header_count = 0usize;
    loop {
        let mut line = String::new();
        let read = std::io::Read::take(
            std::io::Read::by_ref(reader),
            (MAX_REQUEST_BYTES + 1) as u64,
        )
        .read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        if read > MAX_REQUEST_BYTES {
            anyhow::bail!("MCP line or request exceeds {MAX_REQUEST_BYTES} byte limit");
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.starts_with('{') {
            return Ok(Some((serde_json::from_str(trimmed)?, true)));
        }
        header_bytes += read;
        if header_bytes > MAX_HEADER_BYTES {
            anyhow::bail!("MCP headers exceed {MAX_HEADER_BYTES} bytes");
        }
        if trimmed.is_empty() {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            anyhow::bail!("too many MCP headers");
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            if content_length.is_some() {
                anyhow::bail!("duplicate Content-Length header");
            }
            content_length = Some(
                rest.trim()
                    .parse::<usize>()
                    .context("invalid Content-Length header")?,
            );
        }
    }

    let len = content_length.context("missing Content-Length header")?;
    if len > MAX_REQUEST_BYTES {
        anyhow::bail!("MCP request body exceeds {MAX_REQUEST_BYTES} byte limit: {len} bytes");
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let request: JsonRpcRequest = serde_json::from_slice(&body)?;
    Ok(Some((request, false)))
}

pub fn write_response(
    writer: &mut impl Write,
    response: &JsonRpcResponse,
    line_framed: bool,
) -> Result<()> {
    let body = serde_json::to_vec(response)?;
    if line_framed {
        writer.write_all(&body)?;
        writer.write_all(b"\n")?;
    } else {
        write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
        writer.write_all(&body)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn reads_line_framed_requests() {
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let (_, line_framed) = read_request(&mut Cursor::new(input)).unwrap().unwrap();
        assert!(line_framed);
    }

    #[test]
    fn reads_content_length_requests() {
        let body = br#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let input = format!(
            "Content-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        let (_, line_framed) = read_request(&mut Cursor::new(input)).unwrap().unwrap();
        assert!(!line_framed);
    }

    #[test]
    fn rejects_oversized_requests() {
        let input = format!("Content-Length: {}\r\n\r\n", MAX_REQUEST_BYTES + 1);
        let error = read_request(&mut Cursor::new(input)).unwrap_err();
        assert!(error.to_string().contains("exceeds"));
    }

    #[test]
    fn rejects_duplicate_content_length() {
        let input = b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}";
        let error = read_request(&mut Cursor::new(input)).unwrap_err();
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn rejects_oversized_header_budget() {
        let input = format!("X: {}\r\n\r\n", "x".repeat(70_000));
        let error = read_request(&mut Cursor::new(input.as_bytes())).unwrap_err();
        assert!(error.to_string().contains("headers") || error.to_string().contains("request"));
    }

    #[test]
    fn line_framed_request_may_exceed_header_budget() {
        let padding = "x".repeat(MAX_HEADER_BYTES);
        let input = format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"padding":"{padding}"}}}}"#
        );
        let (request, line_framed) = read_request(&mut Cursor::new(input.as_bytes()))
            .unwrap()
            .unwrap();
        assert!(line_framed);
        assert_eq!(request.method, "initialize");
    }
}
