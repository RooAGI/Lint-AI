#![cfg(feature = "roo-runtime")]

use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "roo-process-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        Self(root.canonicalize().unwrap())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct McpProcess {
    child: Child,
    messages: Receiver<Value>,
}
impl McpProcess {
    fn start(root: &Workspace, server_url: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_membrick"))
            .arg("--roo-runtime-serve")
            .arg(&root.0)
            .env("MEMBRICK_SERVER_URL", server_url)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, messages) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else {
                    break;
                };
                let Ok(message) = serde_json::from_str(&line) else {
                    continue;
                };
                if tx.send(message).is_err() {
                    break;
                }
            }
        });
        Self { child, messages }
    }
    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        writeln!(
            self.child.stdin.as_mut().unwrap(),
            "{}",
            json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
        )
        .unwrap();
        self.child.stdin.as_mut().unwrap().flush().unwrap();
        loop {
            let message = self
                .messages
                .recv_timeout(Duration::from_secs(15))
                .expect("MCP response");
            if message["id"] == id {
                return message;
            }
        }
    }
}
impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct HttpServer {
    child: Child,
    url: String,
}
impl HttpServer {
    fn start(root: &Workspace) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let bind = address.to_string();
        let url = format!("http://{address}");
        let mut child = Command::new(env!("CARGO_BIN_EXE_server"))
            .args(["--bind", &bind, "--allow-unauthenticated", "--project-root"])
            .arg(&root.0)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(mut stream) = TcpStream::connect(address) {
                stream
                    .write_all(
                        b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                    )
                    .unwrap();
                let mut response = String::new();
                stream.read_to_string(&mut response).unwrap();
                assert!(
                    response.starts_with("HTTP/1.1 200"),
                    "server health check failed: {response}"
                );
                return Self { child, url };
            }
            if let Some(status) = child.try_wait().unwrap() {
                panic!("resident HTTP server exited before becoming ready: {status}");
            }
            assert!(
                Instant::now() < deadline,
                "resident HTTP server did not become ready"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}
impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn hook(payload: Value, server_url: &str) -> (Value, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_membrick"))
        .args(["--roo-runtime-hook", "post-tool-use"])
        .env("MEMBRICK_SERVER_URL", server_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({"schemaVersion":1,"event":"post_tool_use","runId":"capture-run","payload":payload})
    )
    .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    (
        serde_json::from_slice(&output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

#[test]
fn running_memory_server_observes_hook_writes_and_retries() {
    let root = Workspace::new();
    let other = Workspace::new();
    let http = HttpServer::start(&root);
    assert!(
        !root.0.join(".lint-ai").exists(),
        "legacy directory created by HTTP server startup"
    );
    let mut mcp = McpProcess::start(&root, &http.url);
    assert!(
        !root.0.join(".lint-ai").exists(),
        "legacy directory created by Roo MCP startup"
    );
    assert!(mcp.request(1, "initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"roo-test","version":"1"}})).get("result").is_some());
    // Open the store before the independent hook writes, exercising reader refresh.
    mcp.request(
        2,
        "tools/call",
        json!({"name":"search","arguments":{"query":"process-proof-299"}}),
    );
    assert!(
        !root.0.join(".lint-ai").exists(),
        "legacy directory created during initial MCP search: {:?}",
        fs::read_dir(&root.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>()
    );
    let payload = json!({"workspaceRoot":root.0,"toolName":"fetch","toolUseId":"call-1","toolResponse":{"items":(0..300).map(|i| format!("item-{i}")).collect::<Vec<_>>(),"finding":"process-proof-299"}});
    let (context, stderr) = hook(payload.clone(), &http.url);
    let additional_context = context["additionalContext"].as_str().unwrap_or_else(|| {
        panic!(
            "hook response omitted string additionalContext; stderr: {stderr}; stdout: {context}"
        )
    });
    assert!(
        additional_context.contains("process-proof-299"),
        "hook context did not contain the captured finding; stderr: {stderr}; stdout: {context}"
    );
    hook(payload, &http.url);
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut id = 3;
    loop {
        let response = mcp.request(
            id,
            "tools/call",
            json!({"name":"search","arguments":{"query":"process-proof-299"}}),
        );
        if response.to_string().contains("process-proof-299")
            && response.to_string().contains("roo-runtime://")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "live server failed to see capture: {response}"
        );
        id += 1;
        std::thread::sleep(Duration::from_millis(100));
    }
    let mut isolated = McpProcess::start(&other, &http.url);
    isolated.request(1, "initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"roo-test","version":"1"}}));
    let response = isolated.request(
        2,
        "tools/call",
        json!({"name":"search","arguments":{"query":"process-proof-299"}}),
    );
    assert!(!response.to_string().contains("roo-runtime://"));
}

#[test]
fn invalid_workspace_fails_open_without_claiming_capture() {
    let (response, stderr) = hook(
        json!({"workspaceRoot":"/nonexistent/roo-memory-test-root","toolName":"fetch","toolUseId":"call","toolResponse":{"finding":"evidence"}}),
        "http://127.0.0.1:1",
    );
    assert_eq!(response, json!({}));
    assert!(stderr.contains("failed open"));
}
