#![cfg(feature = "roo-runtime")]

use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Write};
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
    fn start(root: &Workspace) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_lint-ai"))
            .arg("--roo-runtime-serve")
            .arg(&root.0)
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

fn hook(payload: Value) -> (Value, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lint-ai"))
        .args(["--roo-runtime-hook", "post-tool-use"])
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
    let mut mcp = McpProcess::start(&root);
    assert!(mcp.request(1, "initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"roo-test","version":"1"}})).get("result").is_some());
    // Open the store before the independent hook writes, exercising reader refresh.
    mcp.request(
        2,
        "tools/call",
        json!({"name":"search","arguments":{"query":"process-proof-299"}}),
    );
    let payload = json!({"workspaceRoot":root.0,"toolName":"fetch","toolUseId":"call-1","toolResponse":{"items":(0..300).map(|i| format!("item-{i}")).collect::<Vec<_>>(),"finding":"process-proof-299"}});
    let (context, stderr) = hook(payload.clone());
    assert!(
        context["additionalContext"]
            .as_str()
            .unwrap()
            .contains("process-proof-299"),
        "{stderr}"
    );
    hook(payload);
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
    let mut isolated = McpProcess::start(&other);
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
    );
    assert_eq!(response, json!({}));
    assert!(stderr.contains("failed open"));
}
