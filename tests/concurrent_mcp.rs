//! End-to-end characterization test for concurrent MCP startup.
//!
//! This exercises the real `membrick --codex-serve` subprocess boundary. The
//! project root is shared by all clients, so a startup/index-lock regression
//! should surface as an initialize or tools/list failure.

use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
fn short_temp_root() -> std::path::PathBuf {
    std::path::PathBuf::from("/tmp")
}

#[cfg(not(unix))]
fn short_temp_root() -> std::path::PathBuf {
    std::env::temp_dir()
}

fn send_request(stdin: &mut ChildStdin, request: &Value) {
    writeln!(stdin, "{}", request).expect("MCP request should be written");
    stdin.flush().expect("MCP request should be flushed");
}

fn read_response(reader: &mut impl BufRead) -> Value {
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("MCP response should be readable");
    assert!(!line.is_empty(), "MCP server closed before responding");
    serde_json::from_str(line.trim()).expect("MCP response should be valid JSON")
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[cfg(feature = "codex")]
fn concurrent_codex_mcp_clients_initialize_and_open_shared_index() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let root = std::env::current_dir()
        .expect("test directory should be available")
        .join("target")
        .join(format!("concurrent-mcp-{nonce}"));
    fs::create_dir_all(&root).expect("temporary project should be created");
    fs::write(root.join("README.md"), "shared MCP startup test\n")
        .expect("temporary project should contain a document");

    let executable = env!("CARGO_BIN_EXE_membrick");
    let mut clients: Vec<(Child, ChildStdin, BufReader<_>)> = (0..4)
        .map(|_| {
            let mut child = Command::new(executable)
                .args([
                    "--codex-serve",
                    root.to_str().expect("root should be UTF-8"),
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                // Preserve startup diagnostics: this test intentionally
                // exercises concurrent persistent-index initialization, where
                // a child can otherwise close stdout with no explanation.
                .stderr(Stdio::inherit())
                .spawn()
                .expect("Codex MCP server should start");
            let stdin = child
                .stdin
                .take()
                .expect("server stdin should be available");
            let stdout = child
                .stdout
                .take()
                .expect("server stdout should be available");
            (child, stdin, BufReader::new(stdout))
        })
        .collect();

    for (_, stdin, _) in &mut clients {
        send_request(
            stdin,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "concurrent-test", "version": "1"}
                }
            }),
        );
    }
    for (_, _, reader) in &mut clients {
        let response = read_response(reader);
        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["serverInfo"]["name"], "membrick");
    }

    for (_, stdin, _) in &mut clients {
        send_request(
            stdin,
            &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        );
    }
    for (_, _, reader) in &mut clients {
        let response = read_response(reader);
        assert_eq!(response["id"], 2);
        assert!(response["result"]["tools"].is_array());
    }

    for (mut child, _, _) in clients {
        let _ = child.kill();
        let status = child.wait().expect("MCP child should be reaped");
        assert!(status.success() || status.code().is_none());
    }
    fs::remove_dir_all(root).expect("temporary project should be removed");
}

#[test]
#[cfg(feature = "codex")]
fn concurrent_codex_mcp_clients_share_one_resident_workspace_owner() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let temp = short_temp_root().join(format!("mbr-{}", nonce % 1_000_000_000));
    let root = temp.join("project");
    fs::create_dir_all(&root).expect("temporary project should be created");
    // A regular file at HOME makes home-based runtime creation fail. The
    // resident must use the writable project runtime directory instead.
    let home = temp.join("unavailable-home");
    fs::write(&home, "not a home directory").expect("fake HOME should be unavailable");
    fs::write(root.join("README.md"), "shared resident owner test\n")
        .expect("temporary project should contain a document");

    let executable = env!("CARGO_BIN_EXE_membrick");
    let resident_log = root.join(".membrick").join("runtime").join("resident.log");
    fs::create_dir_all(resident_log.parent().expect("log parent should exist"))
        .expect("resident runtime directory should be created");
    let resident_log_file = File::create(&resident_log).expect("resident log should be created");
    let resident = Command::new(executable)
        .arg("--resident-server")
        .current_dir(&root)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(resident_log_file))
        .spawn()
        .expect("resident server should start");
    let _resident = ChildGuard(resident);

    let startup_deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if fs::read_to_string(&resident_log).is_ok_and(|log| log.contains("IPC listening")) {
            break;
        }
        assert!(
            std::time::Instant::now() < startup_deadline,
            "resident server did not become ready"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let mut clients: Vec<(ChildGuard, ChildStdin, BufReader<_>)> = (0..4)
        .map(|_| {
            let mut child = Command::new(executable)
                .args(["--codex-serve", root.to_str().expect("UTF-8 project path")])
                .env("HOME", &home)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("Codex MCP adapter should start");
            let stdin = child
                .stdin
                .take()
                .expect("adapter stdin should be available");
            let stdout = child
                .stdout
                .take()
                .expect("adapter stdout should be available");
            (ChildGuard(child), stdin, BufReader::new(stdout))
        })
        .collect();

    for (_, stdin, _) in &mut clients {
        send_request(
            stdin,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "resident-sharing-test", "version": "1"}
                }
            }),
        );
    }
    for (_, _, reader) in &mut clients {
        assert_eq!(
            read_response(reader)["result"]["serverInfo"]["name"],
            "membrick"
        );
    }

    for (_, stdin, _) in &mut clients {
        send_request(
            stdin,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "info", "arguments": {}}
            }),
        );
    }
    for (_, _, reader) in &mut clients {
        let response = read_response(reader);
        assert_eq!(response["id"], 2);
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("info should return text");
        let info: Value = serde_json::from_str(text).expect("info should return JSON");
        assert_eq!(info["server"]["transport"], "ipc");
        assert_eq!(info["server"]["status"], "connected");
    }

    let log = fs::read_to_string(&resident_log).expect("resident diagnostics should exist");
    assert_eq!(log.matches("IPC listening").count(), 1);
    drop(clients);
    drop(_resident);
    fs::remove_dir_all(temp).expect("temporary files should be removed");
}

#[test]
#[cfg(feature = "codex")]
fn concurrent_codex_lifecycle_hooks_record_all_events() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let root = std::env::current_dir()
        .expect("test directory should be available")
        .join("target")
        .join(format!("concurrent-hook-recording-{nonce}"));
    let session_root = root.join(".membrick").join("codex-sessions");
    fs::create_dir_all(&session_root).expect("recording directory should be created");
    fs::write(session_root.join("recording.json"), r#"{"enabled":true}"#)
        .expect("recording should be enabled");

    let executable = env!("CARGO_BIN_EXE_membrick");
    let mut children = Vec::new();
    for _ in 0..8 {
        let mut child = Command::new(executable)
            .args([
                "--codex-hook",
                "stop",
                root.to_str().expect("root should be UTF-8"),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Codex hook should start");
        let input = json!({
            "session_id": "shared-concurrent-session",
            "cwd": root,
            "hook_event_name": "Stop",
            "prompt": "concurrent lifecycle recording",
            "stop_hook_active": true
        });
        child
            .stdin
            .take()
            .expect("hook stdin should be available")
            .write_all(format!("{input}\n").as_bytes())
            .expect("hook input should be written");
        children.push(child);
    }

    for child in children {
        let output = child
            .wait_with_output()
            .expect("Codex hook should be reaped");
        assert!(output.status.success(), "hook failed: {:?}", output.status);
        serde_json::from_slice::<Value>(&output.stdout).expect("hook output should be JSON");
    }

    let events = fs::read_to_string(
        session_root
            .join("shared-concurrent-session")
            .join("events.jsonl"),
    )
    .expect("recorded events should exist");
    assert_eq!(events.lines().count(), 8);
    let sequences: Vec<u64> = events
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["sequence"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(sequences, (1..=8).collect::<Vec<_>>());
    fs::remove_dir_all(root).expect("temporary project should be removed");
}
