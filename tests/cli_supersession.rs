use serde_json::Value;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lint-ai-cli-supersession-{nanos}"));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn cli_query_and_llm_context_suppress_explicitly_superseded_markdown() {
    let root = temp_dir();
    fs::write(
        root.join("decision-a.md"),
        "# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.\n",
    )
    .unwrap();
    fs::write(
        root.join("decision-b.md"),
        "---\nsupersedes: decision-a.md\n---\n# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.\n",
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_lint-ai");
    let query = "How many retry attempts should we use for gateway timeouts?";

    let output = Command::new(bin)
        .current_dir(&root)
        .args(["--query", query, root.to_str().unwrap()])
        .output()
        .expect("CLI query should run");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("query should return JSON");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(
        results.len(),
        1,
        "superseded evidence must be filtered: {payload:#}"
    );
    assert_eq!(results[0]["doc_id"], "decision-b.md");
    assert_eq!(results[0]["semantic_status"], "current");
    assert!(
        payload["aggregation"].is_null(),
        "recommended quantity must not count evidence rows: {payload:#}"
    );

    let output = Command::new(bin)
        .current_dir(&root)
        .args([
            "--llm-context",
            query,
            "--result-count",
            "5",
            root.to_str().unwrap(),
        ])
        .output()
        .expect("LLM context query should run");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value =
        serde_json::from_slice(&output.stdout).expect("LLM context should return JSON");
    let chunks = payload["top_chunks"].as_array().expect("top_chunks array");
    assert!(!chunks.is_empty());
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["doc_id"] != "decision-a.md"),
        "superseded evidence leaked into LLM context: {payload:#}"
    );
    assert!(chunks.iter().any(|chunk| {
        chunk["doc_id"] == "decision-b.md"
            && chunk["text"]
                .as_str()
                .unwrap_or("")
                .contains("retry attempts: 2")
    }));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn cli_uses_file_timeline_to_supersede_scalar_configuration_without_metadata() {
    let root = temp_dir();
    let old_path = root.join("decision-a.md");
    let new_path = root.join("decision-b.md");
    fs::write(
        &old_path,
        "# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.\n",
    )
    .unwrap();
    fs::write(
        &new_path,
        "# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.\n",
    )
    .unwrap();

    let old_time = UNIX_EPOCH + std::time::Duration::from_secs(1_735_689_600);
    let new_time = UNIX_EPOCH + std::time::Duration::from_secs(1_767_225_600);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&old_path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old_time))
        .unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&new_path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(new_time))
        .unwrap();

    let bin = env!("CARGO_BIN_EXE_lint-ai");
    let query = "How many retry attempts should we use for gateway timeouts?";
    let output = Command::new(bin)
        .current_dir(&root)
        .args(["--query", query, root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let results = payload["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        1,
        "older timeline value must be suppressed: {payload:#}"
    );
    assert_eq!(results[0]["doc_id"], "decision-b.md");
    assert_eq!(results[0]["semantic_status"], "current");
    assert!(payload["aggregation"].is_null());

    let output = Command::new(bin)
        .current_dir(&root)
        .args([
            "--llm-context",
            query,
            "--result-count",
            "5",
            root.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let chunks = payload["top_chunks"].as_array().unwrap();
    assert!(chunks
        .iter()
        .all(|chunk| chunk["doc_id"] != "decision-a.md"));
    assert!(chunks.iter().any(|chunk| {
        chunk["doc_id"] == "decision-b.md"
            && chunk["text"]
                .as_str()
                .unwrap_or("")
                .contains("retry attempts: 2")
    }));
    let _ = fs::remove_dir_all(root);
}
