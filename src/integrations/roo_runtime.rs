//! Roo Runtime process hook adapter. All durable writes use MemoryService's
//! shared workspace memory store, so MCP and hook callers see one index.
use crate::cli::RooRuntimeHook;
use crate::integrations::session_recording::{lint_ai_enabled, RecordingProvider};
use crate::memory_api::{MemoryService, SearchRequest};
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_CONTEXT_BYTES: usize = 8_000;
const MAX_CAPTURE_FIELD_BYTES: usize = 8_000;
const MAX_TOOL_CAPTURE_BYTES: usize = 6 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HookInvocation {
    #[serde(default)]
    schema_version: u8,
    event: String,
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    payload: Value,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct HookResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    additional_context: Option<String>,
}

pub fn run_hook(kind: RooRuntimeHook) -> Result<()> {
    let result = (|| -> Result<HookResult> {
        let mut invocation: HookInvocation = crate::integrations::read_bounded_json()
            .context("failed to parse Roo Runtime hook input")?;
        let expected = event_name(kind);
        if invocation.schema_version != 1 {
            anyhow::bail!(
                "unsupported Roo Runtime hook schema version {}",
                invocation.schema_version
            );
        }
        if invocation.event != expected {
            anyhow::bail!(
                "Roo Runtime hook event mismatch: expected {expected}, got {}",
                invocation.event
            );
        }
        let root_value = invocation
            .payload
            .get("workspaceRoot")
            .and_then(Value::as_str)
            .map(PathBuf::from);
        let root = match (kind, root_value) {
            (RooRuntimeHook::PreToolUse | RooRuntimeHook::PostToolUse, Some(root)) => root,
            (RooRuntimeHook::PreToolUse | RooRuntimeHook::PostToolUse, None) => {
                anyhow::bail!("Roo tool hook payload is missing workspaceRoot")
            }
            (_, Some(root)) => root,
            (RooRuntimeHook::RunStart | RooRuntimeHook::RunEnd, None) => {
                return Ok(HookResult::default())
            }
            (_, None) => anyhow::bail!("Roo memory hook payload is missing workspaceRoot"),
        };
        let root = root
            .canonicalize()
            .with_context(|| format!("cannot resolve Roo workspace {}", root.display()))?;
        // The host-managed memory search tool returns existing evidence;
        // indexing that view again would amplify retrieval copies as new facts.
        if matches!(
            kind,
            RooRuntimeHook::PreToolUse | RooRuntimeHook::PostToolUse
        ) && invocation.payload.get("toolName").and_then(Value::as_str)
            == Some("workspace_memory_search")
        {
            return Ok(HookResult::default());
        }
        load_payload_references(&root, &mut invocation.payload)?;
        if !lint_ai_enabled(RecordingProvider::RooRuntime, &root)? {
            return Ok(HookResult::default());
        }
        match kind {
            RooRuntimeHook::AgentTurnStart => recall(&root, &invocation),
            RooRuntimeHook::AgentTurnEnd => {
                capture(&root, &invocation, "turn")?;
                Ok(HookResult::default())
            }
            RooRuntimeHook::PreToolUse => {
                capture_tool_event(&root, &invocation, "tool_call")?;
                Ok(HookResult::default())
            }
            RooRuntimeHook::PostToolUse => {
                capture_tool_event(&root, &invocation, "tool_result")?;
                Ok(tool_result_context(&invocation))
            }
            RooRuntimeHook::RunEnd => {
                // The current RunEnd projection may omit workspaceRoot. Do
                // not guess from the hook process cwd and write into the
                // wrong project's shared memory store.
                if invocation
                    .payload
                    .get("workspaceRoot")
                    .and_then(Value::as_str)
                    .is_some_and(|path| !path.trim().is_empty())
                {
                    capture(&root, &invocation, "run")?;
                }
                Ok(HookResult::default())
            }
            RooRuntimeHook::RunStart => Ok(HookResult::default()),
        }
    })();
    let result = result.unwrap_or_else(|error| {
        eprintln!("warning: Roo Runtime Lint-AI hook failed open: {error:#}");
        HookResult::default()
    });
    serde_json::to_writer(std::io::stdout().lock(), &result)?;
    Ok(())
}

fn load_payload_references(root: &Path, payload: &mut Value) -> Result<()> {
    let Some(references) = payload
        .get("payloadReferences")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(());
    };
    let allowed = root.join(".roo").join("hook-payloads");
    let canonical_allowed = allowed.canonicalize()?;
    anyhow::ensure!(
        canonical_allowed == allowed,
        "hook payload directory must not be a symlink"
    );
    for (field, reference) in references {
        anyhow::ensure!(
            matches!(field.as_str(), "toolInput" | "toolResponse" | "toolError"),
            "unsupported payload reference field"
        );
        let path = reference
            .get("path")
            .and_then(Value::as_str)
            .context("payload reference path is missing")?;
        let canonical = Path::new(path).canonicalize()?;
        anyhow::ensure!(
            canonical.starts_with(&canonical_allowed) && canonical == Path::new(path),
            "payload reference is outside the host-owned directory"
        );
        let expected = reference
            .get("sizeBytes")
            .and_then(Value::as_u64)
            .context("payload reference size is missing")?;
        anyhow::ensure!(
            expected <= 128 * 1024 * 1024
                && reference.get("encoding").and_then(Value::as_str) == Some("json"),
            "unsupported payload reference"
        );
        anyhow::ensure!(
            fs::symlink_metadata(&canonical)?.is_file(),
            "payload reference must be a regular file"
        );
        let file = crate::pipeline::file_access::open_regular_file(&canonical)?;
        anyhow::ensure!(
            file.metadata()?.len() == expected,
            "payload reference size mismatch"
        );
        let mut bytes = Vec::new();
        use std::io::Read;
        file.take(expected + 1).read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 == expected,
            "payload reference changed during read"
        );
        let expected_hash = reference
            .get("sha256")
            .and_then(Value::as_str)
            .context("payload reference checksum is missing")?;
        use sha2::Digest;
        anyhow::ensure!(
            format!("{:x}", sha2::Sha256::digest(&bytes)) == expected_hash,
            "payload reference checksum mismatch"
        );
        payload[&field] = serde_json::from_slice(&bytes)?;
    }
    Ok(())
}

fn recall(root: &Path, input: &HookInvocation) -> Result<HookResult> {
    let query = bounded(
        input
            .payload
            .get("userMessage")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        4_000,
    );
    if query.trim().is_empty() {
        return Ok(HookResult::default());
    }
    let hits = MemoryService::with_shared_memory(root, |memory| {
        memory.search(SearchRequest {
            query: query.clone(),
            options: None,
            user_id: String::new(),
            top_k: 5,
            session_id: None,
            scope: None,
            filters: None,
        })
    })?;
    if hits.data.is_empty() {
        return Ok(HookResult::default());
    }
    let mut context = String::from("Relevant project memory from Lint-AI:\n");
    for hit in hits.data {
        if context.len() >= MAX_CONTEXT_BYTES {
            break;
        }
        let remaining = MAX_CONTEXT_BYTES.saturating_sub(context.len());
        context.push_str(&relevant_excerpt(
            &hit.content,
            &query,
            remaining.min(2_000),
        ));
        context.push_str("\n\n");
    }
    Ok(recall_context_result(&context))
}

fn recall_context_result(context: &str) -> HookResult {
    let context = bounded(context, MAX_CONTEXT_BYTES);
    if context.len() > 40 {
        HookResult {
            additional_context: Some(context.trim_end().to_string()),
        }
    } else {
        HookResult::default()
    }
}

fn relevant_excerpt(content: &str, query: &str, max_bytes: usize) -> String {
    // ASCII case folding preserves byte offsets, including for UTF-8 content.
    let folded = content.to_ascii_lowercase();
    let mut terms = query
        .split_whitespace()
        .filter(|term| term.len() >= 3)
        .collect::<Vec<_>>();
    terms.sort_by_key(|term| std::cmp::Reverse(term.len()));
    let matched = terms
        .iter()
        .find_map(|term| folded.find(&term.to_ascii_lowercase()));
    let mut start = matched.unwrap_or(0).saturating_sub((max_bytes / 4).min(64));
    while !content.is_char_boundary(start) {
        start -= 1;
    }
    bounded(&content[start..], max_bytes)
}

pub(crate) fn search_results_with_evidence(
    service: &MemoryService,
    query: &str,
    results: Vec<crate::SearchResult>,
) -> Value {
    Value::Array(
        results
            .into_iter()
            .map(|hit| {
                let document = service.source_document_by_id(&hit.doc_id);
                let mut result = serde_json::to_value(&hit).unwrap_or_default();
                if let Some(document) = document {
                    result["content"] =
                        Value::String(relevant_excerpt(&document.content, query, 2_000));
                    result["contentCoverage"] = Value::String("query_excerpt".into());
                    result["provenance"] =
                        serde_json::to_value(&document.filters).unwrap_or_default();
                }
                result
            })
            .collect(),
    )
}

fn memory_session_id(project: &str, input: &HookInvocation) -> String {
    let session = input
        .payload
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .unwrap_or(&input.run_id);
    format!("rooagi_runtime:{project}:{session}")
}

fn capture(root: &Path, input: &HookInvocation, kind: &str) -> Result<()> {
    let content = if kind == "turn" {
        let user = bounded(
            input
                .payload
                .get("userMessage")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            MAX_CAPTURE_FIELD_BYTES,
        );
        let assistant = bounded(
            input
                .payload
                .get("assistantResponse")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            MAX_CAPTURE_FIELD_BYTES,
        );
        if user.trim().is_empty() && assistant.trim().is_empty() {
            return Ok(());
        }
        format!(
            "Roo Runtime completed turn\n\nUser: {user}\n\nAssistant: {assistant}\n\nStatus: {}",
            input
                .payload
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        )
    } else {
        let status = input
            .payload
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        format!(
            "Roo Runtime run {} ended with status: {status}",
            input.run_id
        )
    };
    let project = root.to_string_lossy().into_owned();
    let event_id = if kind == "turn" {
        input
            .payload
            .get("turnId")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(&input.run_id)
    } else {
        &input.run_id
    };
    let source = format!(
        "roo-runtime://{}/{}-{kind}",
        crate::ids::stable_doc_id_from_source(&project),
        event_id
    );
    let mut document = SourceDocument::with_stable_doc_id_from_source(
        source,
        content,
        kind.to_string(),
        Some(memory_session_id(&project, input)),
        vec![],
        vec![],
        input.timestamp.clone(),
        Some("rooagi_runtime".to_string()),
    );
    document.filters = BTreeMap::from([
        ("provider".to_string(), "rooagi_runtime".to_string()),
        ("project_root".to_string(), project),
        ("run_id".to_string(), input.run_id.clone()),
        ("capture_type".to_string(), kind.to_string()),
    ]);
    if let Some(session_id) = input
        .payload
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
    {
        document
            .filters
            .insert("session_id".into(), session_id.to_owned());
    }
    if let Some(id) = input.payload.get("turnId").and_then(Value::as_str) {
        document
            .filters
            .insert("turn_id".to_string(), bounded(id, 256));
    }
    MemoryService::with_shared_memory(root, |memory| {
        memory.upsert(document);
        memory.refresh_index()
    })
}

fn capture_tool_event(root: &Path, input: &HookInvocation, kind: &str) -> Result<()> {
    let tool_name = input
        .payload
        .get("toolName")
        .and_then(Value::as_str)
        .map(|name| bounded(name, 256))
        .filter(|name| !name.trim().is_empty())
        .context("Roo tool hook payload is missing toolName")?;
    let tool_use_id = input
        .payload
        .get("toolUseId")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .context("Roo tool hook payload is missing toolUseId")?;
    anyhow::ensure!(
        !input.run_id.trim().is_empty(),
        "Roo tool hook payload is missing runId"
    );
    let tool_input = input
        .payload
        .get("toolInput")
        .map(|value| bounded(&sanitize_tool_value(value, 0), MAX_TOOL_CAPTURE_BYTES))
        .unwrap_or_else(|| "null".to_string());
    let result = input
        .payload
        .get("toolResponse")
        .or_else(|| input.payload.get("toolError"))
        .map(|value| sanitize_tool_value(value, 0));
    if kind == "tool_result" && result.is_none() {
        return Ok(());
    }

    let content = if let Some(result) = result {
        format!("Roo Runtime tool result\nTool: {tool_name}\nInput: {tool_input}\nResult: {result}")
    } else {
        format!("Roo Runtime tool call\nTool: {tool_name}\nInput: {tool_input}")
    };
    let project = root.to_string_lossy().into_owned();
    let project_id = crate::ids::stable_doc_id_from_source(&project);
    let source = format!(
        "roo-runtime://{project_id}/{}/{}-{kind}",
        input.run_id, tool_use_id
    );
    let mut document = SourceDocument::with_stable_doc_id_from_source(
        source,
        content,
        format!("{kind}: {tool_name}"),
        Some(memory_session_id(&project, input)),
        vec![tool_name.clone()],
        vec![],
        input.timestamp.clone(),
        Some("rooagi_runtime".to_string()),
    );
    document.filters = BTreeMap::from([
        ("provider".to_string(), "rooagi_runtime".to_string()),
        ("project_root".to_string(), project),
        ("run_id".to_string(), input.run_id.clone()),
        ("capture_type".to_string(), kind.to_string()),
        ("tool_name".to_string(), tool_name),
        ("tool_use_id".to_string(), bounded(tool_use_id, 256)),
    ]);
    if let Some(workspace_id) = input.payload.get("workspaceId").and_then(Value::as_str) {
        document
            .filters
            .insert("workspace_id".into(), bounded(workspace_id, 256));
    }
    let truncated = ["toolInput", "toolResponse", "toolError"]
        .iter()
        .any(|key| {
            input.payload.get(*key).is_some_and(|value| {
                tool_capture_is_truncated(value, 0)
                    || (*key == "toolInput"
                        && sanitize_tool_value(value, 0).len() > MAX_TOOL_CAPTURE_BYTES)
            })
        });
    document.filters.insert(
        "capture_completeness".to_string(),
        if truncated {
            "truncated"
        } else {
            "complete_sanitized"
        }
        .to_string(),
    );
    if let Some(session_id) = input
        .payload
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
    {
        document
            .filters
            .insert("session_id".into(), session_id.to_owned());
    }
    if let Some(turn_id) = input.payload.get("turnId").and_then(Value::as_str) {
        document
            .filters
            .insert("turn_id".to_string(), bounded(turn_id, 256));
    }
    let capture_id = crate::ids::stable_doc_id_from_source(&document.source);
    document
        .filters
        .insert("tool_capture_id".into(), capture_id.clone());
    let content = document.content.clone();
    let mut chunks = vec![];
    let mut offset = 0;
    while offset < content.len() {
        let part = bounded(&content[offset..], 32 * 1024);
        let mut chunk = document.clone();
        if !chunks.is_empty() {
            chunk.source = format!("{}/chunk/{}", document.source, chunks.len());
            chunk.doc_id = crate::ids::stable_doc_id_from_source(&chunk.source);
        }
        chunk.content = part;
        chunk.doc_length = chunk.content.len();
        chunk
            .filters
            .insert("chunk_offset_bytes".into(), offset.to_string());
        chunk
            .filters
            .insert("capture_size_bytes".into(), content.len().to_string());
        offset += chunk.content.len();
        chunks.push(chunk);
    }
    MemoryService::with_shared_memory(root, |memory| {
        memory.replace_source_group("tool_capture_id", &capture_id, chunks)
    })
}

// Context is useful even if the original result was too large for Roo's inline
// tool message. Do not claim durable capture here: the caller must ingest first.
fn tool_result_context(input: &HookInvocation) -> HookResult {
    let Some(result) = input
        .payload
        .get("toolResponse")
        .or_else(|| input.payload.get("toolError"))
    else {
        return HookResult::default();
    };
    let text = sanitize_tool_value(result, 0);
    let preview = bounded(&text, MAX_CONTEXT_BYTES.saturating_sub(512));
    let completeness = if preview.len() < text.len() {
        "partial preview"
    } else {
        "sanitized result"
    };
    HookResult {
        additional_context: Some(format!(
            "Tool evidence from Lint-AI ({completeness}). Tool: {}; call: {}.\n{}\nAdditional evidence is searchable in workspace memory; this preview is not an exhaustive dataset.",
            bounded(input.payload.get("toolName").and_then(Value::as_str).unwrap_or("unknown"), 128),
            bounded(input.payload.get("toolUseId").and_then(Value::as_str).unwrap_or("unknown"), 128),
            preview,
        )),
    }
}

fn tool_capture_is_truncated(value: &Value, depth: usize) -> bool {
    if depth >= 16 {
        return true;
    }
    match value {
        Value::Object(fields) => fields
            .values()
            .any(|value| tool_capture_is_truncated(value, depth + 1)),
        Value::Array(items) => items
            .iter()
            .any(|value| tool_capture_is_truncated(value, depth + 1)),
        Value::String(_) => false,
        _ => false,
    }
}

fn sanitize_tool_value(value: &Value, depth: usize) -> String {
    fn sanitize(value: &Value, depth: usize) -> Value {
        if depth >= 16 {
            return Value::String("[maximum nesting depth reached]".to_string());
        }
        match value {
            Value::Object(object) => Value::Object(
                object
                    .iter()
                    .map(|(key, value)| {
                        let key_lower = key.to_ascii_lowercase();
                        let sensitive = [
                            "api_key",
                            "apikey",
                            "authorization",
                            "bearer",
                            "cookie",
                            "credential",
                            "password",
                            "private_key",
                            "secret",
                            "token",
                        ]
                        .iter()
                        .any(|needle| key_lower.contains(needle));
                        (
                            key.clone(),
                            if sensitive {
                                Value::String("[REDACTED]".to_string())
                            } else {
                                sanitize(value, depth + 1)
                            },
                        )
                    })
                    .collect(),
            ),
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| sanitize(item, depth + 1)).collect())
            }
            Value::String(text) => {
                if crate::integrations::session_recording::contains_credential_material(text) {
                    Value::String("[REDACTED]".to_string())
                } else {
                    Value::String(text.clone())
                }
            }
            other => other.clone(),
        }
    }

    serde_json::to_string(&sanitize(value, depth)).unwrap_or_else(|_| "null".to_string())
}

fn bounded(value: &str, max_bytes: usize) -> String {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

fn event_name(kind: RooRuntimeHook) -> &'static str {
    match kind {
        RooRuntimeHook::RunStart => "run_start",
        RooRuntimeHook::AgentTurnStart => "agent_turn_start",
        RooRuntimeHook::AgentTurnEnd => "agent_turn_end",
        RooRuntimeHook::RunEnd => "run_end",
        RooRuntimeHook::PreToolUse => "pre_tool_use",
        RooRuntimeHook::PostToolUse => "post_tool_use",
    }
}

pub fn install_hooks(config_override: Option<&Path>) -> Result<PathBuf> {
    let path = config_override
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut config: Value = if path.exists() {
        serde_json::from_slice(&fs::read(&path)?)?
    } else {
        serde_json::json!({"schemaVersion": 1, "hooks": []})
    };
    let object = config
        .as_object_mut()
        .context("Roo hooks config must be a JSON object")?;
    object.entry("schemaVersion").or_insert(Value::from(1));
    let hooks = object
        .entry("hooks")
        .or_insert(Value::Array(vec![]))
        .as_array_mut()
        .context("Roo hooks config 'hooks' must be an array")?;
    hooks.retain(|hook| {
        !hook
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("lint-ai-roo-runtime-"))
    });
    let executable = std::env::current_exe()?.canonicalize()?;
    for kind in [
        RooRuntimeHook::RunStart,
        RooRuntimeHook::AgentTurnStart,
        RooRuntimeHook::AgentTurnEnd,
        RooRuntimeHook::RunEnd,
        RooRuntimeHook::PreToolUse,
        RooRuntimeHook::PostToolUse,
    ] {
        let name = event_name(kind);
        let profile = if matches!(
            kind,
            RooRuntimeHook::AgentTurnStart | RooRuntimeHook::PostToolUse
        ) {
            "context_provider"
        } else {
            "observer"
        };
        let timeout_seconds = if matches!(kind, RooRuntimeHook::PostToolUse) {
            15
        } else {
            5
        };
        hooks.push(serde_json::json!({
            "id": format!("lint-ai-roo-runtime-{name}"),
            "event": name,
            "handlers": [{ "type": "process", "executable": executable, "args": ["--roo-runtime-hook", name.replace('_', "-")], "sandbox_profile": profile, "require_enforcement": false, "timeout_seconds": timeout_seconds }]
        }));
    }
    let encoded = serde_json::to_vec_pretty(&config)?;
    fs::write(&path, encoded)?;
    Ok(path)
}

fn default_config_path() -> PathBuf {
    std::env::var_os("ROO_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rooagi")))
        .unwrap_or_else(|| PathBuf::from(".rooagi"))
        .join("hooks.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;
    #[test]
    fn event_values_match_roo_wire_names() {
        assert_eq!(
            event_name(RooRuntimeHook::AgentTurnStart),
            "agent_turn_start"
        );
        assert_eq!(
            serde_json::to_value(HookResult::default()).unwrap(),
            serde_json::json!({})
        );
    }
    #[test]
    fn tool_capture_preserves_items_beyond_256_and_redacts_secrets() {
        let value = serde_json::json!({
            "items": (0..300).collect::<Vec<_>>(),
            "password": "must-not-appear",
        });
        let sanitized: Value = serde_json::from_str(&sanitize_tool_value(&value, 0)).unwrap();
        assert_eq!(sanitized["items"].as_array().unwrap().len(), 300);
        assert_eq!(sanitized["items"][299], 299);
        assert_eq!(sanitized["password"], "[REDACTED]");
        assert!(!tool_capture_is_truncated(&value, 0));
    }

    #[test]
    fn post_tool_context_supplies_evidence_and_marks_partial_preview() {
        let input = HookInvocation {
            schema_version: 1,
            event: "post_tool_use".into(),
            run_id: "run".into(),
            timestamp: None,
            payload: serde_json::json!({"toolName": "fetch", "toolUseId": "call", "toolResponse": {"evidence": "a".repeat(MAX_CONTEXT_BYTES * 2), "password": "must-not-appear"}}),
        };
        let context = tool_result_context(&input).additional_context.unwrap();
        assert!(context.contains("partial preview"));
        assert!(context.contains("evidence"));
        assert!(!context.contains("must-not-appear"));
        assert!(context.len() <= MAX_CONTEXT_BYTES);
        assert!(!tool_capture_is_truncated(
            &Value::String("a".repeat(MAX_TOOL_CAPTURE_BYTES + 1)),
            0
        ));
    }

    #[test]
    fn recall_excerpt_finds_evidence_after_a_long_prefix() {
        let content = format!("{} decisive-evidence-299 is the answer", "é".repeat(4_000));
        let excerpt = relevant_excerpt(&content, "decisive-evidence-299", 2_000);
        assert!(excerpt.contains("decisive-evidence-299 is the answer"));
        assert!(excerpt.len() <= 2_000);
    }

    #[test]
    fn tool_result_retries_are_idempotent_and_workspace_isolated() {
        let base = std::env::temp_dir().join(format!(
            "lint-ai-roo-tools-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = base.join("a");
        let other = base.join("b");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&other).unwrap();
        let root = root.canonicalize().unwrap();
        let other = other.canonicalize().unwrap();
        let input = HookInvocation {
            schema_version: 1,
            event: "post_tool_use".into(),
            run_id: "first-run".into(),
            timestamp: None,
            payload: serde_json::json!({"toolName": "fetch", "toolUseId": "call-1", "toolResponse": {"items": (0..300).map(|i| format!("item-{i}")).collect::<Vec<_>>(), "finding": "unique-evidence-299"}}),
        };
        capture_tool_event(&root, &input, "tool_result").unwrap();
        capture_tool_event(&root, &input, "tool_result").unwrap();
        let search = HookInvocation {
            schema_version: 1,
            event: "agent_turn_start".into(),
            run_id: "later-run".into(),
            timestamp: None,
            payload: serde_json::json!({"userMessage": "unique-evidence-299"}),
        };
        let context = recall(&root, &search).unwrap().additional_context.unwrap();
        assert!(context.contains("unique-evidence-299"));
        assert_eq!(context.matches("unique-evidence-299").count(), 1);
        assert!(recall(&other, &search)
            .unwrap()
            .additional_context
            .is_none());
        let mut session_input = input;
        session_input.payload["sessionId"] = serde_json::json!("conversation-42");
        session_input.run_id = "session-run-a".into();
        capture_tool_event(&root, &session_input, "tool_result").unwrap();
        session_input.run_id = "session-run-b".into();
        capture_tool_event(&root, &session_input, "tool_result").unwrap();
        MemoryService::with_shared_memory(&root, |memory| {
            let documents = memory.source_documents();
            let attributed: Vec<_> = documents
                .iter()
                .filter(|doc| {
                    doc.filters.get("session_id").map(String::as_str) == Some("conversation-42")
                })
                .collect();
            assert_eq!(attributed.len(), 2);
            assert_eq!(attributed[0].group_id, attributed[1].group_id);
            assert_eq!(
                attributed[0].group_id.as_deref(),
                Some(memory_session_id(root.to_str().unwrap(), &session_input).as_str())
            );
            assert_ne!(
                attributed[0].filters["run_id"],
                attributed[1].filters["run_id"]
            );
            assert_ne!(attributed[0].doc_id, attributed[1].doc_id);
            Ok(())
        })
        .unwrap();
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn full_recall_budget_still_returns_context() {
        let context = format!(
            "Relevant project memory from Lint-AI:\nbudget-proof {}\n",
            "evidence ".repeat(2_000)
        );
        let result = recall_context_result(&context)
            .additional_context
            .expect("truncation must not discard recall");
        assert!(result.contains("budget-proof"));
        assert!(result.len() <= MAX_CONTEXT_BYTES);
        assert!(result.len() > 7_000);
    }

    #[test]
    fn referenced_results_are_chunked_and_shrinking_replays_remove_old_evidence() {
        let root = std::env::temp_dir().join(format!(
            "roo-ref-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join(".roo/hook-payloads")).unwrap();
        let root = root.canonicalize().unwrap();
        let path = root.join(".roo/hook-payloads/result.json");
        let result =
            serde_json::json!({"text": format!("{} late-evidence-913", "prefix ".repeat(15_000))});
        let bytes = serde_json::to_vec(&result).unwrap();
        fs::write(&path, &bytes).unwrap();
        use sha2::Digest;
        let mut input = HookInvocation {
            schema_version: 1,
            event: "post_tool_use".into(),
            run_id: "run".into(),
            timestamp: None,
            payload: serde_json::json!({"toolName":"fetch","toolUseId":"call", "payloadReferences":{"toolResponse":{"path":path,"sizeBytes":bytes.len(),"encoding":"json","sha256":format!("{:x}", sha2::Sha256::digest(&bytes))}}}),
        };
        load_payload_references(&root, &mut input.payload).unwrap();
        assert_eq!(input.payload["toolResponse"], result);
        capture_tool_event(&root, &input, "tool_result").unwrap();
        let query = HookInvocation {
            schema_version: 1,
            event: "agent_turn_start".into(),
            run_id: "later".into(),
            timestamp: None,
            payload: serde_json::json!({"userMessage":"late-evidence-913"}),
        };
        assert!(recall(&root, &query)
            .unwrap()
            .additional_context
            .unwrap()
            .contains("late-evidence-913"));
        let count =
            MemoryService::with_shared_memory(&root, |memory| Ok(memory.source_documents().len()))
                .unwrap();
        assert!(count > 1);
        input.payload["toolResponse"] = serde_json::json!({"finding":"replacement evidence"});
        capture_tool_event(&root, &input, "tool_result").unwrap();
        assert_eq!(
            MemoryService::with_shared_memory(&root, |memory| Ok(memory.source_documents().len()))
                .unwrap(),
            1
        );
        let mut bad_reference = serde_json::json!({"payloadReferences":{"toolResponse":{"path":root.join("outside.json"),"sizeBytes":2,"encoding":"json","sha256":"wrong"}}});
        fs::write(root.join("outside.json"), "{}").unwrap();
        assert!(load_payload_references(&root, &mut bad_reference).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_text_preserves_utf8() {
        assert_eq!(bounded("éé", 3), "é");
    }
    #[test]
    fn installer_preserves_other_hooks_and_replaces_its_own() {
        let dir = std::env::temp_dir().join(format!("lint-ai-roo-install-{}", std::process::id()));
        let path = dir.join("hooks.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{"schemaVersion":1,"hooks":[{"id":"user-hook","event":"stop","handlers":[]}] }"#,
        )
        .unwrap();
        install_hooks(Some(&path)).unwrap();
        install_hooks(Some(&path)).unwrap();
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let hooks = value["hooks"].as_array().unwrap();
        assert_eq!(hooks.len(), 7);
        assert_eq!(hooks[0]["id"], "user-hook");
        assert_eq!(hooks[2]["event"], "agent_turn_start");
        assert_eq!(
            hooks[2]["handlers"][0]["sandbox_profile"],
            "context_provider"
        );
        assert_eq!(hooks[2]["handlers"][0]["timeout_seconds"], 5);
        assert_eq!(
            hooks[6]["handlers"][0]["sandbox_profile"],
            "context_provider"
        );
        assert_eq!(
            hooks[2]["handlers"][0]["args"],
            serde_json::json!([
                "--roo-runtime-hook",
                RooRuntimeHook::AgentTurnStart
                    .to_possible_value()
                    .unwrap()
                    .get_name()
            ])
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn captured_turn_is_retrievable_from_the_shared_store() {
        let root = std::env::temp_dir().join(format!(
            "lint-ai-roo-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let invocation = HookInvocation {
            schema_version: 1,
            event: "agent_turn_end".to_string(),
            run_id: "roo-run-capture-test".to_string(),
            timestamp: Some("2026-10-04T00:00:00Z".to_string()),
            payload: serde_json::json!({
                "turnId": "roo-run-capture-test:agent:1",
                "userMessage": "What storage rule did we choose for integration-test-unique-42?",
                "assistantResponse": "We chose the shared MemoryService store for integration-test-unique-42.",
                "status": "completed",
                "workspaceRoot": root,
            }),
        };

        capture(&root, &invocation, "turn").unwrap();
        let result = recall(&root, &invocation).unwrap();
        let context = result
            .additional_context
            .expect("captured turn must be recalled");
        assert!(context.contains("integration-test-unique-42"));

        let _ = fs::remove_dir_all(root);
    }
}
