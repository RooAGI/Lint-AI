use crate::integrations::session_recording::{
    lint_ai_enabled, record_event_if_enabled, RecordingProvider,
};
use crate::memory_api::MemoryService;
use crate::pipeline::{MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiHookKind {
    SessionStart,
    BeforeAgent,
    AfterAgent,
    BeforeModel,
    BeforeToolSelection,
    BeforeTool,
    AfterTool,
    PreCompress,
    SessionEnd,
}

impl GeminiHookKind {
    pub fn event_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::BeforeAgent => "BeforeAgent",
            Self::AfterAgent => "AfterAgent",
            Self::BeforeModel => "BeforeModel",
            Self::BeforeToolSelection => "BeforeToolSelection",
            Self::BeforeTool => "BeforeTool",
            Self::AfterTool => "AfterTool",
            Self::PreCompress => "PreCompress",
            Self::SessionEnd => "SessionEnd",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiHookInput {
    pub session_id: String,
    #[serde(default)]
    pub transcript_path: Option<PathBuf>,
    #[serde(default)]
    pub cwd: PathBuf,
    pub hook_event_name: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_input: Option<Value>,
    #[serde(default)]
    pub tool_response: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GeminiHookOutput {
    #[serde(rename = "hookSpecificOutput", skip_serializing_if = "Option::is_none")]
    pub hook_specific_output: Option<GeminiHookSpecificOutput>,
    #[serde(rename = "systemMessage", skip_serializing_if = "Option::is_none")]
    pub system_message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiHookSpecificOutput {
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    #[serde(rename = "additionalContext")]
    pub additional_context: String,
}

pub fn run_hook(kind: GeminiHookKind, fallback_root: &Path) -> Result<()> {
    run_hook_for(kind, fallback_root, RecordingProvider::Gemini, "Gemini")
}

/// Run the Gemini-compatible hook protocol for another CLI adapter.
pub fn run_hook_for(
    kind: GeminiHookKind,
    fallback_root: &Path,
    provider: RecordingProvider,
    provider_label: &str,
) -> Result<()> {
    let raw: Value =
        crate::integrations::read_bounded_json().context("failed to parse Gemini hook input")?;
    let input: GeminiHookInput =
        serde_json::from_value(raw.clone()).context("failed to decode Gemini hook input")?;
    let root = resolve_root(&input.cwd, fallback_root)?;
    if let Err(error) =
        record_event_if_enabled(provider, &root, &input.session_id, kind.event_name(), raw)
    {
        eprintln!("warning: Lint-AI {provider_label} session recording failed open: {error:#}");
    }
    let output =
        handle_hook(kind, &input, &root, provider, provider_label).unwrap_or_else(|error| {
            eprintln!("warning: Lint-AI {provider_label} hook failed open: {error:#}");
            GeminiHookOutput::default()
        });
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &output)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn handle_hook(
    kind: GeminiHookKind,
    input: &GeminiHookInput,
    _root: &Path,
    provider: RecordingProvider,
    provider_label: &str,
) -> Result<GeminiHookOutput> {
    if input.hook_event_name != kind.event_name() {
        anyhow::bail!("{provider_label} hook event mismatch")
    }
    if kind == GeminiHookKind::SessionStart {
        return Ok(GeminiHookOutput {
            system_message: Some(
                "Lint-AI hooks active; session recording follows the configured recording state."
                    .into(),
            ),
            ..Default::default()
        });
    }
    if !lint_ai_enabled(provider, _root)? {
        return Ok(GeminiHookOutput::default());
    }
    match kind {
        GeminiHookKind::AfterAgent => return capture(_root, input, provider, "outcome"),
        GeminiHookKind::PreCompress => return capture(_root, input, provider, "checkpoint"),
        GeminiHookKind::SessionEnd => return capture(_root, input, provider, "session-summary"),
        _ => {}
    }
    let query = input
        .prompt
        .clone()
        .or_else(|| input.tool_name.clone())
        .unwrap_or_default();
    if query.trim().is_empty() {
        return Ok(GeminiHookOutput::default());
    }
    let started = std::time::Instant::now();
    let memory = crate::integrations::mcp_index::shared_memory_root(_root);
    if !memory.exists() {
        let _ = crate::telemetry::record_memory_retrieval(
            _root,
            provider.as_str(),
            &input.session_id,
            &query,
            0,
            &[],
        );
        return Ok(GeminiHookOutput::default());
    }
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut store = MemoryService::at_path(&memory, options)?;
    if store.is_empty() {
        let _ = crate::telemetry::record_memory_retrieval(
            _root,
            provider.as_str(),
            &input.session_id,
            &query,
            started.elapsed().as_millis() as u64,
            &[],
        );
        return Ok(GeminiHookOutput::default());
    }
    let results = store.observe_plain_query(&query, provider.as_str(), Some(&input.session_id), 5);
    let _ = crate::telemetry::record_project_query(
        _root,
        started.elapsed().as_millis() as u64,
        results.is_err(),
        results.as_ref().is_ok_and(Vec::is_empty),
    );
    let mut context = String::new();
    let mut retrieved_memories = Vec::new();
    for result in results? {
        if let Some(record) = store.record_by_id(&result.doc_id) {
            let excerpt = record
                .content
                .lines()
                .take(8)
                .collect::<Vec<_>>()
                .join("\n");
            if !excerpt.trim().is_empty() {
                context.push_str(&format!("\n- Source: {}\n  {}\n", record.source, excerpt));
                retrieved_memories.push(serde_json::json!({
                    "source": record.source,
                    "type": record.filters.get("document_type").map(String::as_str).unwrap_or("memory"),
                    "score": result.score,
                    "excerpt": excerpt,
                }));
            }
            if context.len() > 8_000 {
                break;
            }
        }
    }
    let _ = crate::telemetry::record_memory_retrieval(
        _root,
        provider.as_str(),
        &input.session_id,
        &query,
        started.elapsed().as_millis() as u64,
        &retrieved_memories,
    );
    if context.is_empty() {
        return Ok(GeminiHookOutput::default());
    }
    Ok(GeminiHookOutput {
        hook_specific_output: Some(GeminiHookSpecificOutput {
            hook_event_name: kind.event_name().into(),
            additional_context: context,
        }),
        system_message: None,
    })
}

fn capture(
    root: &Path,
    input: &GeminiHookInput,
    provider: RecordingProvider,
    document_type: &str,
) -> Result<GeminiHookOutput> {
    let content = capture_content(input)?;
    if content.trim().is_empty() {
        return Ok(GeminiHookOutput::default());
    }
    let source = format!(
        "{}://{}/{}",
        provider.as_str(),
        input.session_id,
        document_type
    );
    let mut filters = BTreeMap::from([
        ("provider".to_string(), provider.as_str().to_string()),
        ("session_id".to_string(), input.session_id.clone()),
        ("document_type".to_string(), document_type.to_string()),
    ]);
    filters.insert("source_type".to_string(), "session-memory".to_string());
    let document = SourceDocument {
        doc_id: crate::ids::stable_doc_id_from_source(&source),
        source,
        content: format!("{} {document_type}\n{content}", provider.as_str()),
        concept: document_type.to_string(),
        group_id: Some(format!(
            "{}-session:{}",
            provider.as_str(),
            input.session_id
        )),
        headings: vec![document_type.to_string()],
        links: vec![],
        timestamp: Some(current_timestamp()),
        doc_length: content.len(),
        author_agent: Some(provider.as_str().to_string()),
        filters,
    };
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut store = MemoryService::at_path(
        &crate::integrations::mcp_index::shared_memory_root(root),
        options,
    )?;
    store.upsert(document);
    store.refresh_index()?;
    Ok(GeminiHookOutput::default())
}

fn current_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|timestamp| timestamp.to_rfc3339())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string())
}

fn capture_content(input: &GeminiHookInput) -> Result<String> {
    let value = if let Some(path) = input.transcript_path.as_deref() {
        serde_json::from_str::<Value>(&fs::read_to_string(path).with_context(|| {
            format!(
                "failed to read Gemini-compatible transcript {}",
                path.display()
            )
        })?)
        .unwrap_or_else(|_| Value::String(fs::read_to_string(path).unwrap_or_default()))
    } else {
        let mut value = Map::new();
        if let Some(response) = input.tool_response.clone() {
            value.insert("response".to_string(), response);
        }
        for key in ["response", "result", "output", "message", "content", "text"] {
            if let Some(response) = input.extra.get(key).cloned() {
                value.insert(key.to_string(), response);
            }
        }
        Value::Object(value)
    };
    let mut text = Vec::new();
    collect_capture_text(&value, &mut text);
    Ok(redact_capture_text(&text.join("\n")))
}

fn collect_capture_text(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) if !text.trim().is_empty() => output.push(text.trim().to_string()),
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_capture_text(value, output)),
        Value::Object(values) => {
            for key in ["content", "text", "message", "output", "response", "result"] {
                if let Some(value) = values.get(key) {
                    collect_capture_text(value, output);
                }
            }
        }
        _ => {}
    }
}

fn redact_capture_text(value: &str) -> String {
    value
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let lower = line.to_ascii_lowercase();
            if ["api_key", "authorization", "password", "secret", "token"]
                .iter()
                .any(|marker| lower.contains(marker))
                || crate::integrations::session_recording::contains_credential_material(line)
            {
                Some("[REDACTED]".to_string())
            } else {
                Some(line.chars().take(8_000).collect())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn resolve_root(cwd: &Path, fallback: &Path) -> Result<PathBuf> {
    let fallback = fallback.canonicalize().with_context(|| {
        format!(
            "failed to resolve Gemini fallback root {}",
            fallback.display()
        )
    })?;
    let candidate = if cwd.as_os_str().is_empty() {
        fallback.clone()
    } else {
        cwd.canonicalize()?
    };
    if !candidate.starts_with(&fallback) {
        anyhow::bail!("Gemini hook cwd is outside configured project root")
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::IndexStore;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-gemini-hook-{name}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }
    #[test]
    fn parses_gemini_hook_payload_and_event_names() {
        let input: GeminiHookInput = serde_json::from_value(serde_json::json!({
            "session_id":"s1","cwd":"/tmp","hook_event_name":"BeforeTool",
            "tool_name":"read_file","tool_input":{"path":"README.md"}
        }))
        .unwrap();
        assert_eq!(input.tool_name.as_deref(), Some("read_file"));
        assert_eq!(GeminiHookKind::BeforeTool.event_name(), "BeforeTool");
    }

    #[test]
    fn gemini_compatible_hooks_capture_persist_and_retrieve_for_both_providers() {
        for (provider, label) in [
            (RecordingProvider::Gemini, "Gemini"),
            (RecordingProvider::Agy, "Antigravity"),
        ] {
            let root = temp_root(label);
            let transcript = root.join("transcript.json");
            fs::write(
                &transcript,
                r#"{"message":{"content":"Implemented blue canary deployment routing\napi_key=do-not-store"}}"#,
            )
            .unwrap();

            let capture_input = GeminiHookInput {
                session_id: "session-1".to_string(),
                transcript_path: Some(transcript),
                cwd: root.clone(),
                hook_event_name: "AfterAgent".to_string(),
                prompt: None,
                tool_name: None,
                tool_input: None,
                tool_response: None,
                extra: Map::new(),
            };
            handle_hook(
                GeminiHookKind::AfterAgent,
                &capture_input,
                &root,
                provider,
                label,
            )
            .unwrap();
            handle_hook(
                GeminiHookKind::AfterAgent,
                &capture_input,
                &root,
                provider,
                label,
            )
            .unwrap();
            let mut store = IndexStore::at_path(
                &crate::integrations::mcp_index::shared_memory_root(&root),
                crate::integrations::mcp_index::segmented_store_options(),
            )
            .unwrap();
            store.refresh().unwrap();
            assert_eq!(store.records().len(), 1, "capture must be idempotent");
            assert!(store.records()[0].content.contains("[REDACTED]"));
            assert!(!store.records()[0].content.contains("do-not-store"));
            let timestamp = store.records()[0]
                .timestamp
                .as_deref()
                .expect("captured memory must have a timestamp");
            chrono::DateTime::parse_from_rfc3339(timestamp)
                .expect("captured memory timestamp must be RFC3339");
            drop(store);

            let input = GeminiHookInput {
                session_id: "session-1".to_string(),
                transcript_path: None,
                cwd: root.clone(),
                hook_event_name: "BeforeAgent".to_string(),
                prompt: Some("How does deployment routing work?".to_string()),
                tool_name: None,
                tool_input: None,
                tool_response: None,
                extra: Map::new(),
            };
            let output =
                handle_hook(GeminiHookKind::BeforeAgent, &input, &root, provider, label).unwrap();
            assert!(output
                .hook_specific_output
                .expect("memory should be injected")
                .additional_context
                .contains("Implemented blue canary deployment routing"));
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn gemini_and_agy_memories_share_one_store_with_provider_attribution() {
        let root = temp_root("provider-memory-sharing");
        for (provider, label, content) in [
            (
                RecordingProvider::Gemini,
                "Gemini",
                "Gemini-only blue routing decision",
            ),
            (
                RecordingProvider::Agy,
                "Antigravity",
                "AGY-only green deployment decision",
            ),
        ] {
            let mut extra = Map::new();
            extra.insert("response".to_string(), Value::String(content.to_string()));
            let input = GeminiHookInput {
                session_id: "session-1".to_string(),
                transcript_path: None,
                cwd: root.clone(),
                hook_event_name: "AfterAgent".to_string(),
                prompt: None,
                tool_name: None,
                tool_input: None,
                tool_response: None,
                extra,
            };
            handle_hook(GeminiHookKind::AfterAgent, &input, &root, provider, label).unwrap();
        }

        // Both providers land in the shared store; the provider travels on the
        // documents instead of in the directory layout.
        let mut shared = IndexStore::at_path(
            &crate::integrations::mcp_index::shared_memory_root(&root),
            crate::integrations::mcp_index::segmented_store_options(),
        )
        .unwrap();
        shared.refresh().unwrap();
        assert_eq!(shared.records().len(), 2);
        let contents: Vec<&str> = shared
            .records()
            .iter()
            .map(|record| record.content.as_str())
            .collect();
        assert!(contents.iter().any(|c| c.contains("Gemini-only")));
        assert!(contents.iter().any(|c| c.contains("AGY-only")));
        for document in shared.source_documents() {
            let provider = document.filters.get("provider").unwrap();
            assert!(provider == "gemini-cli" || provider == "agy");
            assert!(document
                .group_id
                .as_deref()
                .unwrap()
                .starts_with(&format!("{provider}-session:")));
        }
        // The legacy per-provider silos are gone.
        assert!(!root.join(".lint-ai/gemini-cli-memory").exists());
        assert!(!root.join(".lint-ai/agy-memory").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
