use super::{
    analyze_corpus, build_llm_context_output, build_matcher, build_memory_index, chunk_graph_stats,
    compute_corpus_fingerprint, debug_phrase_matches, export_chunk_graph_cytoscape_html,
    export_chunk_graph_dot, export_chunk_graph_json, export_entity_graph_cytoscape_html,
    export_entity_graph_dot, export_entity_graph_json, export_graph_cytoscape_html,
    export_graph_dot, export_graph_json, export_ontology_json, graph_to_source_documents,
    load_cached_query_index, query_cache_lexical_dir, save_cached_query_index,
    show_concepts_by_section, show_tier1_entities, show_tier1_terms, write_tier0_index,
    CacheSettings, QueryOutput, DEFAULT_QUERY_TOP_K, LLM_CONTEXT_CANDIDATE_TOP_K, MAX_RESULT_COUNT,
};
use crate::aggregation::build_aggregate_output;
use crate::cli::{GraphExportFormat, GraphLevel, IndexInspectView};
use crate::config::{load_config, normalize_list};
use crate::graph::Graph;
#[cfg(feature = "agy")]
use crate::integrations::agy::hooks::{run_hook as run_agy_hook, AgyHookKind};
#[cfg(feature = "agy")]
use crate::integrations::agy::{
    install_hook_settings as install_agy_hook_settings,
    install_memory_skill as install_agy_memory_skill,
    install_user_config as install_agy_user_config, run_server as run_agy_server, AgyServerOptions,
};
#[cfg(feature = "claude-code")]
use crate::integrations::claude_code::hooks::{run_hook, ClaudeHookKind};
#[cfg(feature = "claude-code")]
use crate::integrations::claude_code::{
    install_hook_settings, install_memory_skill, install_user_config, run_server, run_status_line,
    ClaudeCodeServerOptions,
};
#[cfg(feature = "codex")]
use crate::integrations::codex::hooks::{run_hook as run_codex_hook, CodexHookKind};
#[cfg(feature = "codex")]
use crate::integrations::codex::{
    install_hook_settings as install_codex_hook_settings,
    install_memory_policy as install_codex_memory_policy,
    install_user_config as install_codex_user_config, run_server as run_codex_server,
    run_status_line as run_codex_status_line, CodexServerOptions,
};
#[cfg(feature = "gemini-cli")]
use crate::integrations::gemini_cli::hooks::{run_hook as run_gemini_hook, GeminiHookKind};
#[cfg(feature = "gemini-cli")]
use crate::integrations::gemini_cli::install_hook_settings as install_gemini_hook_settings;
#[cfg(feature = "gemini-cli")]
use crate::integrations::gemini_cli::{
    install_user_config as install_gemini_user_config, run_server as run_gemini_server,
    GeminiCliServerOptions,
};
#[cfg(feature = "muse-code")]
use crate::integrations::muse_code::{
    hooks::run_hook as run_muse_hook, hooks::MuseHookKind,
    install_hook_settings as install_muse_hook_settings,
    install_memory_policy as install_muse_memory_policy,
    install_user_config as install_muse_user_config, run_server as run_muse_server,
    MuseServerOptions,
};
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::query_plan::PreparedQuery;
use crate::report::Report;
use crate::rules::cross_refs::check_cross_refs;
use crate::rules::orphan_pages::check_orphans;
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

/// Run the lint pipeline using CLI arguments.
///
/// This is the main entry point used by the CLI wrapper in `main.rs`.
pub fn run(args: crate::cli::Args) -> Result<()> {
    if let Some(index_path) = args.inspect_index.as_deref() {
        return inspect_index_store(Path::new(index_path), args.inspect_view);
    }

    if args.recall_server.is_some() {
        #[cfg(any(feature = "claude-code", feature = "codex"))]
        {
            let root = args.recall_server.as_deref().expect("checked above");
            let cfg = load_config(
                args.config.as_deref(),
                root,
                args.strict_config,
                args.max_config_bytes,
            )
            .map_err(|err| anyhow::anyhow!(err))?;
            return crate::integrations::recall::run_recall_server(
                Path::new(root),
                &cfg.ignore_paths,
                args.max_bytes,
                args.max_files,
                args.max_depth,
                args.max_total_bytes,
            );
        }
        #[cfg(not(any(feature = "claude-code", feature = "codex")))]
        {
            anyhow::bail!("recall server requires the Claude Code or Codex feature");
        }
    }

    if args.recall.is_some() {
        #[cfg(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        ))]
        {
            let query = args.recall.as_deref().expect("checked above");
            // Without the configured ignores the top hit of a project root is a
            // vendored node_modules README rather than anything the caller asked
            // about, so the config is loaded even though nothing is linted here.
            let cfg = load_config(
                args.config.as_deref(),
                &args.path,
                args.strict_config,
                args.max_config_bytes,
            )
            .map_err(|err| anyhow::anyhow!(err))?;
            let memory_name = crate::integrations::mcp_index::SHARED_MEMORY_DIR;
            let output =
                crate::integrations::recall::recall(&crate::integrations::recall::RecallOptions {
                    root: Path::new(&args.path),
                    query,
                    result_count: args.result_count.clamp(1, MAX_RESULT_COUNT),
                    ignore_paths: &cfg.ignore_paths,
                    max_bytes: args.max_bytes,
                    max_files: args.max_files,
                    max_depth: args.max_depth,
                    max_total_bytes: args.max_total_bytes,
                    memory_name,
                })?;
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }
        #[cfg(not(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        )))]
        {
            anyhow::bail!("recall requires the Claude Code or Codex feature");
        }
    }

    if args.promote_session.is_some() {
        #[cfg(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        ))]
        {
            let session_id = args.promote_session.as_deref().expect("checked above");
            let provider = match args.session_provider {
                crate::cli::SessionProvider::Claude => {
                    crate::integrations::session_recording::RecordingProvider::Claude
                }
                crate::cli::SessionProvider::Codex => {
                    crate::integrations::session_recording::RecordingProvider::Codex
                }
                crate::cli::SessionProvider::Gemini => {
                    crate::integrations::session_recording::RecordingProvider::Gemini
                }
                crate::cli::SessionProvider::Agy => {
                    crate::integrations::session_recording::RecordingProvider::Agy
                }
                #[cfg(feature = "muse-code")]
                crate::cli::SessionProvider::Muse => {
                    crate::integrations::session_recording::RecordingProvider::Muse
                }
            };
            let report = crate::integrations::session_recording::promote_recorded_session(
                provider,
                Path::new(&args.path),
                args.session_root.as_deref().map(Path::new),
                session_id,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        #[cfg(not(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        )))]
        {
            anyhow::bail!("session promotion requires the Claude Code or Codex feature");
        }
    }

    if args.replay_session.is_some() {
        #[cfg(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        ))]
        {
            let session_id = args.replay_session.as_deref().expect("checked above");
            let provider = match args.session_provider {
                crate::cli::SessionProvider::Claude => {
                    crate::integrations::session_recording::RecordingProvider::Claude
                }
                crate::cli::SessionProvider::Codex => {
                    crate::integrations::session_recording::RecordingProvider::Codex
                }
                crate::cli::SessionProvider::Gemini => {
                    crate::integrations::session_recording::RecordingProvider::Gemini
                }
                crate::cli::SessionProvider::Agy => {
                    crate::integrations::session_recording::RecordingProvider::Agy
                }
                #[cfg(feature = "muse-code")]
                crate::cli::SessionProvider::Muse => {
                    crate::integrations::session_recording::RecordingProvider::Muse
                }
            };
            let report = crate::integrations::session_recording::replay_recorded_session(
                provider,
                Path::new(&args.path),
                args.session_root.as_deref().map(Path::new),
                session_id,
                args.replay_enable_lint_ai,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        #[cfg(not(any(
            feature = "claude-code",
            feature = "codex",
            feature = "gemini-cli",
            feature = "agy",
            feature = "muse-code"
        )))]
        {
            anyhow::bail!("session replay requires the Claude Code or Codex feature");
        }
    }

    #[cfg(feature = "claude-code")]
    if args.claude_code_statusline {
        return run_status_line();
    }

    #[cfg(feature = "claude-code")]
    if let Some(hook) = args.claude_code_hook {
        let kind = match hook {
            crate::cli::ClaudeCodeHook::SessionStart => ClaudeHookKind::SessionStart,
            crate::cli::ClaudeCodeHook::UserPromptSubmit => ClaudeHookKind::UserPromptSubmit,
            crate::cli::ClaudeCodeHook::UserPromptExpansion => ClaudeHookKind::UserPromptExpansion,
            crate::cli::ClaudeCodeHook::PreToolUse => ClaudeHookKind::PreToolUse,
            crate::cli::ClaudeCodeHook::PostToolUse => ClaudeHookKind::PostToolUse,
            crate::cli::ClaudeCodeHook::PreCompact => ClaudeHookKind::PreCompact,
            crate::cli::ClaudeCodeHook::Stop => ClaudeHookKind::Stop,
            crate::cli::ClaudeCodeHook::SessionEnd => ClaudeHookKind::SessionEnd,
            crate::cli::ClaudeCodeHook::SubagentStart => ClaudeHookKind::SubagentStart,
            crate::cli::ClaudeCodeHook::SubagentStop => ClaudeHookKind::SubagentStop,
        };
        return run_hook(kind, Path::new(&args.path));
    }

    #[cfg(feature = "codex")]
    if args.codex_statusline {
        return run_codex_status_line();
    }

    #[cfg(feature = "codex")]
    if let Some(hook) = args.codex_hook {
        let kind = match hook {
            crate::cli::CodexHook::SessionStart => CodexHookKind::SessionStart,
            crate::cli::CodexHook::UserPromptSubmit => CodexHookKind::UserPromptSubmit,
            crate::cli::CodexHook::PreToolUse => CodexHookKind::PreToolUse,
            crate::cli::CodexHook::PermissionRequest => CodexHookKind::PermissionRequest,
            crate::cli::CodexHook::PostToolUse => CodexHookKind::PostToolUse,
            crate::cli::CodexHook::UserPromptExpansion => CodexHookKind::UserPromptExpansion,
            crate::cli::CodexHook::PreCompact => CodexHookKind::PreCompact,
            crate::cli::CodexHook::PostCompact => CodexHookKind::PostCompact,
            crate::cli::CodexHook::Stop => CodexHookKind::Stop,
            crate::cli::CodexHook::SessionEnd => CodexHookKind::SessionEnd,
            crate::cli::CodexHook::SubagentStart => CodexHookKind::SubagentStart,
            crate::cli::CodexHook::SubagentStop => CodexHookKind::SubagentStop,
        };
        return run_codex_hook(kind, Path::new(&args.path));
    }

    #[cfg(feature = "gemini-cli")]
    if let Some(hook) = args.gemini_cli_hook {
        let kind = match hook {
            crate::cli::GeminiCliHook::SessionStart => GeminiHookKind::SessionStart,
            crate::cli::GeminiCliHook::BeforeAgent => GeminiHookKind::BeforeAgent,
            crate::cli::GeminiCliHook::AfterAgent => GeminiHookKind::AfterAgent,
            crate::cli::GeminiCliHook::BeforeModel => GeminiHookKind::BeforeModel,
            crate::cli::GeminiCliHook::BeforeToolSelection => GeminiHookKind::BeforeToolSelection,
            crate::cli::GeminiCliHook::BeforeTool => GeminiHookKind::BeforeTool,
            crate::cli::GeminiCliHook::AfterTool => GeminiHookKind::AfterTool,
            crate::cli::GeminiCliHook::PreCompress => GeminiHookKind::PreCompress,
            crate::cli::GeminiCliHook::SessionEnd => GeminiHookKind::SessionEnd,
        };
        return run_gemini_hook(kind, Path::new(&args.path));
    }

    #[cfg(feature = "agy")]
    if let Some(hook) = args.agy_hook {
        let kind = match hook {
            crate::cli::AgyHook::PreToolUse => AgyHookKind::PreToolUse,
            crate::cli::AgyHook::PostToolUse => AgyHookKind::PostToolUse,
            crate::cli::AgyHook::PreInvocation => AgyHookKind::PreInvocation,
            crate::cli::AgyHook::PostInvocation => AgyHookKind::PostInvocation,
            crate::cli::AgyHook::Stop => AgyHookKind::Stop,
        };
        return run_agy_hook(kind, Path::new(&args.path));
    }

    #[cfg(feature = "claude-code")]
    if args.claude_code_install {
        let written = install_memory_skill(Path::new(&args.path), args.claude_code_force_skill)?;
        println!("Wrote Claude Code memory skill to {}", written.display());
        let config_path = args.claude_code_config.as_deref().map(Path::new);
        let written = install_user_config(Path::new(&args.path), config_path)?;
        println!("Wrote Claude Code config to {}", written.display());
        let settings_path = args.claude_code_settings.as_deref().map(Path::new);
        let written = install_hook_settings(Path::new(&args.path), settings_path)?;
        println!("Wrote Claude Code hook settings to {}", written.display());
        return Ok(());
    }

    #[cfg(feature = "codex")]
    if args.codex_install {
        let config_path = args.codex_config.as_deref().map(Path::new);
        let written = install_codex_user_config(Path::new(&args.path), config_path)?;
        println!("Wrote Codex config to {}", written.display());
        let settings_path = args.codex_settings.as_deref().map(Path::new);
        let written = install_codex_hook_settings(Path::new(&args.path), settings_path)?;
        println!("Wrote Codex hook settings to {}", written.display());
        let written = install_codex_memory_policy(Path::new(&args.path))?;
        println!("Wrote Codex memory policy to {}", written.display());
        return Ok(());
    }

    #[cfg(feature = "muse-code")]
    if args.muse_install {
        let config_path = args.muse_config.as_deref().map(Path::new);
        let written = install_muse_user_config(Path::new(&args.path), config_path)?;
        println!("Wrote Muse Code config to {}", written.display());
        let written = install_muse_hook_settings(Path::new(&args.path), config_path)?;
        println!("Wrote Muse Code hook settings to {}", written.display());
        let written = install_muse_memory_policy(Path::new(&args.path))?;
        println!("Wrote Muse Code memory policy to {}", written.display());
        return Ok(());
    }

    #[cfg(feature = "muse-code")]
    if let Some(hook) = args.muse_hook {
        let kind = match hook {
            crate::cli::MuseHook::SessionStart => MuseHookKind::SessionStart,
            crate::cli::MuseHook::UserPromptSubmit => MuseHookKind::UserPromptSubmit,
            crate::cli::MuseHook::PreToolUse => MuseHookKind::PreToolUse,
            crate::cli::MuseHook::PostToolUse => MuseHookKind::PostToolUse,
            crate::cli::MuseHook::PostToolUseFailure => MuseHookKind::PostToolUseFailure,
            crate::cli::MuseHook::Stop => MuseHookKind::Stop,
            crate::cli::MuseHook::SessionEnd => MuseHookKind::SessionEnd,
        };
        return run_muse_hook(kind, Path::new(&args.path));
    }

    #[cfg(feature = "gemini-cli")]
    if args.gemini_cli_install {
        let written = install_gemini_user_config(
            Path::new(&args.path),
            args.gemini_cli_config.as_deref().map(Path::new),
        )?;
        println!("Wrote Gemini CLI config to {}", written.display());
        let written = install_gemini_hook_settings(
            Path::new(&args.path),
            args.gemini_cli_settings.as_deref().map(Path::new),
        )?;
        println!("Wrote Gemini CLI hook settings to {}", written.display());
        return Ok(());
    }

    #[cfg(feature = "gemini-cli")]
    if args.gemini_cli_serve {
        let cfg = load_config(
            args.config.as_deref(),
            &args.path,
            args.strict_config,
            args.max_config_bytes,
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        run_gemini_server(
            Path::new(&args.path),
            GeminiCliServerOptions {
                max_bytes: args.max_bytes,
                max_files: args.max_files,
                max_depth: args.max_depth,
                max_total_bytes: args.max_total_bytes,
                ignore_paths: &cfg.ignore_paths,
            },
        )?;
        return Ok(());
    }

    #[cfg(feature = "gemini-cli")]
    if args.gemini_cli_verify_mcp {
        crate::integrations::mcp_health::verify(
            Path::new(&args.path),
            "--gemini-cli-serve",
            args.mcp_timeout_ms,
        )?;
        return Ok(());
    }

    #[cfg(feature = "agy")]
    if args.agy_install {
        let written = install_agy_memory_skill(Path::new(&args.path), args.agy_force_skill)?;
        println!("Wrote AGY memory skill to {}", written.display());
        let written = install_agy_user_config(
            Path::new(&args.path),
            args.agy_config.as_deref().map(Path::new),
        )?;
        println!("Wrote AGY MCP config to {}", written.display());
        let written = install_agy_hook_settings(
            Path::new(&args.path),
            args.agy_settings.as_deref().map(Path::new),
        )?;
        println!("Wrote AGY hook settings to {}", written.display());
        return Ok(());
    }

    #[cfg(feature = "agy")]
    if args.agy_serve {
        let cfg = load_config(
            args.config.as_deref(),
            &args.path,
            args.strict_config,
            args.max_config_bytes,
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        run_agy_server(
            Path::new(&args.path),
            AgyServerOptions {
                max_bytes: args.max_bytes,
                max_files: args.max_files,
                max_depth: args.max_depth,
                max_total_bytes: args.max_total_bytes,
                ignore_paths: &cfg.ignore_paths,
            },
        )?;
        return Ok(());
    }

    #[cfg(feature = "agy")]
    if args.agy_verify_mcp {
        crate::integrations::mcp_health::verify(
            Path::new(&args.path),
            "--agy-serve",
            args.mcp_timeout_ms,
        )?;
        return Ok(());
    }

    #[cfg(feature = "claude-code")]
    if args.claude_code_serve {
        let cfg = load_config(
            args.config.as_deref(),
            &args.path,
            args.strict_config,
            args.max_config_bytes,
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        run_server(
            Path::new(&args.path),
            ClaudeCodeServerOptions {
                max_bytes: args.max_bytes,
                max_files: args.max_files,
                max_depth: args.max_depth,
                max_total_bytes: args.max_total_bytes,
                ignore_paths: &cfg.ignore_paths,
            },
        )?;
        return Ok(());
    }

    #[cfg(feature = "claude-code")]
    if args.claude_code_verify_mcp {
        crate::integrations::mcp_health::verify(
            Path::new(&args.path),
            "--claude-code-serve",
            args.mcp_timeout_ms,
        )?;
        return Ok(());
    }

    let cfg = load_config(
        args.config.as_deref(),
        &args.path,
        args.strict_config,
        args.max_config_bytes,
    )
    .map_err(|err| anyhow::anyhow!(err))?;
    if args.query.is_some() || args.llm_context.is_some() {
        let cache_settings = CacheSettings {
            root_path: &args.path,
            ner_provider: &args.tier1_ner_provider,
            term_ranker: &args.tier1_term_ranker,
            spacy_model: &args.spacy_model,
            chunk_strategy: &args.chunk_strategy,
            chunk_lines: args.chunk_lines,
            chunk_overlap: args.chunk_overlap,
            chunk_target_tokens: args.chunk_target_tokens,
            chunk_max_tokens: args.chunk_max_tokens,
        };
        let corpus_fingerprint = compute_corpus_fingerprint(
            &args.path,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        );
        let lexical_dir = query_cache_lexical_dir(&cache_settings);
        let mut graph = Graph::build(
            &args.path,
            args.max_bytes,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        )?;
        if !cfg.ignore_paths.is_empty() {
            let ignore = normalize_list(&cfg.ignore_paths);
            graph.pages.retain(|p| {
                let rel = p.rel_path.to_lowercase();
                !ignore.iter().any(|pat| rel.contains(pat))
            });
            let retained: HashSet<String> =
                graph.pages.iter().map(|p| p.rel_path.clone()).collect();
            graph.tier0_records.retain(|r| retained.contains(&r.source));
        }
        let source_docs = graph_to_source_documents(&graph);
        let semantic_relations =
            crate::semantic_relations::SemanticRelationStore::try_from_documents(
                source_docs.iter(),
                crate::semantic_relations::SupersessionOptions::default(),
            )?;
        let document_ids = source_docs
            .iter()
            .map(|doc| doc.doc_id.clone())
            .collect::<Vec<_>>();

        let index = if let Some(cached) =
            load_cached_query_index(&cache_settings, &corpus_fingerprint)
        {
            cached
        } else {
            let built = build_memory_index(
                &graph,
                &args.tier1_ner_provider,
                &args.spacy_model,
                &args.tier1_term_ranker,
                &args.chunk_strategy,
                args.chunk_lines,
                args.chunk_overlap,
                args.chunk_target_tokens,
                args.chunk_max_tokens,
                Some(&lexical_dir),
            )?;
            if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &built)
            {
                eprintln!("warning: unable to persist query cache: {}", err);
            }
            built
        };
        let query_value = args.llm_context.as_deref().or(args.query.as_deref());
        if let Some(query) = query_value {
            let started = Instant::now();
            let prepared = PreparedQuery::new(query);
            let analysis = prepared.analysis().clone();
            if args.llm_context.is_some() {
                let requested = args.result_count.clamp(1, MAX_RESULT_COUNT);
                let candidate_top_k = requested.max(LLM_CONTEXT_CANDIDATE_TOP_K);
                let candidate_results = prepared
                    .execute_on_index_with_semantics(
                        &index,
                        candidate_top_k,
                        None,
                        &semantic_relations,
                        &document_ids,
                    )
                    .0;
                let elapsed_ms = started.elapsed().as_millis();
                let payload = build_llm_context_output(
                    &index,
                    query,
                    &args.path,
                    elapsed_ms,
                    &candidate_results,
                    &args.llm_chunk_strategy,
                    requested,
                );
                if args.simplified {
                    let top_text = payload
                        .top_chunks
                        .iter()
                        .map(|c| c.text.clone())
                        .collect::<Vec<_>>();
                    let mut value = serde_json::to_value(&payload)?;
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("top_chunks".to_string(), serde_json::json!(top_text));
                    }
                    println!("{}", serde_json::to_string_pretty(&value)?);
                } else {
                    println!("{}", serde_json::to_string_pretty(&payload)?);
                }
            } else {
                let results = prepared
                    .execute_on_index_with_semantics(
                        &index,
                        DEFAULT_QUERY_TOP_K,
                        None,
                        &semantic_relations,
                        &document_ids,
                    )
                    .0;
                let elapsed_ms = started.elapsed().as_millis();
                let aggregation =
                    build_aggregate_output(&index, query, &results, DEFAULT_QUERY_TOP_K);
                let payload = QueryOutput {
                    query: query.to_string(),
                    elapsed_ms,
                    result_count: results.len(),
                    analysis,
                    results,
                    aggregation,
                };
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
        }
        return Ok(());
    }

    let mut graph = Graph::build(
        &args.path,
        args.max_bytes,
        args.max_files,
        args.max_depth,
        args.max_total_bytes,
    )?;
    if !cfg.ignore_paths.is_empty() {
        let ignore = normalize_list(&cfg.ignore_paths);
        graph.pages.retain(|p| {
            let rel = p.rel_path.to_lowercase();
            !ignore.iter().any(|pat| rel.contains(pat))
        });
        let retained: HashSet<String> = graph.pages.iter().map(|p| p.rel_path.clone()).collect();
        graph.tier0_records.retain(|r| retained.contains(&r.source));
    }
    if args.show_tier0 {
        println!("{}", serde_json::to_string_pretty(&graph.tier0_records)?);
        return Ok(());
    }
    if args.show_tier1_entities {
        show_tier1_entities(&graph, &args.tier1_ner_provider, &args.spacy_model)?;
        return Ok(());
    }
    if args.show_tier1_terms {
        show_tier1_terms(&graph, &args.tier1_term_ranker)?;
        return Ok(());
    }
    if args.index || args.index_redacted {
        let cache_settings = CacheSettings {
            root_path: &args.path,
            ner_provider: &args.tier1_ner_provider,
            term_ranker: &args.tier1_term_ranker,
            spacy_model: &args.spacy_model,
            chunk_strategy: &args.chunk_strategy,
            chunk_lines: args.chunk_lines,
            chunk_overlap: args.chunk_overlap,
            chunk_target_tokens: args.chunk_target_tokens,
            chunk_max_tokens: args.chunk_max_tokens,
        };
        let corpus_fingerprint = compute_corpus_fingerprint(
            &args.path,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        );
        let lexical_dir = query_cache_lexical_dir(&cache_settings);
        let index = build_memory_index(
            &graph,
            &args.tier1_ner_provider,
            &args.spacy_model,
            &args.tier1_term_ranker,
            &args.chunk_strategy,
            args.chunk_lines,
            args.chunk_overlap,
            args.chunk_target_tokens,
            args.chunk_max_tokens,
            Some(&lexical_dir),
        )?;
        if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &index) {
            eprintln!("warning: unable to persist query cache: {}", err);
        }
        if args.index_redacted {
            println!(
                "{}",
                serde_json::to_string_pretty(&index.redacted_for_export())?
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&index)?);
        }
        return Ok(());
    }
    if let Some(out_path) = args.tier0_index_out.as_deref() {
        let written_path = write_tier0_index(out_path, &graph.tier0_records, &args.path)?;
        println!(
            "Wrote Tier 0 index ({} documents) to {}",
            graph.tier0_records.len(),
            written_path
        );
        return Ok(());
    }

    #[cfg(feature = "codex")]
    if args.codex_serve {
        let cfg = load_config(
            args.config.as_deref(),
            &args.path,
            args.strict_config,
            args.max_config_bytes,
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        run_codex_server(
            Path::new(&args.path),
            CodexServerOptions {
                max_bytes: args.max_bytes,
                max_files: args.max_files,
                max_depth: args.max_depth,
                max_total_bytes: args.max_total_bytes,
                ignore_paths: &cfg.ignore_paths,
            },
        )?;
        return Ok(());
    }

    #[cfg(feature = "codex")]
    if args.codex_verify_mcp {
        crate::integrations::mcp_health::verify(
            Path::new(&args.path),
            "--codex-serve",
            args.mcp_timeout_ms,
        )?;
        return Ok(());
    }

    #[cfg(feature = "muse-code")]
    if args.muse_serve {
        let cfg = load_config(
            args.config.as_deref(),
            &args.path,
            args.strict_config,
            args.max_config_bytes,
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        run_muse_server(
            Path::new(&args.path),
            MuseServerOptions {
                max_bytes: args.max_bytes,
                max_files: args.max_files,
                max_depth: args.max_depth,
                max_total_bytes: args.max_total_bytes,
                ignore_paths: &cfg.ignore_paths,
            },
        )?;
        return Ok(());
    }

    #[cfg(feature = "muse-code")]
    if args.muse_verify_mcp {
        crate::integrations::mcp_health::verify(
            Path::new(&args.path),
            "--muse-serve",
            args.mcp_timeout_ms,
        )?;
        return Ok(());
    }
    if args.show_concepts {
        show_concepts_by_section(&graph, &cfg);
        return Ok(());
    }
    if args.analyze {
        analyze_corpus(&graph, &cfg);
        return Ok(());
    }
    if args.debug_matches {
        let (ac, forms, form_to_concept) = build_matcher(&graph, &cfg);
        let ac = match ac {
            Some(ac) => ac,
            None => return Ok(()),
        };
        for page in &graph.pages {
            println!("{}", page.rel_path);
            let matches = debug_phrase_matches(&page.content, &ac, &forms, &form_to_concept, &cfg);
            for (line, concept, start, end) in matches {
                println!("match [{}..{}]: {} -> {}", start, end, concept, line.trim());
            }
        }
        return Ok(());
    }
    if args.show_headings {
        for page in &graph.pages {
            println!("{}", page.rel_path);
            for heading in &page.headings {
                println!("- {}", heading);
            }
        }
        return Ok(());
    }
    if args.show_chunk_graph_stats {
        println!(
            "{}",
            serde_json::to_string_pretty(&chunk_graph_stats(&graph))?
        );
        return Ok(());
    }
    if let Some(format) = args.export_graph.as_ref() {
        let written = match (&args.graph_level, format) {
            (GraphLevel::Doc, GraphExportFormat::Dot) => export_graph_dot(&graph, &args.graph_out)?,
            (GraphLevel::Doc, GraphExportFormat::Json) => {
                export_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Doc, GraphExportFormat::CytoscapeHtml) => {
                export_graph_cytoscape_html(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::Dot) => {
                export_chunk_graph_dot(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::Json) => {
                export_chunk_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Chunk, GraphExportFormat::CytoscapeHtml) => {
                export_chunk_graph_cytoscape_html(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::Dot) => {
                export_entity_graph_dot(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::Json) => {
                export_entity_graph_json(&graph, &args.graph_out)?
            }
            (GraphLevel::Entity, GraphExportFormat::CytoscapeHtml) => {
                export_entity_graph_cytoscape_html(&graph, &args.graph_out)?
            }
        };
        println!(
            "Wrote graph export ({:?}, level={:?}) with {} pages to {}",
            format,
            args.graph_level,
            graph.pages.len(),
            written
        );
        return Ok(());
    }
    if args.export_ontology {
        let index = build_memory_index(
            &graph,
            &args.tier1_ner_provider,
            &args.spacy_model,
            &args.tier1_term_ranker,
            &args.chunk_strategy,
            args.chunk_lines,
            args.chunk_overlap,
            args.chunk_target_tokens,
            args.chunk_max_tokens,
            None,
        )?;
        let written = export_ontology_json(&index, &args.ontology_out, &args.path)?;
        println!(
            "Wrote ontology export with {} docs to {}",
            index.docs.len(),
            written
        );
        return Ok(());
    }

    // The default invocation and --review share the same project review flow.
    // The graph above remains the source of truth for the existing lint rules;
    // preparing the query index here makes the same corpus immediately
    // available to review consumers without changing the report semantics.
    let cache_settings = CacheSettings {
        root_path: &args.path,
        ner_provider: &args.tier1_ner_provider,
        term_ranker: &args.tier1_term_ranker,
        spacy_model: &args.spacy_model,
        chunk_strategy: &args.chunk_strategy,
        chunk_lines: args.chunk_lines,
        chunk_overlap: args.chunk_overlap,
        chunk_target_tokens: args.chunk_target_tokens,
        chunk_max_tokens: args.chunk_max_tokens,
    };
    let corpus_fingerprint = compute_corpus_fingerprint(
        &args.path,
        args.max_files,
        args.max_depth,
        args.max_total_bytes,
    );
    let lexical_dir = query_cache_lexical_dir(&cache_settings);
    let _review_index = if let Some(cached) =
        load_cached_query_index(&cache_settings, &corpus_fingerprint)
    {
        cached
    } else {
        let built = build_memory_index(
            &graph,
            &args.tier1_ner_provider,
            &args.spacy_model,
            &args.tier1_term_ranker,
            &args.chunk_strategy,
            args.chunk_lines,
            args.chunk_overlap,
            args.chunk_target_tokens,
            args.chunk_max_tokens,
            Some(&lexical_dir),
        )?;
        if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &built) {
            eprintln!("warning: unable to persist review search index: {}", err);
        }
        built
    };
    let mut report = Report::new();

    check_orphans(&graph, &mut report);
    check_cross_refs(&graph, &mut report, &cfg);

    report.print();
    Ok(())
}

fn inspect_index_store(index_path: &Path, view: IndexInspectView) -> Result<()> {
    if !index_path.exists() {
        anyhow::bail!("index path does not exist: {}", index_path.display());
    }
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: crate::segments::SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut store = IndexStore::at_path(index_path, options)?;
    store.refresh()?;
    let payload = match view {
        IndexInspectView::Summary => serde_json::json!({
            "index_path": index_path,
            "index_store": store.inspection(),
        }),
        IndexInspectView::SourceDocuments => serde_json::json!({
            "index_path": index_path,
            "source_documents": store.source_documents(),
        }),
        IndexInspectView::Records => serde_json::json!({
            "index_path": index_path,
            "records": store.records(),
        }),
        IndexInspectView::Segments => serde_json::json!({
            "index_path": index_path,
            "snapshot": store.inspection().snapshot,
        }),
    };
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
