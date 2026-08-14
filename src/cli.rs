use crate::pipeline::{ChunkStrategy, Tier1NerProvider, Tier1TermRankerKind};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
pub enum LlmChunkStrategy {
    All,
    ByDoc,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum GraphExportFormat {
    Dot,
    Json,
    CytoscapeHtml,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum GraphLevel {
    Doc,
    Chunk,
    Entity,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SessionProvider {
    Claude,
    Codex,
    Gemini,
    Agy,
}

#[cfg(feature = "claude-code")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ClaudeCodeHook {
    SessionStart,
    UserPromptSubmit,
    UserPromptExpansion,
    PreToolUse,
    PostToolUse,
    PreCompact,
    Stop,
    SessionEnd,
    SubagentStart,
    SubagentStop,
}

#[cfg(feature = "codex")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CodexHook {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PermissionRequest,
    PostToolUse,
    UserPromptExpansion,
    PreCompact,
    PostCompact,
    Stop,
    SessionEnd,
    SubagentStart,
    SubagentStop,
}

#[cfg(feature = "gemini-cli")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum GeminiCliHook {
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

#[cfg(feature = "agy")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AgyHook {
    PreToolUse,
    PostToolUse,
    PreInvocation,
    PostInvocation,
    Stop,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum IndexInspectView {
    Summary,
    SourceDocuments,
    Records,
    Segments,
}

#[derive(Parser, Debug)]
#[command(name = "lint-ai")]
/// CLI arguments for the lint-ai binary.
pub struct Args {
    #[arg(default_value = ".")]
    pub path: String,
    #[arg(long)]
    pub show_concepts: bool,
    #[arg(long)]
    pub show_headings: bool,
    #[arg(long)]
    pub show_tier0: bool,
    #[arg(long)]
    pub show_tier1_entities: bool,
    #[arg(long)]
    pub show_tier1_terms: bool,
    #[arg(long)]
    pub index: bool,
    #[arg(long)]
    pub index_redacted: bool,
    #[arg(long)]
    pub query: Option<String>,
    #[arg(long)]
    pub llm_context: Option<String>,
    #[arg(long, default_value_t = 5)]
    pub result_count: usize,
    #[arg(long, alias = "simplifed")]
    pub simplified: bool,
    #[arg(long, value_enum, default_value = "all")]
    pub llm_chunk_strategy: LlmChunkStrategy,
    #[arg(long, value_enum)]
    pub export_graph: Option<GraphExportFormat>,
    #[arg(long, default_value = "lint-ai-graph.dot")]
    pub graph_out: String,
    #[arg(long, value_enum, default_value = "doc")]
    pub graph_level: GraphLevel,
    #[arg(long)]
    pub show_chunk_graph_stats: bool,
    #[arg(long)]
    pub export_ontology: bool,
    #[arg(long, default_value = "lint-ai-ontology.json")]
    pub ontology_out: String,
    #[arg(long, num_args = 0..=1, default_missing_value = "tier0-index.json")]
    pub tier0_index_out: Option<String>,
    #[arg(long, value_enum, default_value = "heuristic")]
    pub tier1_ner_provider: Tier1NerProvider,
    #[arg(long, value_enum, default_value = "yake")]
    pub tier1_term_ranker: Tier1TermRankerKind,
    #[arg(long, default_value = "en_core_web_sm")]
    pub spacy_model: String,
    #[arg(long, value_enum, default_value = "heading")]
    pub chunk_strategy: ChunkStrategy,
    #[arg(long, default_value_t = 40)]
    pub chunk_lines: usize,
    #[arg(long, default_value_t = 10)]
    pub chunk_overlap: usize,
    #[arg(long, default_value_t = 450)]
    pub chunk_target_tokens: usize,
    #[arg(long, default_value_t = 800)]
    pub chunk_max_tokens: usize,
    #[arg(long)]
    pub debug_matches: bool,
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub analyze: bool,
    #[arg(long, default_value_t = 5_000_000)]
    pub max_bytes: usize,
    #[arg(long, default_value_t = 50_000)]
    pub max_files: usize,
    #[arg(long, default_value_t = 20)]
    pub max_depth: usize,
    #[arg(long)]
    pub strict_config: bool,
    #[arg(long)]
    pub mcp_timeout_ms: Option<u64>,
    #[arg(long)]
    #[cfg(feature = "claude-code")]
    pub claude_code_install: bool,
    #[arg(long)]
    #[cfg(feature = "claude-code")]
    pub claude_code_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_statusline: bool,
    #[arg(long)]
    pub claude_code_verify_mcp: bool,
    #[arg(long, value_enum)]
    #[cfg(feature = "claude-code")]
    pub claude_code_hook: Option<ClaudeCodeHook>,
    #[arg(long)]
    #[cfg(feature = "claude-code")]
    pub claude_code_config: Option<String>,
    #[arg(long)]
    #[cfg(feature = "claude-code")]
    pub claude_code_settings: Option<String>,
    #[arg(long)]
    #[cfg(feature = "codex")]
    pub codex_install: bool,
    #[arg(long)]
    #[cfg(feature = "codex")]
    pub codex_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_statusline: bool,
    #[arg(long)]
    pub codex_verify_mcp: bool,
    #[arg(long, value_enum)]
    #[cfg(feature = "codex")]
    pub codex_hook: Option<CodexHook>,
    #[arg(long)]
    #[cfg(feature = "codex")]
    pub codex_config: Option<String>,
    #[arg(long)]
    #[cfg(feature = "codex")]
    pub codex_settings: Option<String>,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_install: bool,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_serve: bool,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_verify_mcp: bool,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    #[arg(long, value_enum)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_hook: Option<GeminiCliHook>,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_settings: Option<String>,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_config: Option<String>,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_install: bool,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_serve: bool,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_verify_mcp: bool,
    #[arg(long, value_enum)]
    #[cfg(feature = "agy")]
    pub agy_hook: Option<AgyHook>,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_settings: Option<String>,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_config: Option<String>,
    #[arg(long)]
    pub inspect_index: Option<String>,
    #[arg(long)]
    pub promote_session: Option<String>,
    #[arg(long)]
    pub replay_session: Option<String>,
    #[arg(long)]
    pub replay_enable_lint_ai: bool,
    #[arg(long, conflicts_with = "replay_enable_lint_ai")]
    pub replay_disable_lint_ai: bool,
    #[arg(long, value_enum, default_value = "claude")]
    pub session_provider: SessionProvider,
    #[arg(long)]
    pub session_root: Option<String>,
    #[arg(long, value_enum, default_value = "summary")]
    pub inspect_view: IndexInspectView,
    #[arg(long, default_value_t = 2_000_000)]
    pub max_config_bytes: u64,
    #[arg(long, default_value_t = 100_000_000)]
    pub max_total_bytes: usize,
}

/// Parse CLI arguments from the environment.
pub fn parse() -> Args {
    Args::parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_index_inspection_view() {
        let args = Args::try_parse_from([
            "lint-ai",
            "--inspect-index",
            ".lint-ai/claude-memory",
            "--inspect-view",
            "source-documents",
        ])
        .unwrap();

        assert_eq!(
            args.inspect_index.as_deref(),
            Some(".lint-ai/claude-memory")
        );
        assert!(matches!(
            args.inspect_view,
            IndexInspectView::SourceDocuments
        ));
    }
}
