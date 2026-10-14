use crate::lang::Lang;
use crate::pipeline::{ChunkStrategy, Tier1NerProvider, Tier1TermRankerKind};
use clap::{CommandFactory, Parser, ValueEnum};

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
    #[cfg(feature = "muse-code")]
    Muse,
    #[value(name = "openclaw")]
    OpenClaw,
    #[cfg(feature = "hermes")]
    Hermes,
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

#[cfg(feature = "muse-code")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MuseHook {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    Stop,
    SessionEnd,
}

#[cfg(feature = "codex")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CodexHook {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
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

#[cfg(feature = "openclaw")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OpenClawHook {
    /// Internal `agent:bootstrap` — retrieve and inject memory.
    Bootstrap,
    /// Typed `agent_end` — capture a per-turn Outcome.
    AgentEnd,
    /// Typed `before_reset` — capture the authoritative SessionSummary.
    BeforeReset,
    /// Typed `session_start` — session registry bookkeeping.
    SessionStart,
    /// Typed `session_end` — session registry bookkeeping.
    SessionEnd,
    /// Gateway shutdown — bounded pending-capture flush.
    Shutdown,
}

#[cfg(feature = "roo-runtime")]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RooRuntimeHook {
    RunStart,
    AgentTurnStart,
    AgentTurnEnd,
    RunEnd,
    PreToolUse,
    PostToolUse,
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
#[command(
    name = "membrick",
    version = env!("CARGO_PKG_VERSION"),
    after_help = "Quick provider commands:\n  membrick setup <provider> [PATH]\n  membrick serve <provider> [PATH]\n  membrick verify <provider> [PATH]\n\nProviders: claude, codex, gemini, agy, muse, openclaw, hermes, roo-runtime."
)]
/// CLI arguments for the membrick binary.
pub struct Args {
    /// Skip the interactive release update check for this invocation.
    #[arg(long)]
    pub no_update_check: bool,
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
    #[arg(long, default_value = "membrick-graph.dot")]
    pub graph_out: String,
    #[arg(long, value_enum, default_value = "doc")]
    pub graph_level: GraphLevel,
    #[arg(long)]
    pub show_chunk_graph_stats: bool,
    #[arg(long)]
    pub export_ontology: bool,
    #[arg(long, default_value = "membrick-ontology.json")]
    pub ontology_out: String,
    #[arg(long, num_args = 0..=1, default_missing_value = "tier0-index.json")]
    pub tier0_index_out: Option<String>,
    #[arg(long, value_enum, default_value = "spacy")]
    pub tier1_ner_provider: Tier1NerProvider,
    #[arg(long, value_enum, default_value = "yake")]
    pub tier1_term_ranker: Tier1TermRankerKind,
    #[arg(long, default_value = "en_core_web_sm")]
    pub spacy_model: String,
    /// Content language. `auto` (default) detects per text from script
    /// statistics (plus Spanish signals for Latin text); pass
    /// `zh`/`ko`/`es`/`en` to force it. Drives per-language defaults such
    /// as the spaCy model when `--spacy-model` is not given.
    #[arg(long, value_enum, default_value = "auto")]
    pub lang: Lang,
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
    /// Run the lint checks with the project search index prepared.
    #[arg(long)]
    pub review: bool,
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
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_statusline: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_verify_mcp: bool,
    #[arg(long, value_enum)]
    #[cfg(feature = "claude-code")]
    pub claude_code_hook: Option<ClaudeCodeHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_config: Option<String>,
    #[arg(long)]
    #[cfg(feature = "claude-code")]
    pub claude_code_settings: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_statusline: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_verify_mcp: bool,
    #[arg(long, value_enum)]
    #[cfg(feature = "codex")]
    pub codex_hook: Option<CodexHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_config: Option<String>,
    #[arg(long)]
    #[cfg(feature = "codex")]
    pub codex_settings: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "muse-code")]
    pub muse_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "muse-code")]
    pub muse_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "muse-code")]
    pub muse_verify_mcp: bool,
    #[arg(long)]
    #[cfg(feature = "muse-code")]
    pub muse_config: Option<String>,
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "muse-code")]
    pub muse_hook: Option<MuseHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_serve: bool,
    #[arg(long, hide = true)]
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
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_install: bool,
    #[arg(long)]
    #[cfg(feature = "agy")]
    pub agy_force_skill: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_serve: bool,
    #[arg(long, hide = true)]
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
    /// Install the membrick MCP server and memory skill for OpenClaw.
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_install: bool,
    /// Replace an existing user-modified OpenClaw membrick skill.
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_force_skill: bool,
    /// Run the membrick MCP server over stdio for OpenClaw.
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_serve: bool,
    /// Spawn `--openclaw-serve` and verify the MCP handshake.
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_verify_mcp: bool,
    /// Override the OpenClaw config path (default `~/.openclaw/openclaw.json`).
    #[arg(long)]
    #[cfg(feature = "openclaw")]
    pub openclaw_config: Option<String>,
    /// Override the OpenClaw skills directory (default `~/.openclaw/skills`).
    #[arg(long)]
    #[cfg(feature = "openclaw")]
    pub openclaw_skill_dir: Option<String>,
    /// Handle one OpenClaw lifecycle hook event: read the event JSON from
    /// stdin, write the response JSON to stdout. Spawned by the installed
    /// hook wrappers; fail-open on any error.
    #[arg(long, value_enum)]
    #[cfg(feature = "openclaw")]
    pub openclaw_hook: Option<OpenClawHook>,
    /// Override the OpenClaw internal-hooks directory (default
    /// `$OPENCLAW_STATE_DIR/hooks` or `~/.openclaw/hooks`).
    #[arg(long)]
    #[cfg(feature = "openclaw")]
    pub openclaw_hooks_dir: Option<String>,
    /// Override the OpenClaw plugin (extensions) directory (default
    /// `~/.openclaw/extensions`).
    #[arg(long)]
    #[cfg(feature = "openclaw")]
    pub openclaw_plugin_dir: Option<String>,
    /// Install the membrick MCP server and memory skill for Hermes.
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_install: bool,
    /// Replace an existing user-modified Hermes membrick skill.
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_force_skill: bool,
    /// Run the membrick MCP server over stdio for Hermes.
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_serve: bool,
    /// Spawn `--hermes-serve` and verify the MCP handshake.
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_verify_mcp: bool,
    /// Override the Hermes config path (default `~/.hermes/config.yaml`).
    #[arg(long)]
    #[cfg(feature = "hermes")]
    pub hermes_config: Option<String>,
    /// Override the Hermes skills directory (default `~/.hermes/skills`).
    #[arg(long)]
    #[cfg(feature = "hermes")]
    pub hermes_skill_dir: Option<String>,
    /// Install Roo Runtime lifecycle hooks in the user hooks configuration.
    #[arg(long)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_install: bool,
    /// Run the shared workspace memory MCP server over stdio for Roo Runtime.
    #[arg(long)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_serve: bool,
    /// Handle one Roo Runtime hook invocation from stdin.
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_hook: Option<RooRuntimeHook>,
    /// Override Roo Runtime's hooks.json path (default ~/.rooagi/hooks.json).
    #[arg(long)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_config: Option<String>,
    /// Query the unified store (project documents and recorded memories) once
    /// and print chunk-level hits as JSON. Reads no stdin and starts no server.
    #[arg(long)]
    pub recall: Option<String>,
    /// Keep the project recall indexes open and answer newline-delimited JSON requests.
    #[arg(long)]
    pub recall_server: Option<String>,
    /// Internal mode used by the provider clients to start the local resident memory owner.
    #[arg(long, hide = true)]
    pub resident_server: bool,
    #[arg(long)]
    pub inspect_index: Option<String>,
    #[arg(long)]
    pub promote_session: Option<String>,
    #[arg(long)]
    pub replay_session: Option<String>,
    #[arg(long)]
    pub replay_enable_membrick: bool,
    #[arg(long, conflicts_with = "replay_enable_membrick")]
    pub replay_disable_membrick: bool,
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
    Args::parse_from(normalize_provider_command(std::env::args()))
}

fn normalize_provider_command(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut args: Vec<String> = args.into_iter().collect();
    if args.len() < 3 {
        return args;
    }
    let action = match args[1].as_str() {
        "setup" => "install",
        "serve" => "serve",
        "verify" => "verify-mcp",
        _ => return args,
    };
    let provider = args[2].as_str();
    let option = match (provider, action) {
        ("claude", "install") => "--claude-code-install",
        ("claude", "serve") => "--claude-code-serve",
        ("claude", "verify-mcp") => "--claude-code-verify-mcp",
        ("codex", "install") => "--codex-install",
        ("codex", "serve") => "--codex-serve",
        ("codex", "verify-mcp") => "--codex-verify-mcp",
        ("gemini", "install") => "--gemini-cli-install",
        ("gemini", "serve") => "--gemini-cli-serve",
        ("gemini", "verify-mcp") => "--gemini-cli-verify-mcp",
        ("agy", "install") => "--agy-install",
        ("agy", "serve") => "--agy-serve",
        ("agy", "verify-mcp") => "--agy-verify-mcp",
        ("muse", "install") => "--muse-install",
        ("muse", "serve") => "--muse-serve",
        ("muse", "verify-mcp") => "--muse-verify-mcp",
        ("openclaw", "install") => "--openclaw-install",
        ("openclaw", "serve") => "--openclaw-serve",
        ("openclaw", "verify-mcp") => "--openclaw-verify-mcp",
        ("hermes", "install") => "--hermes-install",
        ("hermes", "serve") => "--hermes-serve",
        ("hermes", "verify-mcp") => "--hermes-verify-mcp",
        ("roo-runtime", "install") => "--roo-runtime-install",
        ("roo-runtime", "serve") => "--roo-runtime-serve",
        _ => return args,
    };
    args.drain(1..3);
    args.insert(1, option.to_string());
    args
}

/// Run the CLI: parse args from the environment and execute.
pub fn run() -> anyhow::Result<()> {
    if std::env::args_os().nth(1).is_none() {
        Args::command().print_help()?;
        println!();
        return Ok(());
    }
    let args = parse();
    crate::engine::run(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn parses_recall_query_and_result_count() {
        let args = Args::try_parse_from([
            "membrick",
            "--recall",
            "why did we drop embeddings",
            "--result-count",
            "8",
        ])
        .unwrap();

        assert_eq!(args.recall.as_deref(), Some("why did we drop embeddings"));
        assert_eq!(args.result_count, 8);
    }

    #[test]
    fn parses_index_inspection_view() {
        let args = Args::try_parse_from([
            "membrick",
            "--inspect-index",
            ".membrick/memory",
            "--inspect-view",
            "source-documents",
        ])
        .unwrap();

        assert_eq!(args.inspect_index.as_deref(), Some(".membrick/memory"));
        assert!(matches!(
            args.inspect_view,
            IndexInspectView::SourceDocuments
        ));
    }

    #[test]
    fn parses_review_flag() {
        let args = Args::try_parse_from(["membrick", "--review", "docs"]).unwrap();

        assert!(args.review);
        assert_eq!(args.path, "docs");
    }

    #[cfg(not(feature = "claude-code"))]
    #[test]
    fn default_help_hides_claude_code_feature_flags() {
        let help = Args::command().render_long_help().to_string();
        assert!(!help.contains("--claude-code-install"));
        assert!(!help.contains("--claude-code-verify-mcp"));

        let error = Args::try_parse_from(["membrick", "--claude-code-install"])
            .expect_err("Claude Code install should be unavailable without the feature")
            .to_string();
        assert!(!error.contains("--claude-code-verify-mcp"));
    }

    #[cfg(feature = "claude-code")]
    #[test]
    fn claude_code_help_exposes_feature_flags_when_enabled() {
        let help = Args::command().render_long_help().to_string();
        assert!(help.contains("membrick setup <provider> [PATH]"));
        assert!(help.contains("Providers: claude,"));
        assert!(!help.contains("--claude-code-install"));
    }

    #[cfg(not(feature = "codex"))]
    #[test]
    fn default_help_hides_codex_feature_flags() {
        let help = Args::command().render_long_help().to_string();
        assert!(!help.contains("--codex-install"));
        assert!(!help.contains("--codex-verify-mcp"));

        let error = Args::try_parse_from(["membrick", "--codex-install"])
            .expect_err("Codex install should be unavailable without the feature")
            .to_string();
        assert!(!error.contains("--codex-verify-mcp"));
    }

    #[cfg(feature = "codex")]
    #[test]
    fn codex_help_exposes_feature_flags_when_enabled() {
        let help = Args::command().render_long_help().to_string();
        assert!(help.contains("membrick setup <provider> [PATH]"));
        assert!(help.contains("Providers: claude, codex,"));
        assert!(!help.contains("--codex-install"));
    }

    #[cfg(not(feature = "muse-code"))]
    #[test]
    fn default_help_hides_muse_code_feature_flags() {
        let help = Args::command().render_long_help().to_string();
        assert!(!help.contains("--muse-install"));
        assert!(!help.contains("--muse-verify-mcp"));

        let error = Args::try_parse_from(["membrick", "--muse-install"])
            .expect_err("Muse Code install should be unavailable without the feature")
            .to_string();
        assert!(!error.contains("--muse-verify-mcp"));
    }

    #[cfg(feature = "muse-code")]
    #[test]
    fn muse_code_help_exposes_feature_flags_when_enabled() {
        let help = Args::command().render_long_help().to_string();
        assert!(help.contains("membrick setup <provider> [PATH]"));
        assert!(help.contains("Providers: claude, codex, gemini, agy, muse,"));
        assert!(!help.contains("--muse-install"));
    }
}
