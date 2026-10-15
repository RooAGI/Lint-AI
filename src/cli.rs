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
    after_help = "Project commands:\n  membrick status [PATH]                  Show local index state; does not scan\n  membrick review [PATH]                  Run project lint checks\n  membrick analyze [PATH]                 Analyze project structure\n  membrick index [PATH] [--redacted]     Build or refresh the project index\n  membrick query <TEXT> [PATH] [--count N]\n  membrick context <TEXT> [PATH]          Prepare context for an LLM\n  membrick recall <TEXT> [PATH] [--count N]\n  membrick recall serve [PATH]            Serve recall requests over stdin/stdout\n  membrick inspect-index <PATH> [--view VIEW]\n  membrick session promote <ID> [--provider PROVIDER] [--root PATH]\n  membrick session replay <ID> [--provider PROVIDER] [--root PATH]\n  membrick export graph <FORMAT> [PATH] [--output FILE] [--level LEVEL]\n  membrick export ontology [PATH] [--output FILE]\n\nProvider commands:\n  membrick install <provider> [PATH]\n  membrick serve <provider> [PATH]\n  membrick verify <provider> [PATH]\n  membrick uninstall <provider> [PATH]\n  membrick hook <provider> <event> [PATH]\n  membrick statusline <claude|codex> [PATH]\n\nProvider options: --provider-config, --provider-settings, --provider-skills-dir,\n--provider-hooks-dir, --provider-plugin-dir, --force-skill, --timeout.\nUninstall is supported for: claude, codex. Providers: claude, codex, gemini, agy, muse, openclaw, hermes, roo-runtime."
)]
/// CLI arguments for the membrick binary.
pub struct Args {
    /// Skip the interactive release update check for this invocation.
    #[arg(long)]
    pub no_update_check: bool,
    #[arg(default_value = ".", hide = true)]
    pub path: String,
    #[arg(long, hide = true)]
    pub show_concepts: bool,
    #[arg(long, hide = true)]
    pub show_headings: bool,
    #[arg(long, hide = true)]
    pub show_tier0: bool,
    #[arg(long, hide = true)]
    pub show_tier1_entities: bool,
    #[arg(long, hide = true)]
    pub show_tier1_terms: bool,
    #[arg(long, hide = true)]
    pub index: bool,
    #[arg(long, hide = true)]
    pub index_redacted: bool,
    #[arg(long, hide = true)]
    pub query: Option<String>,
    #[arg(long, hide = true)]
    pub llm_context: Option<String>,
    #[arg(long, default_value_t = 5, hide = true)]
    pub result_count: usize,
    #[arg(long, alias = "simplifed", hide = true)]
    pub simplified: bool,
    #[arg(long, value_enum, default_value = "all", hide = true)]
    pub llm_chunk_strategy: LlmChunkStrategy,
    #[arg(long, value_enum, hide = true)]
    pub export_graph: Option<GraphExportFormat>,
    #[arg(long, default_value = "membrick-graph.dot", hide = true)]
    pub graph_out: String,
    #[arg(long, value_enum, default_value = "doc", hide = true)]
    pub graph_level: GraphLevel,
    #[arg(long, hide = true)]
    pub show_chunk_graph_stats: bool,
    #[arg(long, hide = true)]
    pub export_ontology: bool,
    #[arg(long, default_value = "membrick-ontology.json", hide = true)]
    pub ontology_out: String,
    #[arg(long, num_args = 0..=1, default_missing_value = "tier0-index.json", hide = true)]
    pub tier0_index_out: Option<String>,
    #[arg(long, value_enum, default_value = "spacy", hide = true)]
    pub tier1_ner_provider: Tier1NerProvider,
    #[arg(long, value_enum, default_value = "yake", hide = true)]
    pub tier1_term_ranker: Tier1TermRankerKind,
    #[arg(long, default_value = "en_core_web_sm", hide = true)]
    pub spacy_model: String,
    /// Content language. `auto` (default) detects per text from script
    /// statistics (plus Spanish signals for Latin text); pass
    /// `zh`/`ko`/`es`/`en` to force it. Drives per-language defaults such
    /// as the spaCy model when `--spacy-model` is not given.
    #[arg(long, value_enum, default_value = "auto", hide = true)]
    pub lang: Lang,
    #[arg(long, value_enum, default_value = "heading", hide = true)]
    pub chunk_strategy: ChunkStrategy,
    #[arg(long, default_value_t = 40, hide = true)]
    pub chunk_lines: usize,
    #[arg(long, default_value_t = 10, hide = true)]
    pub chunk_overlap: usize,
    #[arg(long, default_value_t = 450, hide = true)]
    pub chunk_target_tokens: usize,
    #[arg(long, default_value_t = 800, hide = true)]
    pub chunk_max_tokens: usize,
    #[arg(long, hide = true)]
    pub debug_matches: bool,
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long, hide = true)]
    pub analyze: bool,
    /// Run the lint checks with the project search index prepared.
    #[arg(long, hide = true)]
    pub review: bool,
    #[arg(long, default_value_t = 5_000_000, hide = true)]
    pub max_bytes: usize,
    #[arg(long, default_value_t = 50_000, hide = true)]
    pub max_files: usize,
    #[arg(long, default_value_t = 20, hide = true)]
    pub max_depth: usize,
    #[arg(long, hide = true)]
    pub strict_config: bool,
    #[arg(long, hide = true)]
    pub mcp_timeout_ms: Option<u64>,
    /// Show the index state for a project folder without scanning its files.
    #[arg(long, hide = true)]
    pub status: bool,
    /// Internal parser target for rejecting ambiguous bare paths and commands.
    #[arg(long, hide = true)]
    pub unknown_command: Option<String>,
    /// Override the selected provider's MCP configuration file.
    #[arg(
        long,
        hide = true,
        help = "Override the selected provider's MCP configuration file"
    )]
    pub provider_config: Option<String>,
    /// Override the selected provider's hooks or settings file.
    #[arg(
        long,
        hide = true,
        help = "Override the selected provider's hooks or settings file"
    )]
    pub provider_settings: Option<String>,
    /// Override the selected provider's skills directory.
    #[arg(
        long,
        hide = true,
        help = "Override the selected provider's skills directory"
    )]
    pub provider_skills_dir: Option<String>,
    /// Override the selected provider's hooks directory.
    #[arg(
        long,
        hide = true,
        help = "Override the selected provider's hooks directory"
    )]
    pub provider_hooks_dir: Option<String>,
    /// Override the selected provider's plugin directory.
    #[arg(
        long,
        hide = true,
        help = "Override the selected provider's plugin directory"
    )]
    pub provider_plugin_dir: Option<String>,
    /// Replace a user-modified provider memory skill during setup.
    #[arg(
        long,
        hide = true,
        help = "Replace a user-modified provider memory skill during setup"
    )]
    pub force_skill: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_uninstall: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_statusline: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_verify_mcp: bool,
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_hook: Option<ClaudeCodeHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_config: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "claude-code")]
    pub claude_code_settings: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_uninstall: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_statusline: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_verify_mcp: bool,
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_hook: Option<CodexHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "codex")]
    pub codex_config: Option<String>,
    #[arg(long, hide = true)]
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
    #[arg(long, hide = true)]
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
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_hook: Option<GeminiCliHook>,
    #[arg(long)]
    #[cfg(feature = "gemini-cli")]
    #[arg(long, hide = true)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_settings: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "gemini-cli")]
    pub gemini_cli_config: Option<String>,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_install: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_force_skill: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_serve: bool,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_verify_mcp: bool,
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_hook: Option<AgyHook>,
    #[arg(long, hide = true)]
    #[cfg(feature = "agy")]
    pub agy_settings: Option<String>,
    #[arg(long, hide = true)]
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
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_config: Option<String>,
    /// Override the OpenClaw skills directory (default `~/.openclaw/skills`).
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_skill_dir: Option<String>,
    /// Handle one OpenClaw lifecycle hook event: read the event JSON from
    /// stdin, write the response JSON to stdout. Spawned by the installed
    /// hook wrappers; fail-open on any error.
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_hook: Option<OpenClawHook>,
    /// Override the OpenClaw internal-hooks directory (default
    /// `$OPENCLAW_STATE_DIR/hooks` or `~/.openclaw/hooks`).
    #[arg(long, hide = true)]
    #[cfg(feature = "openclaw")]
    pub openclaw_hooks_dir: Option<String>,
    /// Override the OpenClaw plugin (extensions) directory (default
    /// `~/.openclaw/extensions`).
    #[arg(long, hide = true)]
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
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_config: Option<String>,
    /// Override the Hermes skills directory (default `~/.hermes/skills`).
    #[arg(long, hide = true)]
    #[cfg(feature = "hermes")]
    pub hermes_skill_dir: Option<String>,
    /// Install Roo Runtime lifecycle hooks in the user hooks configuration.
    #[arg(long, hide = true)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_install: bool,
    /// Run the shared workspace memory MCP server over stdio for Roo Runtime.
    #[arg(long, hide = true)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_serve: bool,
    /// Handle one Roo Runtime hook invocation from stdin.
    #[arg(long, value_enum, hide = true)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_hook: Option<RooRuntimeHook>,
    /// Override Roo Runtime's hooks.json path (default ~/.rooagi/hooks.json).
    #[arg(long, hide = true)]
    #[cfg(feature = "roo-runtime")]
    pub roo_runtime_config: Option<String>,
    /// Query the unified store (project documents and recorded memories) once
    /// and print chunk-level hits as JSON. Reads no stdin and starts no server.
    #[arg(long, hide = true)]
    pub recall: Option<String>,
    /// Keep the project recall indexes open and answer newline-delimited JSON requests.
    #[arg(long, hide = true)]
    pub recall_server: Option<String>,
    /// Internal mode used by the provider clients to start the local resident memory owner.
    #[arg(long, hide = true)]
    pub resident_server: bool,
    #[arg(long, hide = true)]
    pub inspect_index: Option<String>,
    #[arg(long, hide = true)]
    pub promote_session: Option<String>,
    #[arg(long, hide = true)]
    pub replay_session: Option<String>,
    #[arg(long, hide = true)]
    pub replay_enable_membrick: bool,
    #[arg(long, conflicts_with = "replay_enable_membrick", hide = true)]
    pub replay_disable_membrick: bool,
    #[arg(long, value_enum, default_value = "claude", hide = true)]
    pub session_provider: SessionProvider,
    #[arg(long, hide = true)]
    pub session_root: Option<String>,
    #[arg(long, value_enum, default_value = "summary", hide = true)]
    pub inspect_view: IndexInspectView,
    #[arg(long, default_value_t = 2_000_000, hide = true)]
    pub max_config_bytes: u64,
    #[arg(long, default_value_t = 100_000_000, hide = true)]
    pub max_total_bytes: usize,
}

/// Parse CLI arguments from the environment.
pub fn parse() -> Args {
    Args::parse_from(normalize_provider_command(std::env::args()))
}

fn normalize_provider_command(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut args: Vec<String> = args.into_iter().collect();
    normalize_project_command(&mut args);
    if args.len() >= 2
        && matches!(
            args[1].as_str(),
            "setup" | "install" | "serve" | "verify" | "uninstall" | "hook" | "statusline"
        )
        && args.len() < 3
    {
        args.truncate(1);
        args.push("--unknown-command".to_string());
        args.push("provider is required".to_string());
        return args;
    }
    if args.len() < 3 {
        return args;
    }
    let command = args[1].clone();
    let action = match command.as_str() {
        "setup" | "install" => "install",
        "serve" => "serve",
        "verify" => "verify-mcp",
        "uninstall" => "uninstall",
        "hook" => "hook",
        "statusline" => "statusline",
        _ => return args,
    };
    let provider = args[2].clone();
    let option = match (provider.as_str(), action) {
        ("claude", "install") => "--claude-code-install",
        ("claude", "uninstall") => "--claude-code-uninstall",
        ("claude", "serve") => "--claude-code-serve",
        ("claude", "verify-mcp") => "--claude-code-verify-mcp",
        ("claude", "hook") => "--claude-code-hook",
        ("claude", "statusline") => "--claude-code-statusline",
        ("codex", "install") => "--codex-install",
        ("codex", "uninstall") => "--codex-uninstall",
        ("codex", "serve") => "--codex-serve",
        ("codex", "verify-mcp") => "--codex-verify-mcp",
        ("codex", "hook") => "--codex-hook",
        ("codex", "statusline") => "--codex-statusline",
        ("muse", "install") => "--muse-install",
        ("muse", "serve") => "--muse-serve",
        ("muse", "verify-mcp") => "--muse-verify-mcp",
        ("muse", "hook") => "--muse-hook",
        ("gemini", "install") => "--gemini-cli-install",
        ("gemini", "serve") => "--gemini-cli-serve",
        ("gemini", "verify-mcp") => "--gemini-cli-verify-mcp",
        ("gemini", "hook") => "--gemini-cli-hook",
        ("gemini-cli", "install") => "--gemini-cli-install",
        ("gemini-cli", "serve") => "--gemini-cli-serve",
        ("gemini-cli", "verify-mcp") => "--gemini-cli-verify-mcp",
        ("gemini-cli", "hook") => "--gemini-cli-hook",
        ("agy", "install") => "--agy-install",
        ("agy", "serve") => "--agy-serve",
        ("agy", "verify-mcp") => "--agy-verify-mcp",
        ("agy", "hook") => "--agy-hook",
        ("openclaw", "install") => "--openclaw-install",
        ("openclaw", "serve") => "--openclaw-serve",
        ("openclaw", "verify-mcp") => "--openclaw-verify-mcp",
        ("openclaw", "hook") => "--openclaw-hook",
        ("hermes", "install") => "--hermes-install",
        ("hermes", "serve") => "--hermes-serve",
        ("hermes", "verify-mcp") => "--hermes-verify-mcp",
        ("roo-runtime", "install") => "--roo-runtime-install",
        ("roo-runtime", "serve") => "--roo-runtime-serve",
        ("roo-runtime", "hook") => "--roo-runtime-hook",
        ("rooagi-runtime", "install") => "--roo-runtime-install",
        ("rooagi-runtime", "serve") => "--roo-runtime-serve",
        ("rooagi-runtime", "hook") => "--roo-runtime-hook",
        _ => {
            args.truncate(1);
            args.push("--unknown-command".to_string());
            args.push(format!(
                "unsupported provider command: {command} {provider}"
            ));
            return args;
        }
    };
    args.drain(1..3);
    args.insert(1, option.to_string());

    let replacements = match provider.as_str() {
        "claude" => [
            ("--provider-config", "--claude-code-config"),
            ("--provider-settings", "--claude-code-settings"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--force-skill"),
        ],
        "codex" => [
            ("--provider-config", "--codex-config"),
            ("--provider-settings", "--codex-settings"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--force-skill"),
        ],
        "muse" => [
            ("--provider-config", "--muse-config"),
            ("--provider-settings", "--muse-config"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--force-skill"),
        ],
        "gemini" | "gemini-cli" => [
            ("--provider-config", "--gemini-cli-config"),
            ("--provider-settings", "--gemini-cli-settings"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--force-skill"),
        ],
        "agy" => [
            ("--provider-config", "--agy-config"),
            ("--provider-settings", "--agy-settings"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--agy-force-skill"),
        ],
        "openclaw" => [
            ("--provider-config", "--openclaw-config"),
            ("--provider-settings", "--provider-settings"),
            ("--provider-skills-dir", "--openclaw-skill-dir"),
            ("--provider-hooks-dir", "--openclaw-hooks-dir"),
            ("--provider-plugin-dir", "--openclaw-plugin-dir"),
            ("--force-skill", "--openclaw-force-skill"),
        ],
        "hermes" => [
            ("--provider-config", "--hermes-config"),
            ("--provider-settings", "--provider-settings"),
            ("--provider-skills-dir", "--hermes-skill-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--hermes-force-skill"),
        ],
        "roo-runtime" | "rooagi-runtime" => [
            ("--provider-config", "--roo-runtime-config"),
            ("--provider-settings", "--roo-runtime-config"),
            ("--provider-skills-dir", "--provider-skills-dir"),
            ("--provider-hooks-dir", "--provider-hooks-dir"),
            ("--provider-plugin-dir", "--provider-plugin-dir"),
            ("--force-skill", "--force-skill"),
        ],
        _ => unreachable!(),
    };
    for argument in args.iter_mut() {
        let (flag, value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(flag, value)| {
                (flag, Some(value))
            });
        if let Some((_, replacement)) = replacements
            .iter()
            .find(|(candidate, _)| *candidate == flag)
        {
            *argument = match value {
                Some(value) => format!("{replacement}={value}"),
                None => replacement.to_string(),
            };
        }
    }
    replace_option(&mut args, "--timeout", "--mcp-timeout-ms");
    args
}

fn normalize_project_command(args: &mut Vec<String>) {
    if args.len() < 2 {
        return;
    }
    let command = args[1].clone();
    if command == "help" {
        args[1] = "--help".to_string();
        args.truncate(2);
        return;
    }
    if command == "recall" && args.get(2).is_some_and(|arg| arg == "serve") {
        args.drain(1..3);
        args.insert(1, "--recall-server".to_string());
        return;
    }
    if command == "session" && args.len() >= 3 {
        let operation = args[2].clone();
        let flag = match operation.as_str() {
            "promote" => "--promote-session",
            "replay" => "--replay-session",
            _ => "",
        };
        if !flag.is_empty() {
            args.drain(1..3);
            args.insert(1, flag.to_string());
            replace_option(args, "--provider", "--session-provider");
            replace_option(args, "--root", "--session-root");
            replace_switch(args, "--enable-membrick", "--replay-enable-membrick");
            replace_switch(args, "--disable-membrick", "--replay-disable-membrick");
            return;
        }
    }
    let replacement = match command.as_str() {
        "status" | "info" => Some("--status"),
        "review" => Some("--review"),
        "analyze" => Some("--analyze"),
        "index" => Some("--index"),
        "query" => Some("--query"),
        "context" => Some("--llm-context"),
        "recall" => Some("--recall"),
        "inspect-index" => Some("--inspect-index"),
        _ => None,
    };
    if let Some(flag) = replacement {
        args.remove(1);
        args.insert(1, flag.to_string());
        if matches!(command.as_str(), "query" | "context" | "recall") {
            replace_option(args, "--count", "--result-count");
        }
        if command == "inspect-index" {
            replace_option(args, "--view", "--inspect-view");
        }
        if command == "index" {
            replace_switch(args, "--redacted", "--index-redacted");
        }
        return;
    }

    if command == "export" && args.len() >= 3 {
        let export_kind = args[2].clone();
        match export_kind.as_str() {
            "graph" => {
                args.drain(1..3);
                args.insert(1, "--export-graph".to_string());
                replace_option(args, "--output", "--graph-out");
                replace_option(args, "--level", "--graph-level");
                return;
            }
            "ontology" => {
                args.drain(1..3);
                args.insert(1, "--export-ontology".to_string());
                replace_option(args, "--output", "--ontology-out");
                return;
            }
            _ => {}
        }
    }
    replace_option(args, "--view", "--inspect-view");

    // A path alone used to start a project scan. Require an explicit action.
    if !command.starts_with('-')
        && !matches!(
            command.as_str(),
            "setup" | "install" | "serve" | "verify" | "uninstall" | "hook" | "statusline"
        )
    {
        args.truncate(1);
        args.push("--unknown-command".to_string());
        args.push(command);
    }
}

fn replace_option(args: &mut [String], from: &str, to: &str) {
    for arg in args {
        if arg == from {
            *arg = to.to_string();
        } else if let Some(value) = arg.strip_prefix(&format!("{from}=")) {
            *arg = format!("{to}={value}");
        }
    }
}

fn replace_switch(args: &mut [String], from: &str, to: &str) {
    for arg in args {
        if arg == from {
            *arg = to.to_string();
        }
    }
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

    fn parse_public_args(args: &[&str]) -> Result<Args, clap::Error> {
        let mut argv = vec!["membrick".to_string()];
        argv.extend(args.iter().map(|arg| (*arg).to_string()));
        Args::try_parse_from(normalize_provider_command(argv))
    }

    #[test]
    fn public_project_command_matrix_parses_without_running_commands() {
        let cases: &[&[&str]] = &[
            &[
                "status",
                ".",
                "--config",
                "membrick.toml",
                "--no-update-check",
            ],
            &["info", "."],
            &["review", "."],
            &["analyze", "."],
            &["index", ".", "--redacted"],
            &["query", "routing decision", ".", "--count", "3"],
            &["context", "routing decision", "."],
            &["recall", "routing decision", ".", "--count", "3"],
            &["recall", "serve", "."],
            &["inspect-index", ".membrick/index", "--view", "segments"],
            &[
                "session",
                "promote",
                "session-1",
                "--provider",
                "codex",
                "--root",
                ".",
            ],
            &[
                "session",
                "replay",
                "session-1",
                "--provider",
                "claude",
                "--root",
                ".",
                "--enable-membrick",
            ],
            &[
                "export",
                "graph",
                "json",
                ".",
                "--output",
                "graph.json",
                "--level",
                "chunk",
            ],
            &["export", "ontology", ".", "--output", "ontology.json"],
        ];

        for args in cases {
            parse_public_args(args)
                .unwrap_or_else(|error| panic!("public command {args:?} failed to parse: {error}"));
        }
    }

    #[test]
    fn bare_project_path_is_rejected_instead_of_starting_a_scan() {
        let args = parse_public_args(&["."]).expect("bare paths are parsed as an explicit error");
        assert_eq!(args.unknown_command.as_deref(), Some("."));
    }

    #[test]
    fn public_provider_commands_and_shared_options_parse_for_enabled_providers() {
        let mut providers = vec![
            "claude",
            "codex",
            "gemini",
            "gemini-cli",
            "agy",
            "muse",
            "openclaw",
            "hermes",
            "roo-runtime",
        ];
        providers.retain(|provider| match *provider {
            "claude" => cfg!(feature = "claude-code"),
            "codex" => cfg!(feature = "codex"),
            "gemini" | "gemini-cli" => cfg!(feature = "gemini-cli"),
            "agy" => cfg!(feature = "agy"),
            "muse" => cfg!(feature = "muse-code"),
            "openclaw" => cfg!(feature = "openclaw"),
            "hermes" => cfg!(feature = "hermes"),
            "roo-runtime" => cfg!(feature = "roo-runtime"),
            _ => false,
        });

        for provider in providers {
            if provider == "claude" {
                parse_public_args(&["setup", provider, "."])
                    .expect("setup remains an alias for install");
            }

            for action in ["install", "serve", "verify"] {
                let args = [
                    action,
                    provider,
                    ".",
                    "--provider-config",
                    "provider-config.toml",
                    "--provider-skills-dir",
                    "skills",
                    "--provider-hooks-dir",
                    "hooks",
                    "--provider-plugin-dir",
                    "plugins",
                    "--timeout",
                    "900",
                ];
                let install_args = [
                    "install",
                    provider,
                    ".",
                    "--provider-config",
                    "provider-config.toml",
                    "--provider-skills-dir",
                    "skills",
                    "--provider-hooks-dir",
                    "hooks",
                    "--provider-plugin-dir",
                    "plugins",
                    "--force-skill",
                    "--timeout",
                    "900",
                ];
                if action == "install" {
                    parse_public_args(&install_args).unwrap_or_else(|error| {
                        panic!("public command {install_args:?} failed to parse: {error}")
                    });
                } else {
                    parse_public_args(&args).unwrap_or_else(|error| {
                        panic!("public command {args:?} failed to parse: {error}")
                    });
                }

                let settings_args = [
                    action,
                    provider,
                    ".",
                    "--provider-settings",
                    "provider-settings.json",
                ];
                parse_public_args(&settings_args).unwrap_or_else(|error| {
                    panic!("public command {settings_args:?} failed to parse: {error}")
                });
            }

            if matches!(provider, "claude" | "codex") {
                for action in ["uninstall", "statusline"] {
                    let args = [action, provider, "."];
                    parse_public_args(&args).unwrap_or_else(|error| {
                        panic!("public command {args:?} failed to parse: {error}")
                    });
                }
            }

            let events: &[&str] = match provider {
                "claude" => &[
                    "session-start",
                    "user-prompt-submit",
                    "user-prompt-expansion",
                    "pre-tool-use",
                    "post-tool-use",
                    "pre-compact",
                    "stop",
                    "session-end",
                    "subagent-start",
                    "subagent-stop",
                ],
                "codex" => &[
                    "session-start",
                    "user-prompt-submit",
                    "pre-tool-use",
                    "post-tool-use",
                    "user-prompt-expansion",
                    "pre-compact",
                    "post-compact",
                    "stop",
                    "session-end",
                    "subagent-start",
                    "subagent-stop",
                ],
                "gemini" | "gemini-cli" => &[
                    "session-start",
                    "before-agent",
                    "after-agent",
                    "before-model",
                    "before-tool-selection",
                    "before-tool",
                    "after-tool",
                    "pre-compress",
                    "session-end",
                ],
                "muse" => &[
                    "session-start",
                    "user-prompt-submit",
                    "pre-tool-use",
                    "post-tool-use",
                    "post-tool-use-failure",
                    "stop",
                    "session-end",
                ],
                "agy" => &[
                    "pre-tool-use",
                    "post-tool-use",
                    "pre-invocation",
                    "post-invocation",
                    "stop",
                ],
                "openclaw" => &[
                    "bootstrap",
                    "agent-end",
                    "before-reset",
                    "session-start",
                    "session-end",
                    "shutdown",
                ],
                "hermes" => &[],
                "roo-runtime" => &[
                    "run-start",
                    "agent-turn-start",
                    "agent-turn-end",
                    "run-end",
                    "pre-tool-use",
                    "post-tool-use",
                ],
                _ => unreachable!(),
            };
            for event in events {
                let args = ["hook", provider, event, "."];
                parse_public_args(&args).unwrap_or_else(|error| {
                    panic!("public command {args:?} failed to parse: {error}")
                });
            }
        }
    }

    #[test]
    fn public_cli_reference_lists_the_command_matrix() {
        let reference = include_str!("../docs/cli.md");
        for command in [
            "membrick status [PATH]",
            "membrick review [PATH]",
            "membrick index [PATH]",
            "membrick query <TEXT>",
            "membrick recall serve [PATH]",
            "membrick install <provider> [PATH]",
            "membrick verify <provider> [PATH]",
            "membrick hook <provider> <event> [PATH]",
            "membrick session promote <ID>",
            "membrick session replay <ID>",
        ] {
            assert!(
                reference.contains(command),
                "CLI reference is missing {command}"
            );
        }
    }

    #[test]
    fn detailed_help_keeps_legacy_and_tuning_flags_hidden() {
        let help = Args::command().render_long_help().to_string();
        for hidden in [
            "--claude-code-install",
            "--codex-install",
            "--max-bytes",
            "--max-files",
            "--chunk-max-tokens",
            "--tier1-ner-provider",
        ] {
            assert!(!help.contains(hidden), "help unexpectedly exposes {hidden}");
        }
    }

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
        assert!(help.contains("membrick install <provider> [PATH]"));
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
        assert!(help.contains("membrick install <provider> [PATH]"));
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
        assert!(help.contains("membrick install <provider> [PATH]"));
        assert!(help.contains("Providers: claude, codex, gemini, agy, muse,"));
        assert!(!help.contains("--muse-install"));
    }
}
