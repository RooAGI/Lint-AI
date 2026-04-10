use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
pub enum Tier1NerProvider {
    Heuristic,
    Spacy,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Tier1TermRankerKind {
    Yake,
    Rake,
    Cvalue,
    Textrank,
}

#[derive(Parser, Debug)]
#[command(name = "lint-ai")]
/// CLI arguments for the lint-ai binary.
pub struct Args {
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
    #[arg(long, num_args = 0..=1, default_missing_value = "tier0-index.json")]
    pub tier0_index_out: Option<String>,
    #[arg(long, value_enum, default_value = "heuristic")]
    pub tier1_ner_provider: Tier1NerProvider,
    #[arg(long, value_enum, default_value = "yake")]
    pub tier1_term_ranker: Tier1TermRankerKind,
    #[arg(long, default_value = "en_core_web_sm")]
    pub spacy_model: String,
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
    #[arg(long, default_value_t = 2_000_000)]
    pub max_config_bytes: u64,
    #[arg(long, default_value_t = 100_000_000)]
    pub max_total_bytes: usize,
}

/// Parse CLI arguments from the environment.
pub fn parse() -> Args {
    Args::parse()
}
