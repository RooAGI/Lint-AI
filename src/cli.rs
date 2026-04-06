use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "lint-ai")]
pub struct Args {
    pub path: String,
    #[arg(long)]
    pub show_concepts: bool,
    #[arg(long)]
    pub show_headings: bool,
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

pub fn parse() -> Args {
    Args::parse()
}
