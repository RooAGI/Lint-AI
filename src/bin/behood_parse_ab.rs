//! A/B harness: behood verdicts per question under each parse backend.
//!
//! Runs every question through `analyze_query_semantics` (parse -> bekind
//! judge) with the process-wide parse provider set from `--provider`, and
//! writes one JSON object per line: the question id, the scope verdicts,
//! and the judged (text, kind) entities.
//!
//! Run once per backend, then diff the two outputs:
//!
//! ```sh
//! BEHOOD_BIN=$HOME/workspace/bekind/target/release/bekind \
//!   cargo run --release --bin behood_parse_ab -- \
//!   --provider spacy --out /tmp/ab_spacy.jsonl
//! BEHOOD_BIN=$HOME/workspace/bekind/target/release/bekind \
//!   cargo run --release --bin behood_parse_ab -- \
//!   --provider heuristic --out /tmp/ab_heuristic.jsonl
//! ```
//!
//! The spacy run exercises `scripts/behood_query.py --serve` (Python+spaCy);
//! the heuristic run must not spawn any Python process.

use clap::{Parser, ValueEnum};
use lint_ai::behood_query::{
    analyze_query_semantics, set_behood_parse_provider, BehoodParseProvider, BehoodQueryDaemon,
    BekindDaemon,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Clone, Copy, ValueEnum)]
enum ProviderArg {
    Spacy,
    Heuristic,
}

#[derive(Parser)]
struct Args {
    /// Parse backend under test.
    #[arg(long, value_enum)]
    provider: ProviderArg,
    /// Where to write the JSONL verdicts.
    #[arg(long)]
    out: PathBuf,
    /// LongMemEval-S cleaned questions file.
    #[arg(long, default_value = "/home/hatch/workspace/lint-ai-benchmark-logs/longmemeval_s_cleaned.json")]
    questions: PathBuf,
    /// Only run the first N questions (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let provider = match args.provider {
        ProviderArg::Spacy => BehoodParseProvider::Spacy,
        ProviderArg::Heuristic => BehoodParseProvider::Heuristic,
    };
    set_behood_parse_provider(provider);
    // Warm both daemons up front so per-question timings are steady-state.
    // The parse daemon is only warmed when it will actually be used: in
    // heuristic mode no Python process may start (this is the assertion
    // under test).
    if matches!(provider, BehoodParseProvider::Spacy) {
        BehoodQueryDaemon::global().prewarm();
    }
    BekindDaemon::global().prewarm();

    let raw = std::fs::read_to_string(&args.questions)?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&raw)?;
    let total = if args.limit > 0 {
        args.limit.min(entries.len())
    } else {
        entries.len()
    };

    let out = File::create(&args.out)?;
    let mut w = BufWriter::new(out);
    for (i, entry) in entries.iter().take(total).enumerate() {
        let id = entry
            .get("question_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let question = entry
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let (scope, entities) = analyze_query_semantics(question);
        let record = serde_json::json!({
            "id": id,
            "scope": scope.iter().map(|s| serde_json::json!({
                "activity_phrase": s.activity_phrase,
                "temporal_words": s.temporal_words,
                "habitual": s.habitual,
            })).collect::<Vec<_>>(),
            "entities": entities.iter().map(|e| serde_json::json!({
                "text": e.text,
                "kind": e.kind,
            })).collect::<Vec<_>>(),
        });
        writeln!(w, "{}", serde_json::to_string(&record)?)?;
        if (i + 1) % 50 == 0 {
            eprintln!("{}/{} questions", i + 1, total);
        }
    }
    w.flush()?;
    eprintln!("wrote {} records to {}", total, args.out.display());
    Ok(())
}
