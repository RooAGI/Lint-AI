//! Prototype: case-based classifier over lint-ai retrieval.
//!
//! Treats each labeled JevBench decision as a memory. For a new (unlabeled)
//! decision, retrieves the most similar labeled decisions via
//! `MemoryService::search`, then lets them vote: similarity-weighted votes
//! become the probability vector over the options.
//!
//! Voting is only defined when the retrieved bank items share the test item's
//! exact option set; otherwise the item abstains (counted in coverage).
//! Local experiment only — nothing is submitted anywhere.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use serde::Deserialize;

use lint_ai::default_production_pipeline_options;
use lint_ai::memory_api::{AddRequest, MemoryService, Message, SearchRequest};

#[derive(Debug, Parser)]
struct Args {
    /// Comma-separated paths to bank JSONL files (labeled decisions).
    #[arg(long)]
    bank: String,
    /// Path to test JSONL file (labels used only for scoring).
    #[arg(long)]
    test: PathBuf,
    /// User id scope for the memory service.
    #[arg(long, default_value = "jevbench")]
    user: String,
    /// Retrieval depth per query before label-set filtering.
    #[arg(long, default_value_t = 30)]
    retrieve_k: usize,
    /// Max voting neighbors sharing the test item's label set.
    #[arg(long, default_value_t = 10)]
    vote_k: usize,
    /// Minimum same-label-set neighbors required; fewer => abstain.
    #[arg(long, default_value_t = 3)]
    min_votes: usize,
    /// Where to write the per-item JSONL report.
    #[arg(long, default_value = "")]
    out: String,
    /// Ablation: omit the Options line from the query text.
    #[arg(long, default_value_t = false)]
    no_options: bool,
}

#[derive(Debug, Deserialize)]
struct RawDecision {
    id: String,
    family: String,
    state: serde_json::Value,
    question: RawQuestion,
    labels: Vec<String>,
    expected: serde_json::Value,
}

fn value_to_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    qtype: String,
    instructions: String,
    #[serde(default)]
    criteria: serde_json::Value,
}

struct Decision {
    id: String,
    family: String,
    qtype: String,
    labels: Vec<String>,
    expected: String,
    state: String,
    rubric: String,
}

impl Decision {
    fn from_raw(r: RawDecision) -> Self {
        let mut rubric = r.question.instructions.clone();
        match &r.question.criteria {
            serde_json::Value::Object(map) => {
                let mut crit: Vec<(&String, &serde_json::Value)> = map.iter().collect();
                crit.sort_by_key(|(k, _)| *k);
                for (k, v) in crit {
                    let vs = v.as_str().unwrap_or("");
                    rubric.push_str(&format!(" [{k}: {vs}]"));
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    let vs = v.as_str().unwrap_or("");
                    rubric.push_str(&format!(" [option{i}: {vs}]"));
                }
            }
            _ => {}
        }
        Decision {
            id: r.id,
            family: r.family,
            qtype: r.question.qtype,
            labels: r.labels,
            expected: value_to_text(&r.expected),
            state: value_to_text(&r.state),
            rubric,
        }
    }

    fn bank_text(&self) -> String {
        format!(
            "Past decision (labeled example).\nSituation: {}\nRubric: {}\nOptions: {}\nDecided: {}",
            self.state,
            self.rubric,
            self.labels.join(", "),
            self.expected
        )
    }

    fn query_text(&self, include_options: bool) -> String {
        let options = if include_options {
            format!("\nOptions: {}", self.labels.join(", "))
        } else {
            String::new()
        };
        format!(
            "Situation: {}\nRubric: {}{}",
            self.state, self.rubric, options
        )
    }
}

fn load_jsonl(path: &str) -> anyhow::Result<Vec<Decision>> {
    let data = fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;
    data.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<RawDecision>(l)
                .map(Decision::from_raw)
                .with_context(|| format!("bad line in {path}"))
        })
        .collect()
}

fn softmax(scores: &[f32]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = scores.iter().map(|s| ((s - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-12)).collect()
}

#[derive(Debug, Default)]
struct Agg {
    n: usize,
    covered: usize,
    correct: usize,
    brier_sum: f64,
    // ece bins over confidence
    bin_n: [usize; 10],
    bin_acc: [usize; 10],
    bin_conf: [f64; 10],
}

impl Agg {
    fn record(&mut self, predicted: Option<(&str, f64, &BTreeMap<String, f64>)>, expected: &str) {
        self.n += 1;
        let Some((pred_label, conf, probs)) = predicted else {
            return;
        };
        self.covered += 1;
        if pred_label == expected {
            self.correct += 1;
        }
        let mut brier = 0.0;
        for (label, p) in probs {
            let y = if label == expected { 1.0 } else { 0.0 };
            brier += (p - y).powi(2);
        }
        self.brier_sum += brier;
        let b = (conf * 10.0).floor() as usize;
        let b = b.min(9);
        self.bin_n[b] += 1;
        self.bin_conf[b] += conf;
        if pred_label == expected {
            self.bin_acc[b] += 1;
        }
    }

    fn ece(&self) -> f64 {
        let mut ece = 0.0;
        for b in 0..10 {
            if self.bin_n[b] == 0 {
                continue;
            }
            let acc = self.bin_acc[b] as f64 / self.bin_n[b] as f64;
            let conf = self.bin_conf[b] / self.bin_n[b] as f64;
            ece += (self.bin_n[b] as f64 / self.covered.max(1) as f64) * (acc - conf).abs();
        }
        ece
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut bank: Vec<Decision> = Vec::new();
    for p in args.bank.split(',') {
        let mut v = load_jsonl(p.trim())?;
        eprintln!("bank {p}: {} items", v.len());
        bank.append(&mut v);
    }
    let test = load_jsonl(args.test.to_str().unwrap())?;
    eprintln!("test {}: {} items", args.test.display(), test.len());

    let bank_labels: HashMap<String, &Decision> = bank.iter().map(|d| (d.id.clone(), d)).collect();

    let mut service = MemoryService::in_memory(default_production_pipeline_options());
    let requests: Vec<AddRequest> = bank
        .iter()
        .map(|d| AddRequest {
            request_id: format!("req-{}", d.id),
            messages: vec![Message {
                role: "user".to_string(),
                timestamp: None,
                content: d.bank_text(),
                expires_at_ms: None,
                supersedes_id: None,
            }],
            user_id: args.user.clone(),
            session_id: d.id.clone(),
        })
        .collect();
    let responses = service.add_batch(requests)?;
    eprintln!("indexed {} bank decisions", responses.len());

    let mut overall = Agg::default();
    let mut by_type: HashMap<String, Agg> = HashMap::new();
    let mut out_lines: Vec<String> = Vec::new();
    let mut trace_shown = 0;

    for t in &test {
        let resp = service.search(SearchRequest {
            query: t.query_text(!args.no_options),
            options: None,
            user_id: args.user.clone(),
            top_k: args.retrieve_k,
            session_id: None,
            scope: None,
            filters: None,

            lang: None,
        })?;

        // Dedup hits by session, keep max score, map to bank decisions.
        let mut best: HashMap<String, f32> = HashMap::new();
        for hit in &resp.data {
            if let Some(sid) = hit.session_id.clone() {
                if bank_labels.contains_key(&sid) {
                    let e = best.entry(sid).or_insert(f32::NEG_INFINITY);
                    if hit.score > *e {
                        *e = hit.score;
                    }
                }
            }
        }
        let mut cands: Vec<(&Decision, f32)> = best
            .iter()
            .filter_map(|(sid, s)| bank_labels.get(sid).map(|d| (*d, *s)))
            .filter(|(d, _)| d.labels == t.labels)
            .collect();
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        cands.truncate(args.vote_k);

        let predicted: Option<(String, f64, BTreeMap<String, f64>)> =
            if cands.len() < args.min_votes {
                None
            } else {
                let scores: Vec<f32> = cands.iter().map(|(_, s)| *s).collect();
                let weights = softmax(&scores);
                let mut probs: BTreeMap<String, f64> = BTreeMap::new();
                for label in &t.labels {
                    probs.insert(label.clone(), 0.0);
                }
                for ((d, _), w) in cands.iter().zip(weights.iter()) {
                    *probs.get_mut(&d.expected).unwrap() += *w;
                }
                let (pred_label, conf) = probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(l, p)| (l.clone(), *p))
                    .unwrap();
                // Show a couple of traces so the run is inspectable.
                if trace_shown < 2 {
                    trace_shown += 1;
                    eprintln!("--- trace {} (expected {})", t.id, t.expected);
                    for ((d, s), w) in cands.iter().zip(weights.iter()) {
                        eprintln!(
                            "    retr {} score={:.3} w={:.3} label={}",
                            d.id, s, w, d.expected
                        );
                    }
                    eprintln!("    predicted={} conf={:.3}", pred_label, conf);
                }
                Some((pred_label, conf, probs))
            };

        overall.record(
            predicted.as_ref().map(|(l, c, p)| (l.as_str(), *c, p)),
            &t.expected,
        );
        by_type.entry(t.qtype.clone()).or_default().record(
            predicted.as_ref().map(|(l, c, p)| (l.as_str(), *c, p)),
            &t.expected,
        );

        let (pred_json, conf_json, probs_json) = match &predicted {
            Some((l, c, p)) => (
                serde_json::json!(l),
                serde_json::json!(c),
                serde_json::json!(p),
            ),
            None => (
                serde_json::json!(null),
                serde_json::json!(null),
                serde_json::json!(null),
            ),
        };
        out_lines.push(
            serde_json::json!({
                "id": t.id,
                "family": t.family,
                "qtype": t.qtype,
                "expected": t.expected,
                "predicted": pred_json,
                "confidence": conf_json,
                "probs": probs_json,
            })
            .to_string(),
        );
    }

    let report = serde_json::json!({
        "bank_items": bank.len(),
        "test_items": test.len(),
        "vote_k": args.vote_k,
        "min_votes": args.min_votes,
        "overall": summarize(&overall),
        "by_qtype": by_type.iter().map(|(k, v)| (k, summarize(v))).collect::<BTreeMap<_, _>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    if !args.out.is_empty() {
        fs::write(&args.out, out_lines.join("\n") + "\n")?;
        eprintln!("per-item report -> {}", args.out);
    }
    Ok(())
}

fn summarize(a: &Agg) -> serde_json::Value {
    serde_json::json!({
        "n": a.n,
        "covered": a.covered,
        "coverage": a.covered as f64 / a.n.max(1) as f64,
        "accuracy_covered": a.correct as f64 / a.covered.max(1) as f64,
        "brier": a.brier_sum / a.covered.max(1) as f64,
        "ece": a.ece(),
    })
}
