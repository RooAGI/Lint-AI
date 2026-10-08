//! BART BPE subword tokenizer (Luyi 2026-10-07).
//!
//! Uses the HuggingFace `tokenizers` crate with BART-base's vocabulary and
//! merge rules to split words into subword pieces. "allergies" -> ["all",
//! "erg", "ies"], "allergic" -> ["all", "ergic"] — the shared "all" piece
//! lets morphologically related forms match.
//!
//! The vocab/merges are embedded at compile time so the binary is
//! self-contained. The tokenizer is built once (lazy) and reused.

use std::sync::OnceLock;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};
use tokenizers::{Decoder, PostProcessor};

/// BART-base vocabulary (embedded).
const BART_VOCAB: &str = include_str!("../tokenizers/bart/vocab.json");
/// BART-base BPE merges (embedded).
const BART_MERGES: &str = include_str!("../tokenizers/bart/merges.txt");

static BART: OnceLock<Tokenizer> = OnceLock::new();

/// Build the BART BPE tokenizer from embedded vocab/merges.
fn build_bart() -> Tokenizer {
    // Parse vocab.json: {"token": id, ...}
    let vocab: std::collections::HashMap<String, u32> =
        serde_json::from_str(BART_VOCAB).expect("bart vocab.json parses");
    // Parse merges.txt: skip #version line, each line is "a b".
    let merges: Vec<(String, String)> = BART_MERGES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            Some((parts.next()?.to_string(), parts.next()?.to_string()))
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("bart BPE builds");
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(Whitespace::default()));
    // BART uses byte-level BPE; the tokenizers crate handles decoding.
    // We keep pieces as-is (with Ġ prefix for word-initial).
    tok
}

/// Get the shared BART tokenizer (built once).
fn bart() -> &'static Tokenizer {
    BART.get_or_init(build_bart)
}

/// Tokenize text into BART BPE subword pieces.
///
/// Returns the piece strings (e.g. ["all", "erg", "ies"]). The Ġ prefix
/// (word boundary marker) is stripped for cleaner matching; pieces are
/// lowercased to agree with the index's lowercase normalization.
pub fn bart_subwords(text: &str) -> Vec<String> {
    match bart().encode(EncodeInput::Single(text.into()), false) {
        Ok(encoding) => encoding
            .get_tokens()
            .iter()
            .map(|t| {
                // Strip the Ġ word-boundary marker and lowercase.
                t.trim_start_matches('Ġ').to_lowercase()
            })
            .filter(|t| !t.is_empty())
            .collect(),
        Err(_) => {
            // Fallback: whitespace split if BPE fails.
            text.split_whitespace()
                .map(|s| s.to_lowercase())
                .collect()
        }
    }
}

/// Tokenize text into BART subwords, joined by spaces for indexing.
///
/// The subword field uses the default tokenizer (splits on whitespace),
/// so pre-tokenized pieces are joined here. Both index and query go
/// through this function, ensuring they meet in the same piece space.
pub fn bart_subwords_joined(text: &str) -> String {
    bart_subwords(text).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bart_allergies_and_allergic_share_piece() {
        let a: std::collections::HashSet<_> =
            bart_subwords("allergies").into_iter().collect();
        let b: std::collections::HashSet<_> =
            bart_subwords("allergic").into_iter().collect();
        let shared: Vec<_> = a.intersection(&b).collect();
        assert!(
            !shared.is_empty(),
            "allergies {a:?} and allergic {b:?} share {shared:?}"
        );
    }

    #[test]
    fn bart_weekends_and_weekend_share_piece() {
        let a: std::collections::HashSet<_> =
            bart_subwords("weekends").into_iter().collect();
        let b: std::collections::HashSet<_> =
            bart_subwords("weekend").into_iter().collect();
        assert!(!a.intersection(&b).collect::<Vec<_>>().is_empty());
    }
}
