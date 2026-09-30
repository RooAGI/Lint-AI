//! Latin-script tokenizer for tantivy's TEXT fields.
//!
//! Registered as an override of the built-in `"default"` tokenizer on every
//! index lint-ai creates (see [`register_latin_tokenizer`]). Overriding the
//! name rather than naming a per-field tokenizer keeps the schema unchanged
//! and applies to already-existing on-disk indexes too: the tokenizer name
//! is resolved through the index's `TokenizerManager` at index and query
//! time.
//!
//! Behavior: replicates tantivy's default tokenizer exactly (split on
//! non-alphanumeric, drop tokens >= 40 bytes, lowercase) with one
//! difference — accented Latin characters are PRESERVED via dual emission:
//! "niño" indexes as both "niño" and "nino" (same position), not just
//! "nino".
//!
//! Why not deunicode, and why not fold-only: script agreement. Like Hangul
//! (which is never romanized), accented Latin is indexed in its original
//! script, and queries are tokenized the same way — index and query meet
//! in the same script. Lossy ASCII-folding ("niño" -> only "nino") would
//! conflate distinct words ("sí" = yes vs "si" = if) and is irreversible.
//! Dual emission keeps the exact form (so exact matches rank higher — a
//! query for "niño" matches two terms in an exact doc vs one in a
//! folded-only doc) while letting unaccented queries ("nino") find
//! accented text. Accent-insensitive matching is therefore supported
//! without silent conflation.
//!
//! Note: `normalize_for_index` (used for the boosted `entities` and
//! `important_terms` fields) still deunicodes. That is a SEPARATE field with
//! its own consistent normalization — query terms for boosted fields go
//! through the same `normalize_for_index`, so index and query agree within
//! each field. The content field (this tokenizer) agrees on the original
//! script plus the folded form; the boosted fields agree on the normalized
//! form. Pure-ASCII text indexes byte-identically to before this change.
//!
//! MERGE NOTE (parallel zh branch): the zh branch registers its own
//! `"default"` override (`CjkTokenizer`: Han runs -> character bigrams,
//! other runs -> default-tokenizer replication). The two overrides target
//! the same name and must be unified: the shared tokenizer must preserve
//! accents on the Latin path via DUAL EMISSION (original + folded form,
//! see `crate::tokenizer::fold_diacritics` — do not revert to fold-only
//! or exact-only) and use character bigrams on the Han path. Delete this
//! module after the merge.

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Drop tokens whose UTF-8 byte length reaches this limit — the same limit
/// as tantivy's built-in default (`RemoveLongFilter::limit(40)` keeps
/// `len < 40`). Applied to the lowercased (but not folded) token.
const MAX_TOKEN_BYTES: usize = 40;

/// Latin-script tokenizer. See the module docs for the contract.
#[derive(Clone, Default)]
pub struct LatinTokenizer;

/// Token stream produced by [`LatinTokenizer`]. Tokens are computed eagerly;
/// positions increment once per emitted token.
#[derive(Clone, Default)]
pub struct LatinTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl Tokenizer for LatinTokenizer {
    type TokenStream<'a> = LatinTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> LatinTokenStream {
        let mut tokens = Vec::new();
        let mut position = 0usize;
        // Replicates SimpleTokenizer: maximal runs of alphanumeric chars.
        let mut word_start: Option<usize> = None;
        let mut flush = |tokens: &mut Vec<Token>,
                         text: &str,
                         start: usize,
                         end: usize,
                         position: &mut usize| {
            let word = &text[start..end];
            // Lowercase only — NO deunicode. The original (accented) form is
            // always emitted so index and query meet in the same script
            // ("niño" == "niño").
            let lowered = word.to_lowercase();
            if lowered.len() >= MAX_TOKEN_BYTES || lowered.is_empty() {
                return;
            }
            let pos = *position;
            tokens.push(Token {
                text: lowered.clone(),
                offset_from: start,
                offset_to: end,
                position: pos,
                position_length: 1,
            });
            // Dual emission: the diacritic-folded form at the SAME position
            // (like a synonym). An unaccented query (`nino`) is a single
            // term that matches the folded emission; an accented query
            // (`niño`) emits both terms and matches two in an exact doc vs
            // one in a folded-only doc — exact matches rank higher with no
            // boost machinery. Pure-ASCII words are unaffected
            // (folded == lowered, single emission, byte-identical to before).
            let folded = crate::tokenizer::fold_diacritics(&lowered);
            if folded != lowered {
                tokens.push(Token {
                    text: folded,
                    offset_from: start,
                    offset_to: end,
                    position: pos,
                    position_length: 1,
                });
            }
            *position += 1;
        };
        for (idx, ch) in text.char_indices() {
            if ch.is_alphanumeric() {
                if word_start.is_none() {
                    word_start = Some(idx);
                }
            } else if let Some(start) = word_start.take() {
                flush(&mut tokens, text, start, idx, &mut position);
            }
        }
        if let Some(start) = word_start.take() {
            flush(&mut tokens, text, start, text.len(), &mut position);
        }
        LatinTokenStream { tokens, index: 0 }
    }
}

impl TokenStream for LatinTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

/// Register the Latin tokenizer as the `"default"` override on `index`.
/// Call once per index creation (and on open — the manager is per-Index,
/// so already-existing on-disk indexes pick it up too).
pub(crate) fn register_latin_tokenizer(index: &tantivy::Index) {
    index.tokenizers().register("default", LatinTokenizer);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(text: &str) -> Vec<String> {
        let mut t = LatinTokenizer;
        let mut stream = t.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn dual_emission_preserves_and_folds() {
        // Script agreement + accent-insensitivity: the accented form is
        // always emitted (index and query meet in the same script), and
        // the folded form is emitted alongside it.
        assert_eq!(
            tokens_of("El niño juega"),
            vec!["el", "niño", "nino", "juega"]
        );
        assert_eq!(
            tokens_of("¿Dónde está?"),
            vec!["dónde", "donde", "está", "esta"]
        );
        // "sí" (yes) and "si" (if) stay distinct — the exact form is never
        // destroyed, only supplemented with its folded twin.
        assert_eq!(tokens_of("sí si"), vec!["sí", "si", "si"]);
    }

    #[test]
    fn folded_form_shares_position() {
        // The folded emission sits at the same position as the original
        // (synonym-style), so phrase queries keep working.
        let mut t = LatinTokenizer;
        let mut stream = t.token_stream("niño juega");
        let mut seen = Vec::new();
        while stream.advance() {
            seen.push((stream.token().text.clone(), stream.token().position));
        }
        assert_eq!(
            seen,
            vec![
                ("niño".to_string(), 0),
                ("nino".to_string(), 0),
                ("juega".to_string(), 1),
            ]
        );
    }

    #[test]
    fn ascii_unchanged() {
        assert_eq!(
            tokens_of("The quick brown fox"),
            vec!["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn drops_long_tokens() {
        let long = "a".repeat(39);
        assert_eq!(tokens_of(&long), vec![long]);
        let too_long = "a".repeat(40);
        assert!(tokens_of(&too_long).is_empty());
    }
}
