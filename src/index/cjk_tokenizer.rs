//! CJK-aware tokenizer for tantivy's TEXT fields.
//!
//! Registered as an override of the built-in `"default"` tokenizer on every
//! index lint-ai creates (see [`register_cjk_tokenizer`]). Overriding the
//! name rather than naming a per-field tokenizer keeps the schema unchanged
//! and applies to already-existing on-disk indexes too: the tokenizer name
//! is resolved through the index's `TokenizerManager` at index and query
//! time.
//!
//! Behavior:
//! - Han runs become sliding character bigrams (`我毕业于清华大学` ->
//!   `我毕 毕业 业于 于清 清华 华大 大学`); Chinese has no word boundaries
//!   to split on, and bigrams are the standard recall-oriented segmentation
//!   for Chinese IR.
//! - Hangul runs (one eojeol each — spaces break runs) emit the eojeol
//!   plus a particle-stripped stem (`학교에` -> `학교에 학교`), so
//!   inflected forms match their stems. See
//!   [`crate::tokenizer::hangul_eojeol_tokens`].
//! - Every other run replicates tantivy's default tokenizer
//!   (split on non-alphanumeric, drop tokens >= 40 bytes, lowercase) plus
//!   deunicode folding of Latin-script tokens ("niño" -> "nino"), so the
//!   index agrees with the deunicoded boosted fields on every term.
//!   Pure-ASCII text indexes byte-identically to before this change
//!   (deunicode is the identity on ASCII). Folding is Latin-path only —
//!   Han bigrams and Hangul tokens are never transliterated.

use deunicode::deunicode;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Drop tokens whose UTF-8 byte length reaches this limit — the same limit
/// as tantivy's built-in default (`RemoveLongFilter::limit(40)` keeps
/// `len < 40`).
const MAX_TOKEN_BYTES: usize = 40;

/// CJK-aware tokenizer. See the module docs for the contract.
#[derive(Clone, Default)]
pub struct CjkTokenizer;

/// Token stream produced by [`CjkTokenizer`]. Tokens are computed eagerly;
/// positions increment once per emitted token.
#[derive(Clone, Default)]
pub struct CjkTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

fn push_default_token(
    tokens: &mut Vec<Token>,
    position: &mut usize,
    word: &str,
    offset_from: usize,
    offset_to: usize,
) {
    // Replicates SimpleTokenizer + RemoveLongFilter(40) + LowerCaser, plus
    // deunicode folding ("niño" -> "nino") so accented Latin terms agree
    // with the deunicoded boosted fields. Latin-path only: never applied
    // to Han bigrams or Hangul tokens (it would transliterate them). The
    // length check runs on the folded form (deunicode can lengthen a
    // token, e.g. "æ" -> "ae").
    let folded = deunicode(word).to_lowercase();
    if folded.len() >= MAX_TOKEN_BYTES || folded.is_empty() {
        return;
    }
    tokens.push(Token {
        text: folded,
        offset_from,
        offset_to,
        position: *position,
        position_length: 1,
    });
    *position += 1;
}

fn push_han_tokens(tokens: &mut Vec<Token>, position: &mut usize, han: &[(usize, char)]) {
    if han.len() == 1 {
        let (from, ch) = han[0];
        tokens.push(Token {
            text: ch.to_string(),
            offset_from: from,
            offset_to: from + ch.len_utf8(),
            position: *position,
            position_length: 1,
        });
        *position += 1;
        return;
    }
    // Interleaved unigrams + bigrams, mirroring
    // [`crate::tokenizer::push_han_bigrams`]: c1, c1c2, c2, c2c3, ..., cn.
    // Index and query must agree, so both emitters change together.
    for (i, (from, ch)) in han.iter().enumerate() {
        tokens.push(Token {
            text: ch.to_string(),
            offset_from: *from,
            offset_to: from + ch.len_utf8(),
            position: *position,
            position_length: 1,
        });
        *position += 1;
        if i + 1 < han.len() {
            let (next_from, next_ch) = han[i + 1];
            let mut text = String::new();
            text.push(*ch);
            text.push(next_ch);
            tokens.push(Token {
                text,
                offset_from: *from,
                offset_to: next_from + next_ch.len_utf8(),
                position: *position,
                position_length: 1,
            });
            *position += 1;
        }
    }
}

fn push_hangul_tokens(
    tokens: &mut Vec<Token>,
    position: &mut usize,
    hangul: &[(usize, char)],
) {
    // One Hangul run is one eojeol. Emit it plus the particle-stripped
    // stem (shared logic with the Rust query tokenizer, so index and query
    // agree).
    let eojeol: String = hangul.iter().map(|(_, c)| *c).collect();
    let from = hangul[0].0;
    let last = hangul[hangul.len() - 1];
    let to = last.0 + last.1.len_utf8();
    for text in crate::tokenizer::hangul_eojeol_tokens(&eojeol) {
        if text.len() >= MAX_TOKEN_BYTES {
            continue;
        }
        tokens.push(Token {
            text: text.to_lowercase(),
            offset_from: from,
            offset_to: to,
            position: *position,
            position_length: 1,
        });
        *position += 1;
    }
}

impl Tokenizer for CjkTokenizer {
    type TokenStream<'a> = CjkTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> CjkTokenStream {
        let mut tokens: Vec<Token> = Vec::new();
        let mut position = 0usize;
        let mut word = String::new();
        let mut word_from: Option<usize> = None;
        // Pending Han run as (byte offset, char) pairs.
        let mut han: Vec<(usize, char)> = Vec::new();
        // Pending Hangul run as (byte offset, char) pairs.
        let mut hangul: Vec<(usize, char)> = Vec::new();

        for (byte_idx, ch) in text.char_indices() {
            if crate::lang::is_han(ch) {
                if let Some(from) = word_from.take() {
                    push_default_token(&mut tokens, &mut position, &word, from, byte_idx);
                    word.clear();
                }
                if !hangul.is_empty() {
                    push_hangul_tokens(&mut tokens, &mut position, &hangul);
                    hangul.clear();
                }
                han.push((byte_idx, ch));
            } else if crate::lang::is_hangul(ch) {
                if let Some(from) = word_from.take() {
                    push_default_token(&mut tokens, &mut position, &word, from, byte_idx);
                    word.clear();
                }
                if !han.is_empty() {
                    push_han_tokens(&mut tokens, &mut position, &han);
                    han.clear();
                }
                hangul.push((byte_idx, ch));
            } else {
                if !han.is_empty() {
                    push_han_tokens(&mut tokens, &mut position, &han);
                    han.clear();
                }
                if !hangul.is_empty() {
                    push_hangul_tokens(&mut tokens, &mut position, &hangul);
                    hangul.clear();
                }
                if ch.is_alphanumeric() {
                    if word_from.is_none() {
                        word_from = Some(byte_idx);
                    }
                    word.push(ch);
                } else if let Some(from) = word_from.take() {
                    push_default_token(&mut tokens, &mut position, &word, from, byte_idx);
                    word.clear();
                }
            }
        }
        let end = text.len();
        if let Some(from) = word_from.take() {
            push_default_token(&mut tokens, &mut position, &word, from, end);
        }
        if !han.is_empty() {
            push_han_tokens(&mut tokens, &mut position, &han);
        }
        if !hangul.is_empty() {
            push_hangul_tokens(&mut tokens, &mut position, &hangul);
        }
        CjkTokenStream { tokens, index: 0 }
    }
}

impl TokenStream for CjkTokenStream {
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

/// Register the CJK-aware tokenizer as the `"default"` tokenizer on
/// `index`. Call once per index right after creation or opening; both
/// indexing and `QueryParser` resolve the field tokenizer through the
/// index's manager, so index-time and query-time segmentation agree.
pub fn register_cjk_tokenizer(index: &tantivy::Index) {
    index.tokenizers().register("default", CjkTokenizer);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_texts(text: &str) -> Vec<String> {
        let mut tok = CjkTokenizer;
        let mut stream = tok.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn chinese_segments_to_interleaved_unigrams_and_bigrams() {
        assert_eq!(
            token_texts("我毕业于清华大学"),
            vec![
                "我", "我毕", "毕", "毕业", "业", "业于", "于", "于清", "清", "清华", "华",
                "华大", "大", "大学", "学"
            ]
        );
    }

    #[test]
    fn korean_eojeol_emits_stem() {
        assert_eq!(token_texts("학교에"), vec!["학교에", "학교"]);
        assert_eq!(
            token_texts("학교에 갔다"),
            vec!["학교에", "학교", "갔다"]
        );
    }

    #[test]
    fn latin_path_folds_accents() {
        // Folded on the Latin path only, so accented terms agree with the
        // deunicoded boosted fields; Han/Hangul paths are untouched.
        assert_eq!(token_texts("El niño juega"), vec!["el", "nino", "juega"]);
        assert_eq!(token_texts("¿Dónde está?"), vec!["donde", "esta"]);
    }

    #[test]
    fn english_matches_default_tokenizer() {
        // Byte-identical contract with SimpleTokenizer + RemoveLongFilter(40)
        // + LowerCaser for Han/Hangul-free text.
        let cases = [
            "Hello, happy tax payer!",
            "Virtual-Machine! Café running",
            "a b cd",
            "supercalifragilisticexpialidocioussupercalifragilistic", // >= 40 bytes: dropped
        ];
        let manager = tantivy::tokenizer::TokenizerManager::default();
        for text in cases {
            let mut reference = manager.get("default").expect("default tokenizer");
            let mut expected = Vec::new();
            {
                let mut stream = reference.token_stream(text);
                while stream.advance() {
                    expected.push(stream.token().text.clone());
                }
            }
            assert_eq!(token_texts(text), expected, "mismatch for {text:?}");
        }
    }
}
