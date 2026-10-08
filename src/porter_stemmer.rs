//! Porter stemmer wrapper (Luyi 2026-10-07).
//!
//! Maps words to their morphological base form using the Snowball/Porter2
//! English stemmer from the `rust-stemmers` crate. "weekends" -> "weekend",
//! "allergies" -> "allergi". Used by the subword field tokenizer for
//! morphological matching.
//!
//! Note: Porter is a stemmer, not a true lemmatizer (it chops endings
//! rather than mapping to dictionary forms). "allergies" -> "allergi" and
//! "allergic" -> "allerg" do NOT match — that derivational gap is bridged
//! by the "food" kind tag, not by stemming.

use rust_stemmers::{Algorithm, Stemmer};
use std::sync::OnceLock;

static STEMMER: OnceLock<Stemmer> = OnceLock::new();

fn stemmer() -> &'static Stemmer {
    STEMMER.get_or_init(|| Stemmer::create(Algorithm::English))
}

/// Stem a (lowercased) word to its Porter base form.
pub fn porter_stem(word: &str) -> String {
    stemmer().stem(word).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porter_weekends_to_weekend() {
        assert_eq!(porter_stem("weekends"), "weekend");
        assert_eq!(porter_stem("weekend"), "weekend");
    }

    #[test]
    fn porter_allergies_to_allergi() {
        // Documented: Porter does NOT unify "allergies" with "allergic".
        assert_eq!(porter_stem("allergies"), "allergi");
        assert_eq!(porter_stem("allergic"), "allerg");
    }
}
