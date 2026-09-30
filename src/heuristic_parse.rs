//! Heuristic descriptor builder: a pure-Rust replacement for the spaCy parse
//! daemon (`scripts/behood_query.py --serve`).
//!
//! The judge daemon (`bekind --serve`) does not care where descriptors come
//! from — it only needs the [`ParsedText`] shape: noun-phrase descriptors
//! (`np_mentions`: text, head lemma, head POS, NER label, modifiers) and
//! proper-noun mentions. This module builds those from the crate's own
//! heuristic POS tagger, with no Python and no spaCy process.
//!
//! Chunking grammar (Penn tags from [`heuristic_pos_tags`]):
//!   chunk := (DT | PRP$ | CD)? (JJ | JJR | JJS | CD)* (NN | NNS | NNP | NNPS)+
//! A buffered run flushes into a chunk only when it ends on (or contains) a
//! nominal head; adjective/determiner-only runs are dropped. The head is the
//! last nominal token (last token as fallback); its lemma is the lowercased
//! surface form, which is all bekind consults (it lowercases anyway).
//!
//! Honest gaps versus the spaCy parse, by design:
//! - `ner_label` is always empty. bekind's teacher-vote mapping only knows
//!   spaCy-style labels (PERSON, GPE/LOC/FAC, ORG, WORK_OF_ART, EVENT); the
//!   heuristic stack cannot distinguish them, so it emits no teacher vote
//!   instead of a wrong one.
//! - `dep` on modifiers is empty — bekind stores but never reads it.
//! - No `_recover_unchunked_nominals` recovery pass: that repaired a spaCy
//!   mistagging quirk the heuristic tagger does not share.
//! - Interrogative fallback is first-nominal, not dependency-seeking: with
//!   heuristic tags an interrogative (WRB) can never sit inside a chunk, so
//!   the only fallback case is "no chunks at all".

use crate::behood_query::ParsedText;
use crate::query_semantics::{heuristic_pos_tags, POSTag};
use serde_json::{json, Value};

/// Penn tag → the universal tag bekind's descriptor checks expect
/// (`head_pos == "PROPN" / "NOUN" / "PRON"`).
fn universal_pos(penn: &str) -> &str {
    match penn {
        "NNP" | "NNPS" => "PROPN",
        "NN" | "NNS" => "NOUN",
        "PRP" => "PRON",
        other => other,
    }
}

/// Penn tags that may appear inside a heuristic noun phrase.
fn is_chunk_pos(penn: &str) -> bool {
    matches!(
        penn,
        "DT" | "PRP$" | "JJ" | "JJR" | "JJS" | "NN" | "NNS" | "NNP" | "NNPS" | "CD" | "POS"
    )
}

fn is_nominal(penn: &str) -> bool {
    matches!(penn, "NN" | "NNS" | "NNP" | "NNPS")
}

/// Split one text's POS-tagged tokens into noun-phrase chunks; each chunk is
/// a slice of token indices into `tags`.
fn chunk_indices(tags: &[POSTag]) -> Vec<Vec<usize>> {
    let mut chunks = Vec::new();
    let mut buf: Vec<usize> = Vec::new();
    let mut flush = |buf: &mut Vec<usize>, chunks: &mut Vec<Vec<usize>>| {
        if !buf.is_empty() && buf.iter().any(|&i| is_nominal(&tags[i].label)) {
            chunks.push(std::mem::take(buf));
        } else {
            buf.clear();
        }
    };
    for (i, tag) in tags.iter().enumerate() {
        if is_chunk_pos(&tag.label) {
            buf.push(i);
        } else {
            flush(&mut buf, &mut chunks);
        }
    }
    flush(&mut buf, &mut chunks);
    chunks
}

fn np_descriptor(tags: &[POSTag], chunk: &[usize], chunk_idx: usize) -> Value {
    let head_pos_in_chunk = chunk
        .iter()
        .rposition(|&i| is_nominal(&tags[i].label))
        .unwrap_or(chunk.len() - 1);
    let head_i = chunk[head_pos_in_chunk];
    let head = &tags[head_i];
    let text = chunk
        .iter()
        .map(|&i| tags[i].word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let modifiers = chunk
        .iter()
        .filter(|&&i| i != head_i)
        .map(|&i| {
            json!({
                "text": tags[i].word,
                "pos": universal_pos(&tags[i].label),
                "dep": "",
            })
        })
        .collect::<Vec<_>>();
    json!({
        "id": format!("q:{chunk_idx}"),
        "text": text,
        "head_lemma": head.word.to_lowercase(),
        "head_pos": universal_pos(&head.label),
        "ner_label": "",
        "modifiers": modifiers,
    })
}

fn mention_descriptor(tags: &[POSTag], tok_i: usize) -> Value {
    let tok = &tags[tok_i];
    json!({
        "id": format!("m:{tok_i}"),
        "text": tok.word,
        "ner_label": "",
        "pos": "PROPN",
        "head_lemma": tok.word.to_lowercase(),
    })
}

/// Parse one text into bekind-ready descriptors without spaCy.
fn parse_one(text: &str, text_idx: usize) -> ParsedText {
    let tags = heuristic_pos_tags(text);
    let mut chunks = chunk_indices(&tags);
    if chunks.is_empty() {
        // Interrogative fallback: the question names no chunkable nominal
        // ("Who did it?"). Fall back to the first nominal token so bekind
        // still has something to judge.
        if let Some(i) = tags.iter().position(|t| is_nominal(&t.label)) {
            chunks.push(vec![i]);
        }
    }
    let np_mentions = chunks
        .iter()
        .enumerate()
        .map(|(ci, chunk)| np_descriptor(&tags, chunk, ci))
        .collect::<Vec<_>>();
    let mentions = tags
        .iter()
        .enumerate()
        .filter(|(_, t)| universal_pos(&t.label) == "PROPN")
        .map(|(i, _)| mention_descriptor(&tags, i))
        .collect::<Vec<_>>();
    ParsedText {
        id: format!("p:{text_idx}"),
        mentions: Value::Array(mentions),
        np_mentions: Value::Array(np_mentions),
    }
}

/// Parse raw texts into bekind-ready descriptors without spaCy. Drop-in for
/// [`BehoodQueryDaemon::parse_texts`][crate::behood_query::BehoodQueryDaemon::parse_texts].
pub fn heuristic_parse_texts(texts: &[&str]) -> Vec<ParsedText> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| parse_one(t, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_proper_nouns_and_common_nouns() {
        let parsed = heuristic_parse_texts(&["When did Jean visit Paris"]);
        assert_eq!(parsed.len(), 1);
        let nps = parsed[0].np_mentions.as_array().unwrap();
        let texts: Vec<&str> = nps
            .iter()
            .map(|v| v["text"].as_str().unwrap())
            .collect();
        assert!(texts.contains(&"Jean"), "chunks: {texts:?}");
        assert!(texts.contains(&"Paris"), "chunks: {texts:?}");
        let jean = nps.iter().find(|v| v["text"] == "Jean").unwrap();
        assert_eq!(jean["head_pos"], "PROPN");
        assert_eq!(jean["head_lemma"], "jean");
        assert_eq!(jean["id"], "q:0");
        let mentions = parsed[0].mentions.as_array().unwrap();
        assert_eq!(mentions.len(), 2);
        assert!(mentions.iter().all(|m| m["pos"] == "PROPN"));
    }

    #[test]
    fn adjective_noun_chunk_takes_last_nominal_head() {
        let parsed = heuristic_parse_texts(&["the quiet coastal town"]);
        let nps = parsed[0].np_mentions.as_array().unwrap();
        assert_eq!(nps.len(), 1);
        let np = &nps[0];
        assert_eq!(np["text"], "the quiet coastal town");
        assert_eq!(np["head_lemma"], "town");
        assert_eq!(np["head_pos"], "NOUN");
        assert_eq!(np["modifiers"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn interrogative_only_falls_back_to_first_nominal() {
        let parsed = heuristic_parse_texts(&["Who did it?"]);
        let nps = parsed[0].np_mentions.as_array().unwrap();
        assert!(nps.is_empty(), "no nominals: {nps:?}");
        let parsed = heuristic_parse_texts(&["What city have both Jean and John visited?"]);
        let nps = parsed[0].np_mentions.as_array().unwrap();
        let heads: Vec<&str> = nps
            .iter()
            .map(|v| v["head_lemma"].as_str().unwrap())
            .collect();
        // "both" rides along as a determiner modifier, as in the spaCy
        // chunker; the heads are what bekind judges.
        assert!(heads.contains(&"city"), "heads: {heads:?}");
        assert!(heads.contains(&"jean"), "heads: {heads:?}");
        assert!(heads.contains(&"john"), "heads: {heads:?}");
    }

    #[test]
    fn descriptor_shape_matches_daemon_contract() {
        // The judge daemon must accept these descriptors exactly as it
        // accepts the Python daemon's: required keys present, dep/ner
        // label honestly empty.
        let parsed = heuristic_parse_texts(&["Jean visited Paris"]);
        let np = &parsed[0].np_mentions.as_array().unwrap()[0];
        for key in ["id", "text", "head_lemma", "head_pos", "ner_label", "modifiers"] {
            assert!(np.get(key).is_some(), "missing {key}");
        }
        let m = &parsed[0].mentions.as_array().unwrap()[0];
        for key in ["id", "text", "ner_label", "pos", "head_lemma"] {
            assert!(m.get(key).is_some(), "missing {key}");
        }
    }
}
