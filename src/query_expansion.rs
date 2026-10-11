use crate::lang::Lang;
use deunicode::deunicode;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    pub original_terms: Vec<String>,
    pub expanded_terms: Vec<String>,
    /// Terms that passed all filters but had no entry in the lexical
    /// store. The store is English-only (WordNet/ConceptNet subsets), so
    /// this always contains the non-English concepts — expansion for
    /// those languages is explicitly unsupported, not silently skipped.
    pub unexpanded_non_english_terms: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Entry {
    term: String,
    related: Vec<Related>,
}

#[derive(Debug, Clone, Deserialize)]
struct Related {
    term: String,
    relation: String,
    confidence: f32,
}

#[derive(Debug, Default)]
struct LexicalStore {
    by_term: HashMap<String, Vec<Related>>,
}

static WORDNET_DATA: &[u8] = include_bytes!("../data/lexical/wordnet_subset.json.gz");
static CONCEPTNET_DATA: &[u8] = include_bytes!("../data/lexical/conceptnet_subset.json.gz");
static STORE: OnceLock<Option<LexicalStore>> = OnceLock::new();
static STEMMER: OnceLock<Stemmer> = OnceLock::new();
static NORMALIZE_RE: OnceLock<Regex> = OnceLock::new();
const MAX_EXPANSIONS_PER_TERM: usize = 1;
const CONCEPTNET_MIN_CONFIDENCE: f32 = 0.82;

/// Eagerly load the WordNet/ConceptNet lexical expansion store.
///
/// The store is a process-wide `OnceLock` that is lazily initialized on the
/// first query that needs expansion (cost: ~2s to decompress + parse 2.3MB
/// of gzipped JSON). Call this at startup (e.g. MemoryService creation or
/// binary main) to move that cost out of the query hot path. Idempotent:
/// subsequent calls return immediately.
pub fn preload_lexical_store() {
    let _ = STORE.get_or_init(load_store);
}

/// Expand query terms with lexical relations (synonyms, hypernyms, ...).
///
/// **Language coverage: English only.** The embedded store is built from
/// English WordNet/ConceptNet subsets; non-English terms are never
/// expanded. This is an explicit, documented limitation — see
/// `ExpandedQuery::unexpanded_non_english_terms`, which lists every term
/// that could not be expanded because it is not English. Spanish queries
/// (`Lang::Es`) skip expansion entirely.
pub fn expand_query_terms(input_terms: &[String], lang: Lang) -> ExpandedQuery {
    let original_terms = input_terms
        .iter()
        .map(|t| normalize_for_index(t))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>();

    // The expansion store is English WordNet/ConceptNet. For non-English
    // queries, accidental cross-language matches (Spanish "pan" = bread vs
    // English "pan" = cooking vessel) are harmful, and Spanish lexical
    // resources are out of scope. No-op: return the terms unexpanded,
    // explicitly listed as unexpanded.
    if !matches!(lang, Lang::En | Lang::Auto) {
        return ExpandedQuery {
            unexpanded_non_english_terms: original_terms.clone(),
            original_terms,
            expanded_terms: Vec::new(),
        };
    }

    let Some(store) = STORE.get_or_init(load_store).as_ref() else {
        return ExpandedQuery {
            original_terms,
            expanded_terms: Vec::new(),
            unexpanded_non_english_terms: Vec::new(),
        };
    };

    let original_set: HashSet<String> = original_terms.iter().cloned().collect();
    let mut expanded = Vec::new();
    let mut unexpanded_non_english = Vec::new();

    for term in &original_terms {
        // Never expand stopwords: their lexical neighborhoods ("and" -> "end",
        // "not") are noise that would pollute both routing and scoring.
        // Language-aware: Spanish stopwords ("no", "son") must be filtered
        // for Spanish queries, not just English ones.
        if crate::tokenizer::is_stopword_for_lang(
            term,
            crate::tokenizer::TokenizerMode::Stemmed,
            lang,
        ) {
            continue;
        }
        // Only expand focus-worthy concepts. Expanding names ("john" ->
        // "gospel accord to john"), question words, or generic verbs adds
        // noise that drowns the useful concept bridges (certificate ->
        // degree). The same judgment that identifies the question's focus
        // gates which terms earn expansions.
        if !is_expandable_concept(term, lang) {
            continue;
        }
        let mut count = 0usize;
        if let Some(related) = store.by_term.get(term) {
            for rel in related {
                if count >= MAX_EXPANSIONS_PER_TERM {
                    break;
                }
                let candidate = normalize_for_index(&rel.term);
                if candidate.is_empty()
                    || original_set.contains(&candidate)
                    || expanded.contains(&candidate)
                {
                    continue;
                }
                expanded.push(candidate);
                count += 1;
            }
        } else if term
            .chars()
            .any(|c| crate::lang::is_han(c) || crate::lang::is_hangul(c))
        {
            // Explicitly record the gap: the lexical store is English-only,
            // so CJK concepts are never expanded. Callers see the miss
            // instead of a silent no-op.
            unexpanded_non_english.push(term.clone());
        }
    }

    ExpandedQuery {
        original_terms,
        expanded_terms: expanded,
        unexpanded_non_english_terms: unexpanded_non_english,
    }
}

fn load_store() -> Option<LexicalStore> {
    let mut store = LexicalStore::default();
    let wn: Vec<Entry> = serde_json::from_slice(&decompress_gzip(WORDNET_DATA)?).ok()?;
    let cn: Vec<Entry> = serde_json::from_slice(&decompress_gzip(CONCEPTNET_DATA)?).ok()?;

    for e in wn {
        let key = normalize_for_index(&e.term);
        if key.is_empty() {
            continue;
        }
        for rel in e.related {
            if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo") {
                continue;
            }
            add_related(&mut store, &key, rel);
        }
    }

    for e in cn {
        let key = normalize_for_index(&e.term);
        if key.is_empty() {
            continue;
        }
        for rel in e.related {
            if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo" | "RelatedTo") {
                continue;
            }
            if rel.relation == "RelatedTo" && rel.confidence < CONCEPTNET_MIN_CONFIDENCE {
                continue;
            }
            add_related(&mut store, &key, rel);
        }
    }

    for rels in store.by_term.values_mut() {
        rels.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Some(store)
}

/// The lexical files are stored gzip-compressed so the embedded vocabulary
/// does not blow the crates.io package size limit; decompress at load.
fn decompress_gzip(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut decoder = libflate::gzip::Decoder::new(data).ok()?;
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok()?;
    Some(out)
}

fn add_related(store: &mut LexicalStore, key: &str, rel: Related) {
    let related_key = normalize_for_index(&rel.term);
    store
        .by_term
        .entry(key.to_string())
        .or_default()
        .push(rel.clone());

    if related_key.is_empty() || related_key == key {
        return;
    }
    if !matches!(rel.relation.as_str(), "Synonym" | "SimilarTo") {
        return;
    }
    store.by_term.entry(related_key).or_default().push(Related {
        term: key.to_string(),
        relation: rel.relation,
        confidence: rel.confidence,
    });
}

pub fn normalize_for_index(input: &str) -> String {
    // Script-aware: Han and Hangul runs are kept in the original script —
    // no deunicode transliteration (which would turn Chinese into Pinyin
    // and Korean into romanization, with homophone collisions like
    // 是/十/事 -> "shi"), no English Porter stemming — so CJK text indexes
    // as native-script tokens and the index and the query agree on script.
    // Other runs keep the historical deunicode + stem behavior; mixed
    // content gets both. For input with no Han/Hangul characters this is
    // byte-identical to the old behavior.
    let mut out: Vec<String> = Vec::new();
    let mut latin_seg = String::new();
    let mut cjk_run = String::new();
    for ch in input.chars() {
        if crate::lang::is_han(ch) || crate::lang::is_hangul(ch) {
            if !latin_seg.is_empty() {
                out.push(normalize_latin_segment(&latin_seg));
                latin_seg.clear();
            }
            cjk_run.push(ch);
        } else {
            if !cjk_run.is_empty() {
                out.push(std::mem::take(&mut cjk_run));
            }
            latin_seg.push(ch);
        }
    }
    if !latin_seg.is_empty() {
        out.push(normalize_latin_segment(&latin_seg));
    }
    if !cjk_run.is_empty() {
        out.push(cjk_run);
    }
    out.into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_latin_segment(seg: &str) -> String {
    let lowered = deunicode(seg).to_lowercase();
    let token_re =
        NORMALIZE_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9]{1,}").expect("valid regex"));
    let stemmer = STEMMER.get_or_init(|| Stemmer::create(Algorithm::English));
    token_re
        .find_iter(&lowered)
        .map(|m| stemmer.stem(m.as_str()).to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize_for_index("Virtual-Machine!"), "virtual machin");
        assert_eq!(normalize_for_index("Café"), "cafe");
        assert_eq!(normalize_for_index("running"), "run");
    }

    #[test]
    fn normalize_keeps_chinese_in_original_script() {
        // No Pinyin transliteration: the original Han is preserved so the
        // index and the query agree on script.
        assert_eq!(normalize_for_index("清华大学"), "清华大学");
        assert_eq!(normalize_for_index("我在学习Rust"), "我在学习 rust");
    }

    #[test]
    fn chinese_terms_are_reported_as_unexpanded_not_silently_dropped() {
        let terms = vec!["清华".to_string(), "大学".to_string()];
        let expanded = expand_query_terms(&terms, Lang::Zh);
        assert!(
            expanded.expanded_terms.is_empty(),
            "English-only store cannot expand Chinese terms"
        );
        assert_eq!(
            expanded.unexpanded_non_english_terms, terms,
            "the gap must be explicit, not a silent no-op"
        );
    }

    #[test]
    fn chinese_stopwords_are_not_expandable_concepts() {
        for word in ["的", "了", "我们", "因为", "可以"] {
            assert!(
                !is_expandable_concept(word, Lang::Zh),
                "{word} should not be an expandable concept"
            );
        }
        assert!(is_expandable_concept("清华", Lang::Zh));
    }

    #[test]
    fn expansion_caps_and_dedups() {
        let terms = vec!["install".to_string(), "setup".to_string()];
        let out = expand_query_terms(&terms, Lang::En);
        assert!(out.expanded_terms.len() <= terms.len() * MAX_EXPANSIONS_PER_TERM);
        for t in &out.expanded_terms {
            assert!(!out.original_terms.contains(t));
        }
    }

    #[test]
    fn install_expands_from_generated_subset() {
        let out = expand_query_terms(&["install".to_string()], Lang::En);
        assert!(!out.expanded_terms.is_empty());
        assert!(out.expanded_terms.iter().all(|t| !t.is_empty()));
    }

    #[test]
    fn spanish_expansion_is_noop() {
        // The expansion store is English-only. Spanish queries must not
        // receive English WordNet expansions ("pan" = bread must not expand
        // via English "pan" = cooking vessel).
        let out = expand_query_terms(&["biblioteca".to_string(), "pan".to_string()], Lang::Es);
        assert!(
            out.expanded_terms.is_empty(),
            "Spanish expansion must be a no-op, got {:?}",
            out.expanded_terms
        );
        assert_eq!(
            out.original_terms,
            vec!["biblioteca".to_string(), "pan".to_string()]
        );
    }

    #[test]
    fn symmetric_relations_expand_back_to_source_terms() {
        let out = expand_query_terms(&["occupation".to_string()], Lang::En);
        assert!(
            out.expanded_terms.iter().any(|t| t == "job"),
            "expected job expansion, got {:?}",
            out.expanded_terms
        );
    }

    #[test]
    fn expanded_lexical_subset_covers_common_search_terms() {
        let out = expand_query_terms(&["job".to_string(), "bug".to_string()], Lang::En);
        assert!(!out.expanded_terms.is_empty());
        for term in &out.expanded_terms {
            assert!(!out.original_terms.contains(term));
        }
    }

    #[test]
    fn certificate_expands_to_degree_via_conceptnet() {
        // Regression test for a LoCoMo miss: the question asked about a
        // "certificate" while the dialogue turn said "degree". The bridge is
        // associative (ConceptNet RelatedTo), not a WordNet synonym.
        let out = expand_query_terms(&["certificate".to_string()], Lang::En);
        let degree = normalize_for_index("degree");
        assert!(
            out.expanded_terms.iter().any(|t| t == &degree),
            "expected degree expansion, got {:?}",
            out.expanded_terms
        );
    }

    #[test]
    fn noise_terms_do_not_expand() {
        // Only the focus earns expansions. Names ("john" -> "gospel accord
        // to john"), question words, and generic verbs must not pollute the
        // expansion with noise that drowns the useful concept bridges.
        for noise in ["john", "what", "did", "receiv"] {
            let out = expand_query_terms(&[noise.to_string()], Lang::En);
            assert!(
                out.expanded_terms.is_empty(),
                "noise term {:?} should not expand, got {:?}",
                noise,
                out.expanded_terms
            );
        }
    }

    #[test]
    fn certificate_question_expands_only_focus() {
        // "What did John receive a certificate for?" — only "certificate"
        // expands (to degree/credential/etc). John, what, did, receiv stay
        // literal so the focus bridge isn't drowned in noise.
        let terms: Vec<String> = ["what", "did", "john", "receiv", "certif", "for"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = expand_query_terms(&terms, Lang::En);
        let degree = normalize_for_index("degree");
        assert!(
            out.expanded_terms.iter().any(|t| t == &degree),
            "expected degree from certif expansion, got {:?}",
            out.expanded_terms
        );
        // No biblical/historical noise from the name.
        assert!(
            !out.expanded_terms.iter().any(|t| t.contains("gospel")),
            "john should not expand to gospel noise, got {:?}",
            out.expanded_terms
        );
    }
}

/// Whether a stemmed term is an expandable concept (vs a constraint).
///
/// Expandable concepts are what questions are ABOUT (e.g. "certificate",
/// "roadtrip", "festival"). Non-expandable terms are constraints and
/// structure: question words, pronouns, generic verbs, and common names.
///
/// This is a heuristic for focus identification. A proper entity-based
/// implementation would use behood/entity recognition instead of the
/// hardcoded name list, but the classification logic (focus vs constraint)
/// is systematic and applies uniformly.
pub(crate) fn is_expandable_concept(term: &str, lang: Lang) -> bool {
    // Chinese function words are never concepts. (Interrogatives are
    // handled separately by crate::question_focus; they never reach here
    // as expandable terms, but excluding them here too is harmless.)
    if crate::tokenizer::chinese_stopwords().contains(term) {
        return false;
    }
    // Question words (stemmed forms)
    const QUESTION_WORDS: &[&str] = &[
        "what", "when", "where", "who", "whom", "whos", "why", "how", "which", "would",
    ];
    if QUESTION_WORDS.contains(&term) {
        return false;
    }
    // Pronouns (stemmed forms)
    const PRONOUNS: &[&str] = &[
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    ];
    if PRONOUNS.contains(&term) {
        return false;
    }
    // Generic verbs whose expansions are noise (stemmed forms)
    const GENERIC_VERBS: &[&str] = &[
        "be", "is", "are", "was", "were", "been", "do", "doe", "did", "done", "have", "has", "had",
        "get", "got", "make", "take", "give", "receiv", "go", "come", "see", "know", "think",
        "want", "like", "use",
    ];
    if GENERIC_VERBS.contains(&term) {
        return false;
    }
    // Auxiliary/modal verbs (stemmed forms) — structure, not concepts.
    const AUXILIARIES: &[&str] = &[
        "will", "shall", "should", "can", "could", "may", "might", "must",
    ];
    if AUXILIARIES.contains(&term) {
        return false;
    }
    if matches!(lang, Lang::Es) {
        // Spanish: same categories, stemmed/deunicoded forms (callers pass
        // `normalize_for_index` output, so "qué" arrives as "que").
        // Only consulted for Lang::Es — surface forms like "no"/"son"
        // would otherwise collide with English words.
        const QUESTION_WORDS_ES: &[&str] = &[
            "que", "quien", "cual", "donde", "cuando", "cuanto", "cuanta", "como", "porque",
        ];
        if QUESTION_WORDS_ES.contains(&term) {
            return false;
        }
        const PRONOUNS_ES: &[&str] = &[
            "yo", "tu", "el", "ella", "ello", "nosotro", "nosotra", "vosotro", "vosotra", "me",
            "te", "se", "no", "le", "mi", "su", "nuestro", "nuestra", "vuestra",
        ];
        if PRONOUNS_ES.contains(&term) {
            return false;
        }
        const GENERIC_VERBS_ES: &[&str] = &[
            "ser", "estar", "haber", "tener", "hacer", "poder", "decir", "ir", "dar", "ver",
            "saber", "querer", "es", "son", "era", "eran", "fue", "fueron", "sea", "sean", "esta",
            "estan", "estaba", "estaban", "estoy", "hay", "tiene", "tienen", "hace", "hacen",
            "puede", "pueden", "debe", "deben",
        ];
        if GENERIC_VERBS_ES.contains(&term) {
            return false;
        }
        const AUXILIARIES_ES: &[&str] = &["he", "ha", "hemo", "han", "haya", "hayan"];
        if AUXILIARIES_ES.contains(&term) {
            return false;
        }
    }
    // Common person names (stemmed forms) - expanding these gives biblical/
    // historical noise ("john" -> "gospel accord to john")
    const COMMON_NAMES: &[&str] = &[
        "john",
        "maria",
        "jame",
        "michael",
        "david",
        "sarah",
        "jennifer",
        "robert",
        "lisa",
        "william",
        "elizabeth",
        "thoma",
        "charle",
        "mary",
        "joseph",
        "daniel",
        "matthew",
        "anthony",
        "mark",
        "paul",
        "steven",
        "andrew",
        "joshua",
        "kevin",
        "brian",
        "georg",
        "edward",
        "jason",
        "jeffrey",
        "ryan",
        "jacob",
        "nichola",
        "gary",
        "jon",
        "nathan",
        "eric",
        "jonathan",
        "stephen",
        "scott",
        "justin",
        "brandon",
        "frank",
        "gregory",
        "samuel",
        "raymond",
        "alexander",
        "patrick",
        "jack",
        "denni",
        "jerry",
        "tyler",
        "aaron",
        "henry",
        "dougla",
        "nathaniel",
        "peter",
        "kyle",
        "ethan",
        "walter",
        "jeremy",
        "keith",
        "roger",
        "gerald",
        "carl",
        "arthur",
        "lawrenc",
        "dylan",
        "bryan",
        "gabriel",
        "logan",
        "alan",
        "juan",
        "wayn",
        "ralph",
        "roy",
        "eugen",
        "russel",
        "bobby",
        "victor",
        "martin",
        "philip",
        "todd",
        "jesse",
        "austin",
        "dian",
        "nanc",
        "sandra",
        "betty",
        "ashley",
        "dorothi",
        "kimberli",
        "michel",
        "carol",
        "ruth",
        "sharon",
        "laura",
        "helen",
        "deborah",
        "jessica",
        "shirley",
        "cynthia",
        "angela",
        "melissa",
        "brenda",
        "amy",
        "anna",
        "rebecca",
        "virginia",
        "kathleen",
        "pamela",
        "martha",
        "debra",
        "amanda",
        "stephani",
        "carolyn",
        "christina",
        "marilyn",
        "janet",
        "caitlin",
        "france",
        "heather",
        "diane",
        "julie",
        "olivia",
        "joyc",
        "victoria",
        "kelly",
        "christin",
        "russ",
        "emma",
        "monica",
        "melani",
        "audrey",
        "jolen",
        "sam",
        "evan",
        "dave",
        "calvin",
        "nat",
        "nate",
    ];
    if COMMON_NAMES.contains(&term) {
        return false;
    }
    true
}
