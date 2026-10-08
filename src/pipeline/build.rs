use super::{
    resolve_store_paths, ChunkStrategy, IndexLocation, IndexStore, MemoryIndexLayout,
    PipelineOptions, Tier1NerProvider, Tier1TermRankerKind,
};
use crate::chunking::{
    chunk_document_hybrid, chunk_document_lines, chunk_document_sections, enrich_section_chunks,
};
use crate::claim_extractor::{ClaimExtractor, ConservativeClaimExtractor};
use crate::index::{DocRecord, MemoryIndex, Provenance};
use crate::source::SourceDocument;
use crate::temporal::extract_temporal_terms;
use crate::tier1::{
    default_spacy_script_path, CValueStyleTermRanker, HeuristicKeyEntityRanker,
    ImportantTermRanker, KeyEntityRanker, RakeStyleTermRanker, SpacyKeyEntityRanker,
    TextRankStyleTermRanker, Tier1DocInput, Tier1Entity, YakeStyleTermRanker,
    BEHOOD_NP_ENTITY_SOURCE,
};
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
/// Joined section-chunk content for a record: the text the definitional
/// semantic-tag layers judge.
fn record_content_text(record: &DocRecord) -> String {
    record
        .section_chunks
        .iter()
        .map(|c| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn select_term_ranker(ranker_kind: &Tier1TermRankerKind) -> Box<dyn ImportantTermRanker> {
    match ranker_kind {
        Tier1TermRankerKind::Yake => Box::new(YakeStyleTermRanker),
        Tier1TermRankerKind::Rake => Box::new(RakeStyleTermRanker),
        Tier1TermRankerKind::Cvalue => Box::new(CValueStyleTermRanker),
        Tier1TermRankerKind::Textrank => Box::new(TextRankStyleTermRanker),
    }
}

fn guess_doc_type(headings: &[String], content: &str) -> Option<String> {
    let joined_headings = headings.join(" ").to_lowercase();
    let content_l = content.to_lowercase();
    if joined_headings.contains("incident") || content_l.contains("postmortem") {
        Some("incident".to_string())
    } else if joined_headings.contains("runbook") || content_l.contains("playbook") {
        Some("runbook".to_string())
    } else if joined_headings.contains("changelog") || content_l.contains("release notes") {
        Some("changelog".to_string())
    } else if joined_headings.contains("reference") {
        Some("reference".to_string())
    } else if joined_headings.contains("tutorial") || joined_headings.contains("quick start") {
        Some("tutorial".to_string())
    } else if joined_headings.contains("decision") || content_l.contains("adr") {
        Some("decision".to_string())
    } else {
        None
    }
}

/// The extraction-affecting option identities that are stamped into every
/// record's provenance. These (plus the chunking settings hashed in
/// [`doc_record_content_hash`]) are the only `PipelineOptions` inputs that can
/// change a built [`DocRecord`].
fn extraction_identity(options: &PipelineOptions) -> (String, String) {
    // Luyi 2026-09-29: heuristic ranker restored alongside spaCy (spaCy stays
    // the default). The provider is part of the extraction identity so a
    // provider switch invalidates cached records.
    let ner_provider_name = match &options.ner_provider {
        Tier1NerProvider::Heuristic => "heuristic".to_string(),
        Tier1NerProvider::Spacy => format!("spacy:{}:{:?}", options.spacy_model, options.lang),
    };
    let term_ranker_name = select_term_ranker(&options.term_ranker).name().to_string();
    (ner_provider_name, term_ranker_name)
}

/// Rank Tier1 key entities for a batch of documents per `options.ner_provider`.
///
/// SpaCy documents are grouped by their selected language model so a mixed
/// corpus uses the right model for each document. A failed model group falls
/// back to the heuristic ranker for that group.
pub(crate) fn rank_key_entities_batched(
    docs: &[Tier1DocInput],
    options: &PipelineOptions,
) -> Result<HashMap<String, Vec<Tier1Entity>>> {
    let heuristic = HeuristicKeyEntityRanker;
    match &options.ner_provider {
        Tier1NerProvider::Heuristic => heuristic.rank_docs(docs),
        Tier1NerProvider::Spacy => {
            let mut by_model: BTreeMap<String, Vec<Tier1DocInput>> = BTreeMap::new();
            for doc in docs {
                by_model
                    .entry(options.spacy_model_for_text(&doc.content))
                    .or_default()
                    .push(doc.clone());
            }
            let mut out = HashMap::new();
            for (model, group_docs) in by_model {
                let spacy = SpacyKeyEntityRanker {
                    model: model.clone(),
                    script_path: default_spacy_script_path().display().to_string(),
                };
                match spacy.rank_docs(&group_docs) {
                    Ok(entities) => out.extend(entities),
                    Err(err) => {
                        eprintln!(
                            "warning: {} ranker unavailable for model {model} ({err}), falling back to heuristic",
                            spacy.name(),
                        );
                        out.extend(heuristic.rank_docs(&group_docs).unwrap_or_default());
                    }
                }
            }
            Ok(out)
        }
    }
}

fn hash_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_opt_str(hasher: &mut Sha256, value: &Option<String>) {
    match value {
        Some(text) => {
            hasher.update([1u8]);
            hash_len_prefixed(&mut *hasher, text.as_bytes());
        }
        None => hasher.update([0u8]),
    }
}

/// Schema version of the derived [`DocRecord`] build feeding
/// [`doc_record_content_hash`].
///
/// This pins the *implementation* behind the hashed option identities: it
/// must be bumped whenever anything that can change a derived record's
/// content changes without renaming an option — the NER implementation or
/// model, the term-ranker implementation, the chunking strategy
/// implementation, claim extraction, key-entity ranking, or the set of
/// fields fed into the hash.
///
/// Bumping it invalidates every stored hash (mismatch → exactly one full
/// rebuild on the next refresh, after which the new hashes are stamped),
/// so no separate version field on [`DocRecord`] is needed. Downgrades fail
/// safe in the same direction (mismatch → rebuild).
pub const DOC_RECORD_BUILD_VERSION: u32 = 2;

/// Content hash gating the doc-record rebuild short-circuit in
/// `IndexStore::prepare_pending_changes`.
///
/// The hash covers every input that can change the resulting [`DocRecord`]:
/// all record-affecting [`SourceDocument`] fields (including `concept`, which
/// feeds the key-entity ranker, and the ordered `headings`/`links`/`filters`),
/// the extraction-affecting option identities from [`extraction_identity`]
/// (NER provider + model, term ranker), the chunking strategy and its numeric
/// settings, the claim-extraction flag, and [`DOC_RECORD_BUILD_VERSION`], which
/// pins the implementation behind those option identities.
///
/// [`assemble_doc_record`] calls this to stamp each built record, so the
/// cheap pre-build hash computed here and the hash on a stored record agree
/// by construction. Keep the hashed field set in sync with
/// `assemble_doc_record`: any input it starts (or stops) consuming must be
/// added (or may be dropped) here, and the domain tag bumped; whenever
/// derived-record behavior changes without renaming an option, bump
/// [`DOC_RECORD_BUILD_VERSION`] instead.
pub(crate) fn doc_record_content_hash(
    source_doc: &SourceDocument,
    options: &PipelineOptions,
) -> String {
    doc_record_content_hash_with_version(source_doc, options, DOC_RECORD_BUILD_VERSION)
}

/// [`doc_record_content_hash`] parameterized by build version.
///
/// The version travels as data inside the hash (length-prefixed, like the
/// other inputs) rather than as a struct field: any bump makes every stored
/// hash mismatch, forcing exactly one rebuild. The domain tag stays fixed —
/// it identifies the hash *construction format*, while the version pins the
/// *build semantics*.
pub(crate) fn doc_record_content_hash_with_version(
    source_doc: &SourceDocument,
    options: &PipelineOptions,
    build_version: u32,
) -> String {
    let (ner_provider_name, term_ranker_name) = extraction_identity(options);
    let chunk_strategy_name = match options.chunk_strategy {
        ChunkStrategy::Heading => "heading",
        ChunkStrategy::Line => "line",
        ChunkStrategy::Hybrid => "hybrid",
    };
    let mut hasher = Sha256::new();
    // Domain separator: bump if the hashed field set ever changes, so old
    // hashes never compare equal to new ones.
    //
    // v2: key_phrases joined the field set. A backfilled document must
    // compare hash-unequal to its phrase-less record, otherwise
    // prepare_pending_changes skips reprocessing and the phrases never
    // reach the segment profiles.
    hash_len_prefixed(&mut hasher, b"lint-ai-doc-record-content-hash/v2");
    // Build version pins the implementation behind the option identities;
    // any bump invalidates every stored hash (mismatch -> rebuild).
    hash_len_prefixed(&mut hasher, &build_version.to_le_bytes());
    hash_len_prefixed(&mut hasher, source_doc.doc_id.as_bytes());
    hash_len_prefixed(&mut hasher, source_doc.source.as_bytes());
    hash_len_prefixed(&mut hasher, source_doc.content.as_bytes());
    hash_len_prefixed(&mut hasher, source_doc.concept.as_bytes());
    hasher.update((source_doc.headings.len() as u64).to_le_bytes());
    for heading in &source_doc.headings {
        hash_len_prefixed(&mut hasher, heading.as_bytes());
    }
    hasher.update((source_doc.links.len() as u64).to_le_bytes());
    for link in &source_doc.links {
        hash_len_prefixed(&mut hasher, link.as_bytes());
    }
    hash_opt_str(&mut hasher, &source_doc.timestamp);
    hasher.update((source_doc.doc_length as u64).to_le_bytes());
    hash_opt_str(&mut hasher, &source_doc.author_agent);
    hash_opt_str(&mut hasher, &source_doc.group_id);
    hasher.update((source_doc.filters.len() as u64).to_le_bytes());
    for (key, value) in &source_doc.filters {
        hash_len_prefixed(&mut hasher, key.as_bytes());
        hash_len_prefixed(&mut hasher, value.as_bytes());
    }
    // Grammar-accepted entity mentions: backfilling these must invalidate
    // the stored record so the segment entity channel picks them up.
    hasher.update((source_doc.key_phrases.len() as u64).to_le_bytes());
    for phrase in &source_doc.key_phrases {
        hash_len_prefixed(&mut hasher, phrase.text.as_bytes());
        hash_len_prefixed(&mut hasher, phrase.kind.as_bytes());
    }
    hash_len_prefixed(&mut hasher, ner_provider_name.as_bytes());
    hash_len_prefixed(&mut hasher, term_ranker_name.as_bytes());
    hash_len_prefixed(&mut hasher, chunk_strategy_name.as_bytes());
    hasher.update((options.chunk_lines as u64).to_le_bytes());
    hasher.update((options.chunk_overlap as u64).to_le_bytes());
    hasher.update((options.chunk_target_tokens as u64).to_le_bytes());
    hasher.update((options.chunk_max_tokens as u64).to_le_bytes());
    hasher.update([u8::from(options.claim_extraction)]);
    format!("{:x}", hasher.finalize())
}

pub fn source_documents_to_tier1_inputs(docs: &[SourceDocument]) -> Vec<Tier1DocInput> {
    docs.iter()
        .map(|doc| Tier1DocInput {
            id: doc.doc_id.clone(),
            source: doc.source.clone(),
            content: doc.content.clone(),
            concept: doc.concept.clone(),
            headings: doc.headings.clone(),
        })
        .collect()
}

/// Extracts [`DocRecord`]s from source documents (NER + term ranking +
/// chunking). This is the expensive per-document pipeline phase; the
/// benchmark harness calls it once per question and builds both the
/// single-layout snapshot and the segmented index from the same records
/// instead of extracting twice.
pub fn build_doc_records(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<Vec<DocRecord>> {
    let docs = source_documents_to_tier1_inputs(source_docs);

    // Luyi 2026-09-29: provider-selected NER; spaCy failures fall back to the
    // heuristic ranker (never silently to zero entities).
    let entities_by_doc = rank_key_entities_batched(&docs, options)?;

    let term_ranker = select_term_ranker(&options.term_ranker);
    let (ner_provider_name, term_ranker_name) = extraction_identity(options);

    let source_by_id: HashMap<&str, &SourceDocument> = source_docs
        .iter()
        .map(|doc| (doc.doc_id.as_str(), doc))
        .collect();

    let mut records = Vec::new();
    for doc in docs {
        let source_doc = source_by_id
            .get(doc.id.as_str())
            .copied()
            .expect("source document should exist for tier1 input");
        let key_entities = entities_by_doc.get(&doc.id).cloned().unwrap_or_default();
        let important_terms = term_ranker.rank_terms(&doc);
        records.push(assemble_doc_record(
            source_doc,
            &doc,
            key_entities,
            important_terms,
            &ner_provider_name,
            &term_ranker_name,
            options,
        ));
    }

    Ok(records)
}

/// Record assembly from precomputed NER key entities. Lets the refresh loop
/// batch NER across dirty docs (one daemon call) instead of one call per
/// document, while the per-doc bookkeeping stays per-doc.
pub(crate) fn build_doc_record_with_entities(
    source_doc: &SourceDocument,
    doc: &Tier1DocInput,
    key_entities: Vec<Tier1Entity>,
    options: &PipelineOptions,
) -> Result<DocRecord> {
    let term_ranker = select_term_ranker(&options.term_ranker);
    let important_terms = term_ranker.rank_terms(doc);
    let (ner_provider_name, term_ranker_name) = extraction_identity(options);

    Ok(assemble_doc_record(
        source_doc,
        doc,
        key_entities,
        important_terms,
        &ner_provider_name,
        &term_ranker_name,
        options,
    ))
}

/// Reduce a grammar-accepted key phrase to its referring core: the head noun
/// plus proper-noun modifiers. Possessors ("user's", "my"), determiners
/// ("the") and descriptive modifiers ("favorite") are not the thing the
/// phrase denotes, so they must not enter the entity channel: a generic
/// token like "user" is rare in the entities field, and its high per-field
/// IDF at 2.4x weight inflated near-miss documents (mem-04: the restaurant
/// fact outscored the peanut-allergy fact on "user" alone).
///
/// Rule (systematic, applies to every key phrase):
/// - drop possessive-marked tokens ("user's", "dogs'") and possessive
///   determiners (my/your/his/her/its/our/their);
/// - drop leading articles (the/a/an);
/// - keep capitalized tokens (proper-noun modifiers are part of naming:
///   "Harry Potter" in "Harry Potter conference") and the final token
///   (the head; English NPs are head-final);
/// - drop remaining lowercase non-final tokens (descriptive modifiers).
/// Fail-open: if nothing survives, the original text is kept.
///
/// Limitations (documented, not fixed here): capitalization is a heuristic
/// proxy for proper-nounhood (misses lowercase proper nouns); multi-word
/// common-noun compounds reduce to the final token ("ice cream" -> "cream");
/// interior glue ("of" in "University of Washington") is dropped. The full
/// phrase text stays in the content/terms fields, so nothing becomes
/// unretrievable -- only the 2.4x entity-channel precision changes.
fn head_noun_phrase(text: &str) -> String {
    const POSSESSIVE_DETS: [&str; 7] = ["my", "your", "his", "her", "its", "our", "their"];
    const ARTICLES: [&str; 3] = ["the", "a", "an"];

    fn is_possessive(tok: &str) -> bool {
        let lower = tok.to_lowercase();
        lower.ends_with("'s")
            || lower.ends_with("\u{2019}s")
            || lower.ends_with("s'")
            || lower.ends_with("s\u{2019}")
            || POSSESSIVE_DETS.contains(&lower.as_str())
    }

    // Word tokens with original case; apostrophes stay inside the token so
    // possessives ("user's") are detectable.
    let mut tokens: Vec<&str> = text
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}'))
        .filter(|t| !t.is_empty())
        .collect();

    tokens.retain(|t| !is_possessive(t));
    while tokens
        .first()
        .is_some_and(|t| ARTICLES.contains(&t.to_lowercase().as_str()))
    {
        tokens.remove(0);
    }

    let n = tokens.len();
    let kept: Vec<&str> = tokens
        .into_iter()
        .enumerate()
        .filter(|(i, t)| *i == n - 1 || t.chars().next().is_some_and(|c| c.is_uppercase()))
        .map(|(_, t)| t)
        .collect();

    if kept.is_empty() {
        text.to_string()
    } else {
        kept.join(" ")
    }
}

fn assemble_doc_record(
    source_doc: &SourceDocument,
    doc: &Tier1DocInput,
    key_entities: Vec<Tier1Entity>,
    important_terms: Vec<crate::tier1::RankedTerm>,
    ner_provider_name: &str,
    term_ranker_name: &str,
    options: &PipelineOptions,
) -> DocRecord {
    // Grammar-accepted entity mentions (behood noun-phrase layer) join the
    // key entities with full score and provenance. The segment summary
    // indexes their literal (unstemmed) tokens in the entity channel, which
    // is exempt from the local-memory term cap.
    //
    // Grammar-accepted mentions get 2x score: the dependency grammar +
    // ontology is higher precision than heuristic NER, so these are stronger
    // retrieval signals in both routing and per-document entity scoring.
    let mut key_entities = key_entities;
    key_entities.extend(source_doc.key_phrases.iter().map(|kp| Tier1Entity {
        // Luyi 2026-09-28: admit the head noun, not the whole possessive
        // phrase -- possessors and descriptive modifiers are not the thing
        // the phrase denotes ("user's favorite restaurant" -> "restaurant").
        text: head_noun_phrase(&kp.text),
        label: kp.kind.clone(),
        start: 0,
        end: 0,
        score: Some(2.0),
        source: BEHOOD_NP_ENTITY_SOURCE.to_string(),
    }));
    let probable_topic = if let Some(first_heading) = doc.headings.first() {
        Some(first_heading.clone())
    } else {
        important_terms.first().map(|t| t.term.clone())
    };
    let base_chunks = match &options.chunk_strategy {
        ChunkStrategy::Heading => chunk_document_sections(&doc.content, &doc.id),
        ChunkStrategy::Line => chunk_document_lines(
            &doc.content,
            &doc.id,
            options.chunk_lines,
            options.chunk_overlap,
        ),
        ChunkStrategy::Hybrid => chunk_document_hybrid(
            &doc.content,
            &doc.id,
            options.chunk_lines,
            options.chunk_overlap,
            options.chunk_target_tokens,
            options.chunk_max_tokens,
        ),
    };
    let mut section_chunks = enrich_section_chunks(base_chunks, &key_entities, &important_terms);
    for chunk in &mut section_chunks {
        if chunk.timestamp.is_none() {
            chunk.timestamp = source_doc.timestamp.clone();
        }
    }
    let temporal_terms =
        extract_temporal_terms(source_doc.timestamp.as_deref(), &doc.content, &doc.headings);
    // Synthetic semantic tags (Luyi 2026-10-07): bekind's canonical
    // judgments computed once at write time (MemoryService is the single
    // write path). Stored in DocRecord, read by the index builder.
    let semantic_tags: Vec<String> = {
        let contents: Vec<&str> = section_chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect();
        crate::semantic_tags::batch_doc_semantic_tags(&contents)
            .into_iter()
            .flatten()
            .collect()
    };
    // Semantic tags as key phrases (Luyi 2026-10-08): "key_phrases and
    // concept and semantic tags to me are the same". The beKIND scope/kind
    // judgments ("food", "habitual", "weekend") are stored as KeyPhrase
    // entries with kind="semantic_tag", unifying the annotation mechanism.
    // Single truth (Luyi 2026-10-08): the tags are written back into a
    // SourceDocument (enriched), not appended only to DocRecord. The index
    // populates its semantic_tags field from these.
    let mut enriched_source = source_doc.clone();
    for tag in &semantic_tags {
        // Avoid duplicates if the tag was already added.
        if !enriched_source.key_phrases.iter().any(|kp| kp.text == *tag) {
            enriched_source.key_phrases.push(crate::source::KeyPhrase {
                text: tag.clone(),
                kind: "semantic_tag".to_string(),
            });
        }
    }
    // Use the enriched SourceDocument's key_phrases for the DocRecord.
    // DocRecord is a deterministic view of SourceDocument; no second way.
    let key_phrases = enriched_source.key_phrases.clone();
    let mut record = DocRecord {
        doc_id: doc.id.clone(),
        source: doc.source.clone(),
        content: doc.content.clone(),
        timestamp: source_doc.timestamp.clone(),
        doc_length: source_doc.doc_length,
        author_agent: source_doc.author_agent.clone(),
        group_id: source_doc.group_id.clone(),
        filters: source_doc.filters.clone(),
        probable_topic,
        doc_type_guess: guess_doc_type(&doc.headings, &doc.content),
        headings: doc.headings.clone(),
        doc_links: source_doc.links.clone(),
        temporal_terms,
        key_entities,
        important_terms,
        section_chunks,
        // Carry the source-level key phrases (and their extraction stamp)
        // into the record so they survive a persist/reload round-trip.
        key_phrases,
        key_phrase_extraction_hash: source_doc.key_phrase_extraction_hash.clone(),
        embedding: None,
        top_claims: Vec::new(),
        provenance: Provenance {
            source: doc.source.clone(),
            timestamp: source_doc.timestamp.clone(),
            ner_provider: ner_provider_name.to_string(),
            term_ranker: term_ranker_name.to_string(),
            index_version: "v1-memory-hybrid".to_string(),
        },
        // Stamped here so both build paths (single + batch) agree with the
        // cheap pre-build hash by construction.
        content_hash: doc_record_content_hash(source_doc, options),
        semantic_tags,
    };

    if options.claim_extraction {
        let extractor = ConservativeClaimExtractor;
        record.top_claims = extractor.extract(&record).claims;
    }
    record
}

/// Builds a single-layout [`MemoryIndex`] from pre-extracted records.
/// Paired with [`build_doc_records`]: extract once, then build the snapshot
/// and any segmented indexes from the same records.
pub fn build_query_snapshot_from_records(
    records: &[DocRecord],
    options: &PipelineOptions,
) -> Result<MemoryIndex> {
    let store_paths = resolve_store_paths(None, options)?;
    Ok(MemoryIndex::from_records_with_lexical_dir(
        records.to_vec(),
        store_paths.lexical_dir.as_deref(),
        options.text_rerank_ngram,
        options.text_rerank_lcs,
        options.claim_extraction,
    ))
}

pub fn build_query_snapshot(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<MemoryIndex> {
    let records = build_doc_records(source_docs, options)?;
    build_query_snapshot_from_records(&records, options)
}

#[allow(clippy::too_many_arguments)]
pub fn build_query_snapshot_from_source_documents(
    source_docs: &[SourceDocument],
    provider: &Tier1NerProvider,
    spacy_model: &str,
    ranker_kind: &Tier1TermRankerKind,
    chunk_strategy: &ChunkStrategy,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
    text_rerank_ngram: bool,
    text_rerank_lcs: bool,
) -> Result<MemoryIndex> {
    let options = PipelineOptions {
        ner_provider: provider.clone(),
        spacy_model: spacy_model.to_string(),
        lang: crate::lang::Lang::Auto,
        term_ranker: ranker_kind.clone(),
        chunk_strategy: chunk_strategy.clone(),
        chunk_lines,
        chunk_overlap,
        chunk_target_tokens,
        chunk_max_tokens,
        text_rerank_ngram,
        text_rerank_lcs,
        claim_extraction: false,
        supersession: crate::semantic_relations::SupersessionOptions::default(),
        index_location: IndexLocation::InMemory,
        memory_index_layout: MemoryIndexLayout::Single,
        fuse_global_arm: false,
        conversational_rerank: true,
        structured_fact_retrieval: true,
        key_phrase_enrichment: false,
        extractor_script: None,
    };
    build_query_snapshot(source_docs, &options)
}

pub fn build_index_store(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<IndexStore> {
    let mut index = IndexStore::with_documents(options.clone(), source_docs.to_vec());
    index.refresh()?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::head_noun_phrase;

    #[test]
    fn possessive_phrase_reduces_to_head_noun() {
        assert_eq!(head_noun_phrase("user's favorite restaurant"), "restaurant");
    }

    #[test]
    fn proper_noun_modifiers_are_kept() {
        assert_eq!(
            head_noun_phrase("Harry Potter conference"),
            "Harry Potter conference"
        );
    }

    #[test]
    fn leading_article_is_dropped() {
        assert_eq!(head_noun_phrase("the Eiffel Tower"), "Eiffel Tower");
    }

    #[test]
    fn possessive_determiner_is_dropped() {
        assert_eq!(head_noun_phrase("my mom"), "mom");
    }

    #[test]
    fn single_token_phrase_survives() {
        assert_eq!(head_noun_phrase("EpiPen"), "EpiPen");
    }

    #[test]
    fn empty_after_strip_falls_back_to_original() {
        assert_eq!(head_noun_phrase("John's"), "John's");
    }
}
