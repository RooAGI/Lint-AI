use super::{
    is_directory_empty, resolve_store_paths, ChunkStrategy, IndexLocation, IndexStore,
    MemoryIndexLayout, PipelineOptions, Tier1NerProvider, Tier1TermRankerKind,
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
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::Value;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};
/// How long to wait for another session to finish writing before giving up.
const WRITER_LOCK_WAIT: Duration = Duration::from_secs(10);
const WRITER_LOCK_RETRY: Duration = Duration::from_millis(150);

pub(crate) struct LexicalState {
    index: Index,
    /// Created on first write. Tantivy's writer lock is exclusive across
    /// processes, so constructing one eagerly means a second agent session
    /// searching the same project dies with LockBusy before it can read
    /// anything. Readers need no lock, and many can share one index.
    writer: Option<IndexWriter>,
    reader: IndexReader,
    doc_id_f: Field,
    content_f: Field,
    headings_f: Field,
    terms_f: Field,
    entities_f: Field,
}

impl LexicalState {
    pub(crate) fn new(index_dir: Option<PathBuf>) -> Result<Self> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("doc_id", STRING | STORED);
        schema_builder.add_text_field("content", TEXT);
        schema_builder.add_text_field("headings", TEXT);
        schema_builder.add_text_field("important_terms", TEXT);
        schema_builder.add_text_field("entities", TEXT);
        let schema = schema_builder.build();

        let index = match index_dir.as_deref() {
            Some(dir) => Self::open_or_create_on_disk(dir, &schema)?,
            None => Index::create_in_ram(schema),
        };
        let writer = None;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let schema_ref = index.schema();
        let doc_id_f = schema_ref.get_field("doc_id")?;
        let content_f = schema_ref.get_field("content")?;
        let headings_f = schema_ref.get_field("headings")?;
        let terms_f = schema_ref.get_field("important_terms")?;
        let entities_f = schema_ref.get_field("entities")?;
        Ok(Self {
            index,
            writer,
            reader,
            doc_id_f,
            content_f,
            headings_f,
            terms_f,
            entities_f,
        })
    }

    fn open_or_create_on_disk(dir: &Path, schema: &Schema) -> Result<Index> {
        fs::create_dir_all(dir)?;
        if is_directory_empty(dir)? {
            return Ok(Index::create_in_dir(dir, schema.clone())?);
        }
        match Index::open_in_dir(dir) {
            Ok(index) => Ok(index),
            Err(err) => Err(anyhow::anyhow!(
                "failed to open tantivy index at {}: {}",
                dir.display(),
                err
            )),
        }
    }

    pub(crate) fn upsert_record(&mut self, record: &DocRecord) -> Result<()> {
        let headings_text = record
            .section_chunks
            .iter()
            .map(|c| c.heading.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let terms_text = record
            .section_chunks
            .iter()
            .flat_map(|c| c.important_terms.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let entities_text = record
            .section_chunks
            .iter()
            .flat_map(|c| c.key_entities.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let content_text = record
            .section_chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Field handles are Copy, so they are taken before the writer borrow.
        let (doc_id_f, content_f, headings_f, terms_f, entities_f) = (
            self.doc_id_f,
            self.content_f,
            self.headings_f,
            self.terms_f,
            self.entities_f,
        );
        let doc_id = record.doc_id.clone();
        let writer = self.writer()?;
        writer.delete_term(Term::from_field_text(doc_id_f, &doc_id));
        writer.add_document(doc!(
            doc_id_f => doc_id,
            content_f => content_text,
            headings_f => headings_text,
            terms_f => terms_text,
            entities_f => entities_text
        ))?;
        Ok(())
    }

    pub(crate) fn remove_doc(&mut self, doc_id: &str) -> Result<()> {
        let doc_id_f = self.doc_id_f;
        self.writer()?
            .delete_term(Term::from_field_text(doc_id_f, doc_id));
        Ok(())
    }

    /// Takes the index lock, waiting briefly if another process is mid-write.
    ///
    /// Tantivy's writer lock is exclusive across processes, so two agent sessions
    /// working in the same project contend for it. Holding one for the life of the
    /// process makes that fatal for whichever starts second; taking it per write
    /// and waiting a moment makes them take turns.
    fn writer(&mut self) -> Result<&mut IndexWriter> {
        if self.writer.is_none() {
            let mut waited = Duration::ZERO;
            loop {
                match self.index.writer(50_000_000) {
                    Ok(writer) => {
                        self.writer = Some(writer);
                        break;
                    }
                    Err(error) if waited < WRITER_LOCK_WAIT => {
                        std::thread::sleep(WRITER_LOCK_RETRY);
                        waited += WRITER_LOCK_RETRY;
                        let _ = error;
                    }
                    Err(error) => {
                        return Err(anyhow::anyhow!(
                            "another session is writing this project's index: {error}"
                        ))
                    }
                }
            }
        }
        self.writer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("index writer missing immediately after creation"))
    }

    pub(crate) fn commit_reload(&mut self) -> Result<()> {
        // Nothing was written, so there is nothing to commit and no reason to
        // have taken the lock.
        if let Some(writer) = self.writer.as_mut() {
            writer.commit()?;
        }
        // Dropping the writer releases the lock. Keeping it would hold the index
        // against every other session in this project for as long as we run.
        self.writer = None;
        self.reader.reload()?;
        Ok(())
    }

    pub(crate) fn search(&self, query: &str, top_k: usize) -> Result<HashMap<String, f32>> {
        let searcher = self.reader.searcher();
        let mut query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.content_f,
                self.headings_f,
                self.terms_f,
                self.entities_f,
            ],
        );
        query_parser.set_field_boost(self.content_f, 1.0);
        query_parser.set_field_boost(self.headings_f, 1.4);
        query_parser.set_field_boost(self.terms_f, 2.0);
        query_parser.set_field_boost(self.entities_f, 2.4);
        let parsed = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(top_k))?;
        let mut out = HashMap::new();
        for (score, addr) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = retrieved.get_first(self.doc_id_f) {
                if let Some(doc_id) = v.as_str() {
                    out.insert(doc_id.to_string(), score);
                }
            }
        }
        Ok(out)
    }
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
    let ner_provider_name = match &options.ner_provider {
        Tier1NerProvider::Heuristic => "heuristic".to_string(),
        Tier1NerProvider::Spacy => format!("spacy:{}", options.spacy_model),
    };
    let term_ranker_name = select_term_ranker(&options.term_ranker).name().to_string();
    (ner_provider_name, term_ranker_name)
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
pub const DOC_RECORD_BUILD_VERSION: u32 = 1;

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
    hash_len_prefixed(&mut hasher, b"lint-ai-doc-record-content-hash/v1");
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

    let heuristic = HeuristicKeyEntityRanker;
    let entities_by_doc = match &options.ner_provider {
        Tier1NerProvider::Heuristic => heuristic.rank_docs(&docs)?,
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: options.spacy_model.clone(),
                script_path: default_spacy_script_path().display().to_string(),
            };
            match spacy.rank_docs(&docs) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic.rank_docs(&docs).unwrap_or_default()
                }
            }
        }
    };

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

pub(crate) fn build_doc_record(
    source_doc: &SourceDocument,
    options: &PipelineOptions,
) -> Result<DocRecord> {
    let docs = source_documents_to_tier1_inputs(std::slice::from_ref(source_doc));
    let doc = docs
        .into_iter()
        .next()
        .expect("single source document should yield one tier1 input");

    let heuristic = HeuristicKeyEntityRanker;
    let key_entities = match &options.ner_provider {
        Tier1NerProvider::Heuristic => heuristic
            .rank_docs(std::slice::from_ref(&doc))?
            .remove(&doc.id)
            .unwrap_or_default(),
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: options.spacy_model.clone(),
                script_path: default_spacy_script_path().display().to_string(),
            };
            match spacy.rank_docs(std::slice::from_ref(&doc)) {
                Ok(mut out) => out.remove(&doc.id).unwrap_or_default(),
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic
                        .rank_docs(std::slice::from_ref(&doc))
                        .unwrap_or_default()
                        .remove(&doc.id)
                        .unwrap_or_default()
                }
            }
        }
    };

    let term_ranker = select_term_ranker(&options.term_ranker);
    let important_terms = term_ranker.rank_terms(&doc);
    let (ner_provider_name, term_ranker_name) = extraction_identity(options);

    Ok(assemble_doc_record(
        source_doc,
        &doc,
        key_entities,
        important_terms,
        &ner_provider_name,
        &term_ranker_name,
        options,
    ))
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
        text: kp.text.clone(),
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
