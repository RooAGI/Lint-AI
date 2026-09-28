use crate::source::SourceDocument;
use crate::temporal::parse_temporal_date;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SupersessionOptions {
    pub enabled: bool,
    pub suppress_confidence: f32,
    pub conflict_confidence: f32,
}

impl Default for SupersessionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            suppress_confidence: 0.90,
            conflict_confidence: 0.65,
        }
    }
}

impl SupersessionOptions {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.conflict_confidence)
            || !(0.0..=1.0).contains(&self.suppress_confidence)
        {
            anyhow::bail!("semantic relationship confidence thresholds must be between 0 and 1");
        }
        if self.conflict_confidence > self.suppress_confidence {
            anyhow::bail!("conflict confidence must not exceed suppress confidence");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRelationKind {
    Supersedes,
    ConflictsWith,
    Confirms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticStatus {
    Current,
    Superseded,
    Conflicted,
    Historical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticClaim {
    pub claim_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source_doc_id: String,
    pub evidence: String,
    pub effective_at: Option<String>,
    pub scope: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRelation {
    pub relation_id: String,
    pub kind: SemanticRelationKind,
    /// The newer or confirming claim.
    pub source_claim_id: String,
    /// The older or conflicting claim.
    pub target_claim_id: String,
    pub source_doc_id: String,
    pub target_doc_id: String,
    pub confidence: f32,
    pub method: String,
    pub evidence: Vec<String>,
    pub scope: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentSemanticState {
    pub status: Option<SemanticStatus>,
    pub superseded_by: Option<String>,
    pub relation_confidence: Option<f32>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticRelationStore {
    claims: Vec<SemanticClaim>,
    relations: Vec<SemanticRelation>,
    document_states: HashMap<String, DocumentSemanticState>,
    /// Incremental-update indexes. Each is derived from `claims`/`relations`
    /// and is fully rebuilt by [`Self::try_from_documents`]; [`Self::update_documents`]
    /// maintains them for the documents it touches.
    claim_by_id: HashMap<String, SemanticClaim>,
    claims_by_doc: HashMap<String, Vec<String>>,
    /// Canonical claim chains per relation key, in [`canonical_claim_cmp`] order.
    chains: HashMap<RelationKey, Vec<String>>,
    chain_relations: HashMap<RelationKey, Vec<SemanticRelation>>,
    explicit_relations: Vec<SemanticRelation>,
    /// Every `supersedes_id` link (source doc -> target doc id), whether or not
    /// it currently yields a relation: scope mismatches and missing targets
    /// produce no relation, but a later change on either side can create one.
    supersedes_links: HashMap<String, String>,
    /// Whether the store was ever built (full or incremental). A store built
    /// from a claim-free corpus is still built: without this flag,
    /// `update_documents` could not tell "never built" from "built but
    /// empty" and would fall back to a corpus-wide rebuild on every update.
    initialized: bool,
}

/// Identifies one canonical-claim chain: claims that can supersede, confirm,
/// or conflict with each other share scope, subject, and predicate.
type RelationKey = (String, String, String);

impl SemanticRelationStore {
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty() && self.document_states.is_empty()
    }
    pub fn from_documents<'a, I>(documents: I, options: SupersessionOptions) -> Self
    where
        I: IntoIterator<Item = &'a SourceDocument>,
    {
        Self::try_from_documents(documents, options).expect("invalid semantic supersession options")
    }

    pub fn try_from_documents<'a, I>(
        documents: I,
        options: SupersessionOptions,
    ) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = &'a SourceDocument>,
    {
        if !options.enabled {
            return Ok(Self::default());
        }
        options.validate()?;

        // Sort by doc id so explicit-relation order (and everything derived
        // from it) is deterministic across runs; HashMap iteration order is not.
        let mut documents = documents.into_iter().collect::<Vec<_>>();
        documents.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));

        let docs_by_id = documents
            .iter()
            .map(|doc| (doc.doc_id.as_str(), *doc))
            .collect::<HashMap<_, _>>();
        let mut claims = documents
            .iter()
            .flat_map(|doc| extract_claims(doc))
            .collect::<Vec<_>>();
        claims.sort_by(canonical_claim_cmp);
        let claim_by_id = claims
            .iter()
            .map(|claim| (claim.claim_id.clone(), claim.clone()))
            .collect::<HashMap<_, _>>();
        let mut claims_by_doc = HashMap::<String, Vec<String>>::new();
        let mut chains = HashMap::<RelationKey, Vec<String>>::new();
        for claim in &claims {
            claims_by_doc
                .entry(claim.source_doc_id.clone())
                .or_default()
                .push(claim.claim_id.clone());
            chains
                .entry(relation_key(claim))
                .or_default()
                .push(claim.claim_id.clone());
        }

        // Explicit links are authoritative evidence, regardless of source type.
        let mut explicit_relations = Vec::new();
        let mut supersedes_links = HashMap::<String, String>::new();
        for doc in &documents {
            if let Some(target_id) = doc.filters.get("supersedes_id") {
                supersedes_links.insert(doc.doc_id.clone(), target_id.clone());
            }
            let Some((source_claim, target_claim)) =
                explicit_relation_claims(doc, &docs_by_id, &claims_by_doc, &claim_by_id)
            else {
                continue;
            };
            push_relation(
                &mut explicit_relations,
                &mut HashSet::new(),
                &source_claim,
                &target_claim,
                SemanticRelationKind::Supersedes,
                1.0,
                "explicit",
                vec![format!(
                    "{} explicitly supersedes {}",
                    doc.doc_id, target_claim.source_doc_id
                )],
            );
        }

        let mut chain_relations = HashMap::<RelationKey, Vec<SemanticRelation>>::new();
        for (key, chain) in &chains {
            chain_relations.insert(
                key.clone(),
                build_chain_relations(chain, &claim_by_id, &docs_by_id),
            );
        }

        let relations =
            merge_relations_canonical(&explicit_relations, &chain_relations, &claim_by_id);
        let document_states = build_document_states(
            &documents,
            &relations,
            &docs_by_id,
            &claims_by_doc,
            &claim_by_id,
            options,
        );
        Ok(Self {
            claims,
            relations,
            document_states,
            claim_by_id,
            claims_by_doc,
            chains,
            chain_relations,
            explicit_relations,
            supersedes_links,
            initialized: true,
        })
    }

    /// Incrementally refreshes the store after `changed` documents were
    /// upserted and `removed` document ids were tombstoned, without
    /// re-extracting claims from the untouched corpus.
    ///
    /// `all_docs` is the full current source-document map (post-update);
    /// `changed` entries must be drawn from it. Only relations involving the
    /// changed/removed documents' canonical-claim neighborhoods are rebuilt;
    /// everything else is carried over untouched. The result is identical to
    /// [`Self::try_from_documents`] over `all_docs` (modulo the explicit
    /// determinism note there).
    ///
    /// Requires the same `options` the store was built with; option changes
    /// need a full rebuild.
    pub fn update_documents(
        &mut self,
        all_docs: &HashMap<String, SourceDocument>,
        changed: &[&SourceDocument],
        removed: &[String],
        options: SupersessionOptions,
    ) -> anyhow::Result<()> {
        if !options.enabled {
            *self = Self::default();
            return Ok(());
        }
        options.validate()?;
        let docs_by_id = all_docs
            .iter()
            .map(|(id, doc)| (id.as_str(), doc))
            .collect::<HashMap<_, _>>();

        // Defensive: a store that was never built falls back to the oracle.
        // (`initialized`, not claim emptiness: a store built from a
        // claim-free corpus must stay on the incremental path.)
        if !self.initialized && (!changed.is_empty() || !removed.is_empty()) {
            *self = Self::try_from_documents(all_docs.values(), options)?;
            return Ok(());
        }

        let mut changed_ids: Vec<&str> = changed.iter().map(|doc| doc.doc_id.as_str()).collect();
        changed_ids.sort_unstable();
        changed_ids.dedup();
        let changed_set: HashSet<&str> = changed_ids.iter().copied().collect();
        let removed_set: HashSet<&str> = removed.iter().map(String::as_str).collect();
        if changed_ids.is_empty() && removed_set.is_empty() {
            return Ok(());
        }

        let mut affected_keys = HashSet::<RelationKey>::new();
        let mut affected_docs = HashSet::<String>::new();
        let mut claims_changed = false;
        let mut dropped_claim_ids = HashSet::<String>::new();

        // --- tombstoned documents: drop their claims from every index ---
        for doc_id in &removed_set {
            affected_docs.insert((*doc_id).to_string());
            self.document_states.remove(*doc_id);
            self.supersedes_links.remove(*doc_id);
            if let Some(old_ids) = self.claims_by_doc.remove(*doc_id) {
                if !old_ids.is_empty() {
                    claims_changed = true;
                }
                for claim_id in &old_ids {
                    dropped_claim_ids.insert(claim_id.clone());
                    if let Some(claim) = self.claim_by_id.remove(claim_id) {
                        let key = relation_key(&claim);
                        affected_keys.insert(key.clone());
                        if let Some(chain) = self.chains.get_mut(&key) {
                            chain.retain(|id| id != claim_id);
                        }
                    }
                }
            }
        }

        // --- changed documents: swap their claims ---
        // Always remove and reinsert: claim ids are (doc id, index) based, so
        // an edit can keep the same ids while changing scope, timestamps, or
        // content. Comparing ids alone would miss those changes.
        let mut fresh_claims = Vec::<SemanticClaim>::new();
        for doc_id in &changed_ids {
            let Some(doc) = docs_by_id.get(doc_id) else {
                continue;
            };
            affected_docs.insert((*doc_id).to_string());
            if let Some(old_ids) = self.claims_by_doc.remove(*doc_id) {
                if !old_ids.is_empty() {
                    claims_changed = true;
                }
                for claim_id in &old_ids {
                    dropped_claim_ids.insert(claim_id.clone());
                    if let Some(claim) = self.claim_by_id.remove(claim_id) {
                        let key = relation_key(&claim);
                        affected_keys.insert(key.clone());
                        if let Some(chain) = self.chains.get_mut(&key) {
                            chain.retain(|id| id != claim_id);
                        }
                    }
                }
            }
            let mut new_claims = extract_claims(doc);
            new_claims.sort_by(canonical_claim_cmp);
            if !new_claims.is_empty() {
                claims_changed = true;
            }
            // Group the doc's new claims per key, then merge each group
            // into its chain (chains stay in canonical order).
            let mut by_key = HashMap::<RelationKey, Vec<&SemanticClaim>>::new();
            for claim in &new_claims {
                let key = relation_key(claim);
                affected_keys.insert(key.clone());
                self.claim_by_id
                    .insert(claim.claim_id.clone(), claim.clone());
                by_key.entry(key).or_default().push(claim);
            }
            for (key, group) in by_key {
                let chain = self.chains.entry(key).or_default();
                chain.extend(group.iter().map(|claim| claim.claim_id.clone()));
                chain.sort_by(|a, b| {
                    canonical_claim_cmp(&self.claim_by_id[a], &self.claim_by_id[b])
                });
            }
            let new_ids: Vec<String> = new_claims
                .iter()
                .map(|claim| claim.claim_id.clone())
                .collect();
            self.claims_by_doc.insert((*doc_id).to_string(), new_ids);
            fresh_claims.extend(new_claims);
            // The link map tracks filters, which can change without the
            // claims changing (e.g. `supersedes_id` retargeted).
            match doc.filters.get("supersedes_id") {
                Some(target) => {
                    self.supersedes_links
                        .insert((*doc_id).to_string(), target.clone());
                }
                None => {
                    self.supersedes_links.remove(*doc_id);
                }
            }
        }
        if claims_changed {
            // Prune exactly the dropped claim ids (not "absent from
            // claim_by_id": reinserted claims share ids with the stale
            // entries they replace, so that filter would keep ghosts and
            // duplicate them below).
            self.claims
                .retain(|claim| !dropped_claim_ids.contains(&claim.claim_id));
            self.claims.extend(fresh_claims);
            self.claims.sort_by(canonical_claim_cmp);
        }

        // --- explicit relations: recompute for changed/removed docs and for
        // docs whose link target changed or disappeared ---
        let mut explicit_affected = HashSet::<String>::new();
        for doc_id in changed_ids.iter().chain(removed_set.iter()) {
            explicit_affected.insert((*doc_id).to_string());
        }
        for (source, target) in &self.supersedes_links {
            if changed_set.contains(target.as_str()) || removed_set.contains(target.as_str()) {
                explicit_affected.insert(source.clone());
            }
        }
        let is_stale_explicit = |relation: &SemanticRelation| {
            explicit_affected.contains(relation.source_doc_id.as_str())
                || changed_set.contains(relation.target_doc_id.as_str())
                || removed_set.contains(relation.target_doc_id.as_str())
        };
        // Stale relation ids are dropped from the merged list below; the
        // incremental merge never re-sorts the untouched corpus.
        let mut stale_relation_ids = HashSet::<String>::new();
        for relation in self
            .explicit_relations
            .iter()
            .filter(|r| is_stale_explicit(r))
        {
            affected_docs.insert(relation.source_doc_id.clone());
            affected_docs.insert(relation.target_doc_id.clone());
            stale_relation_ids.insert(relation.relation_id.clone());
        }
        self.explicit_relations.retain(|r| !is_stale_explicit(r));
        let mut recompute: Vec<&str> = explicit_affected
            .iter()
            .map(String::as_str)
            .filter(|id| docs_by_id.contains_key(*id))
            .collect();
        recompute.sort_unstable();
        let mut fresh_explicit = Vec::<SemanticRelation>::new();
        for doc_id in recompute {
            let doc = docs_by_id[doc_id];
            let Some((source_claim, target_claim)) =
                explicit_relation_claims(doc, &docs_by_id, &self.claims_by_doc, &self.claim_by_id)
            else {
                continue;
            };
            push_relation(
                &mut fresh_explicit,
                &mut HashSet::new(),
                &source_claim,
                &target_claim,
                SemanticRelationKind::Supersedes,
                1.0,
                "explicit",
                vec![format!(
                    "{} explicitly supersedes {}",
                    doc.doc_id, target_claim.source_doc_id
                )],
            );
            affected_docs.insert(source_claim.source_doc_id.clone());
            affected_docs.insert(target_claim.source_doc_id.clone());
        }
        self.explicit_relations
            .extend(fresh_explicit.iter().cloned());

        // --- chain relations: rebuild every affected key's chain ---
        // The rebuilt relations are spliced into the merged list below;
        // nothing else in the corpus is re-sorted.
        let mut affected_keys: Vec<RelationKey> = affected_keys.into_iter().collect();
        affected_keys.sort();
        let mut fresh_chain = Vec::<SemanticRelation>::new();
        for key in affected_keys {
            if let Some(old) = self.chain_relations.get(&key) {
                for relation in old {
                    affected_docs.insert(relation.source_doc_id.clone());
                    affected_docs.insert(relation.target_doc_id.clone());
                    stale_relation_ids.insert(relation.relation_id.clone());
                }
            }
            let chain = self.chains.get(&key).cloned().unwrap_or_default();
            if chain.is_empty() {
                self.chains.remove(&key);
            }
            let rebuilt = build_chain_relations(&chain, &self.claim_by_id, &docs_by_id);
            for relation in &rebuilt {
                affected_docs.insert(relation.source_doc_id.clone());
                affected_docs.insert(relation.target_doc_id.clone());
            }
            fresh_chain.extend(rebuilt.iter().cloned());
            if rebuilt.is_empty() {
                self.chain_relations.remove(&key);
            } else {
                self.chain_relations.insert(key, rebuilt);
            }
        }

        // --- merged relations: incremental canonical merge ---
        // `relations` is always `explicit_sorted ++ chain_sorted` (dedup by
        // relation id, explicit wins). Drop exactly the stale ids, then
        // sorted-merge the fresh relations into their runs: untouched
        // relations keep their positions, so this stays O(affected log
        // affected + corpus) instead of O(corpus log corpus).
        self.relations
            .retain(|r| !stale_relation_ids.contains(&r.relation_id));
        // Every explicit relation lands in the merged list exactly once
        // (unique ids, explicit first), so the merged vec starts with the
        // explicit run: locate its end via the prefix invariant.
        let explicit_ids: HashSet<&str> = self
            .explicit_relations
            .iter()
            .map(|r| r.relation_id.as_str())
            .collect();
        let boundary = self
            .relations
            .partition_point(|r| explicit_ids.contains(r.relation_id.as_str()));
        fresh_explicit.sort_by(explicit_relation_cmp);
        fresh_explicit.dedup_by(|a, b| a.relation_id == b.relation_id);
        // Chain relations defer to explicit ones on id collision, matching
        // the full rebuild's dedup order (a retained chain relation can never
        // share an id with a fresh explicit one: the changed docs' claims only
        // live in affected keys, whose old relations were all dropped above).
        fresh_chain.retain(|r| !explicit_ids.contains(r.relation_id.as_str()));
        fresh_chain.sort_by(chain_relation_cmp(&self.claim_by_id));
        fresh_chain.dedup_by(|a, b| a.relation_id == b.relation_id);
        let chain_run = self.relations.split_off(boundary);
        let explicit_run = std::mem::take(&mut self.relations);
        self.relations = merge_into_sorted_run(explicit_run, fresh_explicit, explicit_relation_cmp);
        let chain_run = merge_into_sorted_run(
            chain_run,
            fresh_chain,
            chain_relation_cmp(&self.claim_by_id),
        );
        self.relations.extend(chain_run);
        let mut relations_by_doc = HashMap::<&str, Vec<&SemanticRelation>>::new();
        for relation in &self.relations {
            relations_by_doc
                .entry(relation.source_doc_id.as_str())
                .or_default()
                .push(relation);
            if relation.target_doc_id != relation.source_doc_id {
                relations_by_doc
                    .entry(relation.target_doc_id.as_str())
                    .or_default()
                    .push(relation);
            }
        }
        for doc_id in &affected_docs {
            if removed_set.contains(doc_id.as_str()) {
                continue;
            }
            if !docs_by_id.contains_key(doc_id.as_str()) {
                self.document_states.remove(doc_id);
                continue;
            }
            let state = document_state_for(
                doc_id,
                relations_by_doc.get(doc_id.as_str()).map(Vec::as_slice),
                &docs_by_id,
                &self.claims_by_doc,
                &self.claim_by_id,
                options,
            );
            self.document_states.insert(doc_id.clone(), state);
        }
        self.initialized = true;
        Ok(())
    }

    pub fn claims(&self) -> &[SemanticClaim] {
        &self.claims
    }

    pub fn relations(&self) -> &[SemanticRelation] {
        &self.relations
    }

    pub fn document_state(&self, doc_id: &str) -> DocumentSemanticState {
        self.document_states
            .get(doc_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Canonical claim order: (effective date, doc id, claim id). Shared by the
/// full rebuild and incremental updates so both produce identical chains.
fn canonical_claim_cmp(a: &SemanticClaim, b: &SemanticClaim) -> Ordering {
    claim_date(a)
        .cmp(&claim_date(b))
        .then_with(|| a.source_doc_id.cmp(&b.source_doc_id))
        .then_with(|| a.claim_id.cmp(&b.claim_id))
}

fn relation_key(claim: &SemanticClaim) -> RelationKey {
    (
        claim.scope.clone(),
        normalize(&claim.subject),
        normalize(&claim.predicate),
    )
}

/// The claim of `doc_id` that the full rebuild would pick first: the minimum
/// in canonical order, mirroring `claims.iter().find(...)` on the sorted vec.
fn first_claim<'a>(
    doc_id: &str,
    claims_by_doc: &HashMap<String, Vec<String>>,
    claim_by_id: &'a HashMap<String, SemanticClaim>,
) -> Option<&'a SemanticClaim> {
    claims_by_doc
        .get(doc_id)?
        .iter()
        .filter_map(|id| claim_by_id.get(id))
        .min_by(|a, b| canonical_claim_cmp(a, b))
}

/// The (source claim, target claim) backing a doc's explicit `supersedes_id`
/// link, or `None` when the link yields no relation (missing target or scope
/// mismatch). Mirrors the explicit-link pass of the full rebuild.
fn explicit_relation_claims(
    doc: &SourceDocument,
    docs_by_id: &HashMap<&str, &SourceDocument>,
    claims_by_doc: &HashMap<String, Vec<String>>,
    claim_by_id: &HashMap<String, SemanticClaim>,
) -> Option<(SemanticClaim, SemanticClaim)> {
    let target_id = doc.filters.get("supersedes_id")?;
    let target = docs_by_id.get(target_id.as_str())?;
    if semantic_scope(doc) != semantic_scope(target) {
        return None;
    }
    let source_claim = first_claim(&doc.doc_id, claims_by_doc, claim_by_id)
        .cloned()
        .unwrap_or_else(|| document_claim(doc));
    let target_claim = first_claim(&target.doc_id, claims_by_doc, claim_by_id)
        .cloned()
        .unwrap_or_else(|| document_claim(target));
    Some((source_claim, target_claim))
}

/// Decides the relation between a claim and its predecessor on the same
/// canonical-claim chain. Extracted verbatim from the full rebuild so both
/// paths share the decision logic.
fn push_chain_pair_relation(
    relations: &mut Vec<SemanticRelation>,
    seen: &mut HashSet<String>,
    claim: &SemanticClaim,
    previous: &SemanticClaim,
    docs_by_id: &HashMap<&str, &SourceDocument>,
) {
    if normalize(&previous.object) == normalize(&claim.object) {
        push_relation(
            relations,
            seen,
            claim,
            previous,
            SemanticRelationKind::Confirms,
            0.95,
            "canonical_claim",
            vec!["same canonical claim and value".to_string()],
        );
        return;
    }
    let source_doc = docs_by_id.get(claim.source_doc_id.as_str()).copied();
    let target_doc = docs_by_id.get(previous.source_doc_id.as_str()).copied();
    let direct_correction = source_doc.is_some_and(|doc| has_correction_cue(&doc.content));
    let chronological = claim_date(claim)
        .zip(claim_date(previous))
        .is_some_and(|(newer, older)| newer > older);
    let same_source_kind = source_doc
        .zip(target_doc)
        .is_some_and(|(source, target)| source_kind(source) == source_kind(target));
    let (kind, confidence, method) = if direct_correction {
        (SemanticRelationKind::Supersedes, 0.95, "correction_cue")
    } else if chronological && same_source_kind {
        (
            SemanticRelationKind::Supersedes,
            0.90,
            "canonical_claim_and_time",
        )
    } else {
        (
            SemanticRelationKind::ConflictsWith,
            0.75,
            "canonical_claim_conflict",
        )
    };
    push_relation(
        relations,
        seen,
        claim,
        previous,
        kind,
        confidence,
        method,
        vec![format!(
            "{} changed from '{}' to '{}'",
            claim.subject, previous.object, claim.object
        )],
    );
}

/// Rebuilds the relations for one canonical-claim chain: consecutive claims in
/// canonical order (from different docs) relate to each other, exactly as the
/// full rebuild's per-key pass does.
fn build_chain_relations(
    chain: &[String],
    claim_by_id: &HashMap<String, SemanticClaim>,
    docs_by_id: &HashMap<&str, &SourceDocument>,
) -> Vec<SemanticRelation> {
    let mut relations = Vec::new();
    let mut seen = HashSet::new();
    let mut latest: Option<&SemanticClaim> = None;
    for claim_id in chain {
        let Some(claim) = claim_by_id.get(claim_id) else {
            continue;
        };
        if let Some(previous) = latest {
            if previous.source_doc_id != claim.source_doc_id {
                push_chain_pair_relation(&mut relations, &mut seen, claim, previous, docs_by_id);
            }
        }
        latest = Some(claim);
    }
    relations
}

/// Canonical order key for a relation's source claim: (claim date, source
/// doc id, source claim id). Shared by the full-rebuild merge and the
/// incremental merge so both produce the identical global order.
fn source_claim_key<'a>(
    relation: &'a SemanticRelation,
    claim_by_id: &'a HashMap<String, SemanticClaim>,
) -> (Option<chrono::NaiveDate>, &'a str, &'a str) {
    match claim_by_id.get(&relation.source_claim_id) {
        Some(claim) => (
            claim_date(claim),
            claim.source_doc_id.as_str(),
            claim.claim_id.as_str(),
        ),
        // Transient `document_claim` fallbacks never back chain relations;
        // sort them stably by id if one ever appears here.
        None => (None, "", relation.source_claim_id.as_str()),
    }
}

/// Canonical order for explicit relations: by source doc, then target doc,
/// then relation id. Shared by the full-rebuild merge and the incremental
/// merge so both produce the identical global order.
fn explicit_relation_cmp(a: &SemanticRelation, b: &SemanticRelation) -> Ordering {
    a.source_doc_id
        .cmp(&b.source_doc_id)
        .then_with(|| a.target_doc_id.cmp(&b.target_doc_id))
        .then_with(|| a.relation_id.cmp(&b.relation_id))
}

/// Canonical order for chain relations: by source-claim canonical order,
/// then relation id. Shared by the full-rebuild merge and the incremental
/// merge so both produce the identical global order.
fn chain_relation_cmp(
    claim_by_id: &HashMap<String, SemanticClaim>,
) -> impl Fn(&SemanticRelation, &SemanticRelation) -> Ordering + '_ {
    move |a, b| {
        source_claim_key(a, claim_by_id)
            .cmp(&source_claim_key(b, claim_by_id))
            .then_with(|| a.relation_id.cmp(&b.relation_id))
    }
}

/// Splices `fresh` (sorted by `cmp`) into `run` (sorted by `cmp`), keeping
/// the merged vec sorted. Lets the incremental update insert rebuilt
/// relations into the canonical merged list without re-sorting the
/// untouched corpus.
fn merge_into_sorted_run(
    run: Vec<SemanticRelation>,
    fresh: Vec<SemanticRelation>,
    cmp: impl Fn(&SemanticRelation, &SemanticRelation) -> Ordering,
) -> Vec<SemanticRelation> {
    if fresh.is_empty() {
        return run;
    }
    if run.is_empty() {
        return fresh;
    }
    let mut merged = Vec::with_capacity(run.len() + fresh.len());
    let mut run = run.into_iter().peekable();
    let mut fresh = fresh.into_iter().peekable();
    loop {
        match (run.peek(), fresh.peek()) {
            (Some(r), Some(f)) => {
                if cmp(r, f) == Ordering::Less {
                    merged.push(run.next().expect("peeked run item"));
                } else {
                    merged.push(fresh.next().expect("peeked fresh item"));
                }
            }
            (Some(_), None) => {
                merged.extend(run);
                break;
            }
            (None, Some(_)) => {
                merged.extend(fresh);
                break;
            }
            (None, None) => break,
        }
    }
    merged
}

/// Merges explicit and chain relations into the canonical global order:
/// explicit relations first (by source doc, then target doc), then chain
/// relations (by source-claim canonical order). Duplicate relation ids resolve
/// in favor of the earlier group, matching the full rebuild's push order.
///
/// Invariant: the returned vec is always `explicit_sorted ++ chain_sorted`.
/// [`SemanticRelationStore::update_documents`] relies on this prefix structure
/// to splice rebuilt relations in without re-sorting the corpus.
fn merge_relations_canonical(
    explicit: &[SemanticRelation],
    chain_relations: &HashMap<RelationKey, Vec<SemanticRelation>>,
    claim_by_id: &HashMap<String, SemanticClaim>,
) -> Vec<SemanticRelation> {
    let mut merged = Vec::new();
    let mut seen = HashSet::new();
    let mut explicit_sorted: Vec<&SemanticRelation> = explicit.iter().collect();
    explicit_sorted.sort_by(|a, b| explicit_relation_cmp(a, b));
    for relation in explicit_sorted {
        if seen.insert(relation.relation_id.clone()) {
            merged.push(relation.clone());
        }
    }
    let mut chained: Vec<&SemanticRelation> = chain_relations.values().flatten().collect();
    let chain_cmp = chain_relation_cmp(claim_by_id);
    chained.sort_by(|a, b| chain_cmp(a, b));
    for relation in chained {
        if seen.insert(relation.relation_id.clone()) {
            merged.push(relation.clone());
        }
    }
    merged
}

/// Recomputes one document's semantic state from the relations involving it
/// (in canonical global order). Equivalent to the per-document slice of
/// [`build_document_states`].
fn document_state_for(
    doc_id: &str,
    relations: Option<&[&SemanticRelation]>,
    docs_by_id: &HashMap<&str, &SourceDocument>,
    claims_by_doc: &HashMap<String, Vec<String>>,
    claim_by_id: &HashMap<String, SemanticClaim>,
    options: SupersessionOptions,
) -> DocumentSemanticState {
    let mut state = DocumentSemanticState {
        status: Some(SemanticStatus::Current),
        ..DocumentSemanticState::default()
    };
    let Some(relations) = relations else {
        return state;
    };
    for relation in relations {
        match relation.kind {
            SemanticRelationKind::Supersedes
                if relation.confidence >= options.suppress_confidence
                    && relation.target_doc_id == doc_id =>
            {
                let only_target_claim = docs_by_id.get(doc_id).is_some_and(|target_doc| {
                    document_contains_only_target_claim(
                        target_doc,
                        claims_by_doc.get(doc_id),
                        claim_by_id,
                        relation,
                    )
                });
                if relation.method != "explicit" && !only_target_claim {
                    if state.status != Some(SemanticStatus::Superseded) {
                        state.status = Some(SemanticStatus::Conflicted);
                        state.relation_confidence = Some(relation.confidence);
                        state.evidence = relation.evidence.clone();
                    }
                    continue;
                }
                if state.relation_confidence.unwrap_or_default() <= relation.confidence {
                    state.status = Some(SemanticStatus::Superseded);
                    state.superseded_by = Some(relation.source_doc_id.clone());
                    state.relation_confidence = Some(relation.confidence);
                    state.evidence = relation.evidence.clone();
                }
            }
            SemanticRelationKind::ConflictsWith
                if relation.confidence >= options.conflict_confidence
                    && (relation.source_doc_id == doc_id || relation.target_doc_id == doc_id) =>
            {
                if state.status != Some(SemanticStatus::Superseded) {
                    state.status = Some(SemanticStatus::Conflicted);
                    state.relation_confidence = Some(relation.confidence);
                    state.evidence = relation.evidence.clone();
                }
            }
            _ => {}
        }
    }
    state
}

fn build_document_states(
    documents: &[&SourceDocument],
    relations: &[SemanticRelation],
    docs_by_id: &HashMap<&str, &SourceDocument>,
    claims_by_doc: &HashMap<String, Vec<String>>,
    claim_by_id: &HashMap<String, SemanticClaim>,
    options: SupersessionOptions,
) -> HashMap<String, DocumentSemanticState> {
    // Index relations by involved doc, preserving canonical global order, so
    // each document's state is computed from exactly the relations the
    // single-pass version would apply to it, in the same order.
    let mut by_doc = HashMap::<&str, Vec<&SemanticRelation>>::new();
    for relation in relations {
        by_doc
            .entry(relation.source_doc_id.as_str())
            .or_default()
            .push(relation);
        if relation.target_doc_id != relation.source_doc_id {
            by_doc
                .entry(relation.target_doc_id.as_str())
                .or_default()
                .push(relation);
        }
    }
    documents
        .iter()
        .map(|doc| {
            let state = document_state_for(
                doc.doc_id.as_str(),
                by_doc.get(doc.doc_id.as_str()).map(Vec::as_slice),
                docs_by_id,
                claims_by_doc,
                claim_by_id,
                options,
            );
            (doc.doc_id.clone(), state)
        })
        .collect()
}

fn document_contains_only_target_claim(
    target_doc: &SourceDocument,
    target_claim_ids: Option<&Vec<String>>,
    claim_by_id: &HashMap<String, SemanticClaim>,
    relation: &SemanticRelation,
) -> bool {
    let Some(ids) = target_claim_ids else {
        return false;
    };
    if ids.len() != 1 || ids[0] != relation.target_claim_id {
        return false;
    }
    let Some(target_claim) = claim_by_id.get(&ids[0]) else {
        return false;
    };
    normalize(&claim_bearing_document_content(&target_doc.content))
        == normalize(&target_claim.evidence)
}

fn claim_bearing_document_content(content: &str) -> String {
    content
        .lines()
        .filter(|line| !is_markdown_heading_line(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_markdown_heading_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    let hashes = trimmed.bytes().take_while(|byte| *byte == b'#').count();
    (1..=6).contains(&hashes)
        && trimmed
            .as_bytes()
            .get(hashes)
            .is_some_and(u8::is_ascii_whitespace)
}

#[allow(clippy::too_many_arguments)]
fn push_relation(
    relations: &mut Vec<SemanticRelation>,
    seen: &mut HashSet<String>,
    source: &SemanticClaim,
    target: &SemanticClaim,
    kind: SemanticRelationKind,
    confidence: f32,
    method: &str,
    evidence: Vec<String>,
) {
    let relation_id = format!("{}::{:?}::{}", source.claim_id, kind, target.claim_id);
    if !seen.insert(relation_id.clone()) {
        return;
    }
    relations.push(SemanticRelation {
        relation_id,
        kind,
        source_claim_id: source.claim_id.clone(),
        target_claim_id: target.claim_id.clone(),
        source_doc_id: source.source_doc_id.clone(),
        target_doc_id: target.source_doc_id.clone(),
        confidence,
        method: method.to_string(),
        evidence,
        scope: source.scope.clone(),
    });
}

fn extract_claims(doc: &SourceDocument) -> Vec<SemanticClaim> {
    let mut claims = Vec::new();
    for sentence in sentences(&doc.content) {
        if let Some((subject, predicate, object)) = ownership_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
        if let Some((subject, predicate, object)) = assignment_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
        if let Some((subject, predicate, object)) = usage_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
        if let Some((subject, predicate, object)) = configuration_claim(sentence) {
            claims.push(make_claim_in_scope(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
                scalar_configuration_scope(doc),
            ));
        }
    }
    dedupe_claims(&mut claims);
    claims
}

fn document_claim(doc: &SourceDocument) -> SemanticClaim {
    make_claim(
        doc,
        &doc.content.chars().take(240).collect::<String>(),
        &doc.concept,
        "document_state",
        &doc.doc_id,
        0,
    )
}

fn make_claim(
    doc: &SourceDocument,
    evidence: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    index: usize,
) -> SemanticClaim {
    make_claim_in_scope(
        doc,
        evidence,
        subject,
        predicate,
        object,
        index,
        semantic_scope(doc),
    )
}

fn make_claim_in_scope(
    doc: &SourceDocument,
    evidence: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    index: usize,
    scope: String,
) -> SemanticClaim {
    let subject = clean_phrase(subject);
    let object = clean_phrase(object);
    SemanticClaim {
        claim_id: format!("{}::claim-{}", doc.doc_id, index),
        subject,
        predicate: predicate.to_string(),
        object,
        source_doc_id: doc.doc_id.clone(),
        evidence: evidence.trim().to_string(),
        effective_at: doc.timestamp.clone(),
        scope,
        confidence: 0.85,
    }
}

fn ownership_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static OWNS: OnceLock<Regex> = OnceLock::new();
    static RESPONSIBLE: OnceLock<Regex> = OnceLock::new();
    static OWNED_BY: OnceLock<Regex> = OnceLock::new();
    let owns = OWNS.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,60}?)\s+owns\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid ownership regex")
    });
    if let Some(caps) = owns.captures(sentence) {
        return Some((caps[2].to_string(), "owner", caps[1].to_string()));
    }
    let responsible = RESPONSIBLE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,60}?)\s+is\s+(?:now\s+)?responsible\s+for\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid responsibility regex")
    });
    if let Some(caps) = responsible.captures(sentence) {
        return Some((caps[2].to_string(), "owner", caps[1].to_string()));
    }
    let owned_by = OWNED_BY.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+is\s+(?:now\s+)?owned\s+by\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,60})")
            .expect("valid owned-by regex")
    });
    owned_by
        .captures(sentence)
        .map(|caps| (caps[1].to_string(), "owner", caps[2].to_string()))
}

fn assignment_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static ASSIGNED: OnceLock<Regex> = OnceLock::new();
    let assigned = ASSIGNED.get_or_init(|| {
        Regex::new(r"(?i)assigned\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+to\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,60})")
            .expect("valid assignment regex")
    });
    assigned
        .captures(sentence)
        .map(|caps| (caps[1].to_string(), "owner", caps[2].to_string()))
}

fn usage_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static USE_FOR: OnceLock<Regex> = OnceLock::new();
    static USES: OnceLock<Regex> = OnceLock::new();
    let use_for = USE_FOR.get_or_init(|| {
        Regex::new(r"(?i)(?:use|adopt|choose|selected)\s+(?:the\s+)?([a-z][a-z0-9 _/.-]{1,60}?)\s+for\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid use-for regex")
    });
    if let Some(caps) = use_for.captures(sentence) {
        return Some((caps[2].to_string(), "implementation", caps[1].to_string()));
    }
    let uses = USES.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+uses\s+(?:the\s+)?([a-z][a-z0-9 _/.-]{1,60})")
            .expect("valid uses regex")
    });
    uses.captures(sentence)
        .map(|caps| (caps[1].to_string(), "implementation", caps[2].to_string()))
}

fn configuration_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static SCALAR_CONFIGURATION: OnceLock<Regex> = OnceLock::new();
    let scalar = SCALAR_CONFIGURATION.get_or_init(|| {
        Regex::new(
            r"(?i)^(?:[-*]\s*)?(?:the\s+)?([a-z][a-z0-9 _/.-]{1,100}?)\s*(?::|=)\s*(-?\d+(?:\.\d+)?(?:\s*(?:ms|s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hours?|%|kb|mb|gb|tb))?|true|false|enabled|disabled)$",
        )
        .expect("valid scalar configuration regex")
    });
    scalar
        .captures(sentence)
        .map(|caps| (caps[1].to_string(), "value", caps[2].to_string()))
}

fn sentences(content: &str) -> Vec<&str> {
    content
        .split(['\n', '.', '!', '?'])
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

fn clean_phrase(value: &str) -> String {
    let value = value
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .to_lowercase();
    let value = value
        .strip_prefix("the ")
        .or_else(|| value.strip_prefix("current "))
        .or_else(|| value.strip_prefix("new "))
        .unwrap_or(&value);
    value.trim().to_string()
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !matches!(*token, "the" | "a" | "an" | "current" | "new" | "old"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedupe_claims(claims: &mut Vec<SemanticClaim>) {
    let mut seen = HashSet::new();
    claims.retain(|claim| {
        !claim.subject.is_empty()
            && !claim.object.is_empty()
            && seen.insert((
                normalize(&claim.subject),
                claim.predicate.clone(),
                normalize(&claim.object),
            ))
    });
}

fn semantic_scope(doc: &SourceDocument) -> String {
    doc.filters
        .get("semantic_scope")
        .map(|scope| format!("scope:{scope}"))
        .or_else(|| {
            doc.filters
                .get("memory_user_id")
                .map(|user| format!("user:{user}"))
        })
        .unwrap_or_else(|| "store".to_string())
}

fn scalar_configuration_scope(doc: &SourceDocument) -> String {
    let base = semantic_scope(doc);
    if doc.filters.contains_key("semantic_scope") {
        return base;
    }
    if let Some(group_id) = doc.group_id.as_deref() {
        let group = normalize(group_id);
        if !group.is_empty() {
            return format!("{base}::group:{group}");
        }
    }
    if source_kind(doc) == "document" {
        let parent = document_parent_scope(doc);
        if let Some(heading) = doc.headings.iter().find_map(|heading| {
            let heading = normalize(heading);
            (!heading.is_empty() && !is_generic_scalar_heading(&heading)).then_some(heading)
        }) {
            return format!("{base}::parent:{parent}::heading:{heading}");
        }
        let concept = normalize(&doc.concept);
        if !concept.is_empty() {
            return format!("{base}::parent:{parent}::concept:{concept}");
        }
    }
    format!("{base}::doc:{}", doc.doc_id)
}

fn document_parent_scope(doc: &SourceDocument) -> String {
    doc.source
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .or_else(|| doc.source.rsplit_once('\\').map(|(parent, _)| parent))
        .map(normalize)
        .unwrap_or_default()
}

fn is_generic_scalar_heading(heading: &str) -> bool {
    matches!(
        heading,
        "config"
            | "configuration"
            | "setting"
            | "settings"
            | "policy"
            | "decision"
            | "defaults"
            | "parameter"
            | "parameters"
            | "option"
            | "options"
    )
}

fn source_kind(doc: &SourceDocument) -> &'static str {
    if doc.source.starts_with("codex://")
        || doc.source.starts_with("claude://")
        || doc.source.starts_with("claude-code://")
        || doc.source.starts_with("gemini-cli://")
        || doc.source.starts_with("agy://")
        || doc.source.starts_with("lint-ai://")
        || doc.filters.contains_key("document_type")
    {
        "memory"
    } else {
        "document"
    }
}

fn has_correction_cue(content: &str) -> bool {
    let lower = content.to_lowercase();
    [
        "supersedes",
        "replaces",
        "instead of",
        "no longer",
        "changed from",
        "moved from",
        "previously",
        "formerly",
    ]
    .iter()
    .any(|cue| lower.contains(cue))
}

fn claim_date(claim: &SemanticClaim) -> Option<chrono::NaiveDate> {
    claim
        .effective_at
        .as_deref()
        .and_then(|value| parse_temporal_date(Some(value)))
}

pub fn is_historical_query(query: &str) -> bool {
    let lower = query.to_lowercase();
    [
        "history",
        "historical",
        "previous",
        "previously",
        "formerly",
        "superseded",
        "what changed",
        "why did we change",
        "what was true before",
        "who owned this before",
        "who was responsible before",
        "before the change",
        "before we changed",
        "timeline",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn doc(id: &str, content: &str, timestamp: Option<&str>, source: &str) -> SourceDocument {
        SourceDocument {
            doc_id: id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            concept: "decision".to_string(),
            group_id: None,
            headings: vec!["Decision".to_string()],
            links: vec![],
            timestamp: timestamp.map(str::to_string),
            doc_length: content.len(),
            author_agent: None,
            filters: BTreeMap::new(),
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    #[test]
    fn newer_same_kind_ownership_claim_supersedes_old_claim() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn timestamp_only_cross_source_change_is_a_visible_conflict() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "codex://project/session/outcome",
        );
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Conflicted)
        );
        assert_eq!(
            store.document_state("new").status,
            Some(SemanticStatus::Conflicted)
        );
    }

    #[test]
    fn explicit_supersession_respects_user_scope() {
        let mut old = doc("old", "old decision", None, "memory://old");
        old.filters.insert("memory_user_id".into(), "a".into());
        let mut new = doc("new", "new decision", None, "memory://new");
        new.filters.insert("memory_user_id".into(), "b".into());
        new.filters.insert("supersedes_id".into(), "old".into());
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_ne!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
    }

    #[test]
    fn explicit_supersession_without_extractable_claims_is_authoritative() {
        let old = doc(
            "old",
            "Original architecture decision.",
            None,
            "docs/old.md",
        );
        let mut new = doc("new", "Updated architecture decision.", None, "docs/new.md");
        new.filters.insert("supersedes_id".into(), "old".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        let relation = store
            .relations()
            .iter()
            .find(|relation| relation.kind == SemanticRelationKind::Supersedes)
            .expect("explicit metadata should create a supersession relation");
        assert_eq!(relation.method, "explicit");
        assert_eq!(relation.confidence, 1.0);
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn explicit_document_supersession_still_hides_a_multi_claim_document() {
        let old = doc(
            "old",
            concat!(
                "The Platform POD owns valve control. ",
                "The deployment strategy uses blue-green releases."
            ),
            None,
            "docs/old.md",
        );
        let mut new = doc("new", "Updated architecture decision.", None, "docs/new.md");
        new.filters.insert("supersedes_id".into(), "old".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn correction_language_can_supersede_without_timestamps() {
        let old = doc(
            "a-old",
            "The National Park POD owns valve control.",
            None,
            "docs/old.md",
        );
        let new = doc(
            "z-new",
            "This replaces the previous decision. The Controls POD owns valve control.",
            None,
            "docs/new.md",
        );

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("a-old").status,
            Some(SemanticStatus::Superseded)
        );
        assert!(store.relations().iter().any(|relation| {
            relation.kind == SemanticRelationKind::Supersedes && relation.method == "correction_cue"
        }));
    }

    #[test]
    fn changed_claim_without_time_or_correction_stays_visible_as_conflict() {
        let old = doc(
            "a-old",
            "The National Park POD owns valve control.",
            None,
            "docs/old.md",
        );
        let new = doc(
            "z-new",
            "The Controls POD owns valve control.",
            None,
            "docs/new.md",
        );

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("a-old").status,
            Some(SemanticStatus::Conflicted)
        );
        assert_eq!(
            store.document_state("z-new").status,
            Some(SemanticStatus::Conflicted)
        );
        assert!(store
            .relations()
            .iter()
            .all(|relation| relation.kind != SemanticRelationKind::Supersedes));
    }

    #[test]
    fn equivalent_wording_confirms_the_existing_claim() {
        let first = doc(
            "first",
            "The Controls POD owns valve control.",
            Some("2026-01-01"),
            "docs/first.md",
        );
        let second = doc(
            "second",
            "Valve control is owned by the Controls POD.",
            Some("2026-02-01"),
            "docs/second.md",
        );

        let store = SemanticRelationStore::from_documents(
            [&first, &second],
            SupersessionOptions::default(),
        );

        assert!(store
            .relations()
            .iter()
            .any(|relation| relation.kind == SemanticRelationKind::Confirms));
        assert_eq!(
            store.document_state("first").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("second").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn semantic_scope_prevents_automatic_cross_project_relations() {
        let mut old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        old.filters
            .insert("semantic_scope".into(), "project-a".into());
        let mut new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        new.filters
            .insert("semantic_scope".into(), "project-b".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert!(store.relations().is_empty());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("new").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn disabled_supersession_produces_no_claims_relations_or_states() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let options = SupersessionOptions {
            enabled: false,
            suppress_confidence: 1.2,
            conflict_confidence: -0.1,
        };

        let store = SemanticRelationStore::from_documents([&old, &new], options);

        assert!(store.claims().is_empty());
        assert!(store.relations().is_empty());
        assert_eq!(
            store.document_state("old"),
            DocumentSemanticState::default()
        );
    }

    #[test]
    fn enabled_supersession_rejects_invalid_thresholds() {
        let document = doc(
            "decision",
            "The Controls POD owns valve control.",
            None,
            "doc.md",
        );
        let options = SupersessionOptions {
            suppress_confidence: 1.2,
            ..SupersessionOptions::default()
        };

        let error = SemanticRelationStore::try_from_documents([&document], options)
            .expect_err("enabled supersession must reject invalid thresholds");

        assert!(error
            .to_string()
            .contains("confidence thresholds must be between 0 and 1"));
    }

    #[test]
    fn suppression_threshold_can_keep_detected_relation_visible() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let options = SupersessionOptions {
            suppress_confidence: 0.95,
            ..SupersessionOptions::default()
        };

        let store = SemanticRelationStore::from_documents([&old, &new], options);

        assert!(store
            .relations()
            .iter()
            .any(|relation| relation.kind == SemanticRelationKind::Supersedes));
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn historical_query_detection_covers_history_but_not_current_state() {
        for query in [
            "show the decision history",
            "what changed about ownership",
            "who owned this before",
            "show the superseded guidance",
            "ownership timeline",
        ] {
            assert!(
                is_historical_query(query),
                "expected historical query: {query}"
            );
        }
        for query in [
            "who owns valve control now",
            "what must happen before deployment",
            "validate inputs before saving",
            "check authorization before updating status",
        ] {
            assert!(
                !is_historical_query(query),
                "expected current-state query: {query}"
            );
        }
    }
}

#[cfg(test)]
mod scalar_configuration_supersession_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn scalar_doc(id: &str, content: &str, timestamp: &str) -> SourceDocument {
        SourceDocument {
            doc_id: id.to_string(),
            source: format!("docs/{id}.md"),
            content: content.to_string(),
            concept: "gateway retry policy".to_string(),
            group_id: None,
            filters: BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: Some(timestamp.to_string()),
            doc_length: content.len(),
            author_agent: None,
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    #[test]
    fn unrelated_scalar_configuration_domains_do_not_supersede_each_other() {
        let mut payments = scalar_doc("payments", "Retry attempts: 5.", "2026-01-01");
        payments.concept = "payments".to_string();
        payments.headings = vec!["Payments Retry Policy".to_string()];

        let mut worker = scalar_doc("image-worker", "Retry attempts: 2.", "2026-06-01");
        worker.concept = "image worker".to_string();
        worker.headings = vec!["Image Worker Retry Policy".to_string()];

        let store = SemanticRelationStore::from_documents(
            [&payments, &worker],
            SupersessionOptions::default(),
        );
        assert_eq!(
            store.document_state("payments").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("image-worker").status,
            Some(SemanticStatus::Current)
        );
        assert!(!store.relations().iter().any(|relation| {
            relation.kind == SemanticRelationKind::Supersedes
                && relation.source_doc_id == "image-worker"
                && relation.target_doc_id == "payments"
        }));
    }

    #[test]
    fn same_filename_concept_in_different_directories_does_not_collide() {
        let mut payments = scalar_doc("payments-config", "Retry attempts: 5.", "2026-01-01");
        payments.source = "services/payments/config.md".to_string();
        payments.concept = "config".to_string();
        payments.headings.clear();

        let mut worker = scalar_doc("worker-config", "Retry attempts: 2.", "2026-06-01");
        worker.source = "services/image-worker/config.md".to_string();
        worker.concept = "config".to_string();
        worker.headings.clear();

        let store = SemanticRelationStore::from_documents(
            [&payments, &worker],
            SupersessionOptions::default(),
        );
        assert_eq!(
            store.document_state("payments-config").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("worker-config").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn generic_headings_in_different_directories_do_not_collide() {
        let mut payments = scalar_doc(
            "payments-config",
            "# Configuration\n\nTimeout: 30s.",
            "2026-01-01",
        );
        payments.source = "services/payments/config.md".to_string();
        payments.concept = "config".to_string();
        payments.headings = vec!["Configuration".to_string()];

        let mut worker = scalar_doc(
            "worker-config",
            "# Configuration\n\nTimeout: 10s.",
            "2026-06-01",
        );
        worker.source = "services/image-worker/config.md".to_string();
        worker.concept = "config".to_string();
        worker.headings = vec!["Configuration".to_string()];

        let store = SemanticRelationStore::from_documents(
            [&payments, &worker],
            SupersessionOptions::default(),
        );
        assert_eq!(
            store.document_state("payments-config").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("worker-config").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn same_specific_heading_in_different_directories_does_not_collide() {
        let mut payments = scalar_doc(
            "payments-retry",
            "# Retry Policy\n\nRetry attempts: 5.",
            "2026-01-01",
        );
        payments.source = "services/payments/retry.md".to_string();
        payments.headings = vec!["Retry Policy".to_string()];

        let mut worker = scalar_doc(
            "worker-retry",
            "# Retry Policy\n\nRetry attempts: 2.",
            "2026-06-01",
        );
        worker.source = "services/image-worker/retry.md".to_string();
        worker.headings = vec!["Retry Policy".to_string()];

        let store = SemanticRelationStore::from_documents(
            [&payments, &worker],
            SupersessionOptions::default(),
        );
        assert_eq!(
            store.document_state("payments-retry").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("worker-retry").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn same_heading_scalar_configuration_still_supersedes_by_time() {
        let mut old = scalar_doc(
            "decision-a",
            "# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.",
            "2026-01-01",
        );
        old.headings = vec!["Gateway Retry Policy".to_string()];
        old.concept = "decision a".to_string();

        let mut new = scalar_doc(
            "decision-b",
            "# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.",
            "2026-06-01",
        );
        new.headings = vec!["Gateway Retry Policy".to_string()];
        new.concept = "decision b".to_string();

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("decision-a").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("decision-a").superseded_by.as_deref(),
            Some("decision-b")
        );
    }

    #[test]
    fn non_heading_extra_content_prevents_whole_document_suppression() {
        let mut old = scalar_doc(
            "decision-a",
            "# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.\nAdditional rationale remains relevant.",
            "2026-01-01",
        );
        old.headings = vec!["Gateway Retry Policy".to_string()];
        let mut new = scalar_doc(
            "decision-b",
            "# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.",
            "2026-06-01",
        );
        new.headings = vec!["Gateway Retry Policy".to_string()];

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("decision-a").status,
            Some(SemanticStatus::Conflicted)
        );
    }

    #[test]
    fn newer_scalar_configuration_claim_supersedes_older_value_by_time() {
        let old = scalar_doc(
            "decision-a",
            "Gateway timeout retry attempts: 5.",
            "2026-01-01",
        );
        let new = scalar_doc(
            "decision-b",
            "Gateway timeout retry attempts: 2.",
            "2026-06-01",
        );
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("decision-a").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("decision-a").superseded_by.as_deref(),
            Some("decision-b")
        );
        assert!(store.relations().iter().any(|relation| {
            relation.kind == SemanticRelationKind::Supersedes
                && relation.method == "canonical_claim_and_time"
                && (relation.confidence - 0.90).abs() < f32::EPSILON
        }));
    }
}

#[cfg(test)]
mod incremental_update_tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    fn sdoc(id: &str, content: &str, timestamp: Option<&str>) -> SourceDocument {
        SourceDocument {
            doc_id: id.to_string(),
            source: format!("docs/{id}.md"),
            content: content.to_string(),
            concept: "decision".to_string(),
            group_id: None,
            headings: vec![],
            links: vec![],
            timestamp: timestamp.map(str::to_string),
            doc_length: content.len(),
            author_agent: None,
            filters: BTreeMap::new(),
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    fn sdoc_with_source(
        id: &str,
        content: &str,
        timestamp: Option<&str>,
        source: &str,
    ) -> SourceDocument {
        let mut doc = sdoc(id, content, timestamp);
        doc.source = source.to_string();
        doc
    }

    fn full_build(docs: &HashMap<String, SourceDocument>) -> SemanticRelationStore {
        SemanticRelationStore::try_from_documents(docs.values(), SupersessionOptions::default())
            .expect("full rebuild oracle must succeed")
    }

    /// Asserts the incrementally-updated store is identical to a fresh full
    /// rebuild: same claims, same relations (same canonical order), same
    /// per-document states.
    fn assert_store_eq(
        full: &SemanticRelationStore,
        incremental: &SemanticRelationStore,
        doc_ids: &[&str],
    ) {
        assert_eq!(full.claims(), incremental.claims(), "claims diverged");
        assert_eq!(
            full.relations(),
            incremental.relations(),
            "relations diverged"
        );
        for doc_id in doc_ids {
            assert_eq!(
                full.document_state(doc_id),
                incremental.document_state(doc_id),
                "document state diverged for {doc_id}"
            );
        }
    }

    fn apply(
        store: &mut SemanticRelationStore,
        docs: &HashMap<String, SourceDocument>,
        changed_ids: &[&str],
        removed_ids: &[&str],
    ) {
        let changed: Vec<&SourceDocument> = changed_ids
            .iter()
            .map(|id| docs.get(*id).expect("changed doc must exist"))
            .collect();
        let removed: Vec<String> = removed_ids.iter().map(|s| s.to_string()).collect();
        store
            .update_documents(docs, &changed, &removed, SupersessionOptions::default())
            .expect("incremental update must succeed");
    }

    fn check(
        store: &mut SemanticRelationStore,
        docs: &HashMap<String, SourceDocument>,
        changed_ids: &[&str],
        removed_ids: &[&str],
        all_ids: &[&str],
    ) {
        apply(store, docs, changed_ids, removed_ids);
        assert_store_eq(&full_build(docs), store, all_ids);
    }

    fn chain_corpus() -> HashMap<String, SourceDocument> {
        [
            (
                "a",
                "The National Park POD owns valve control.",
                Some("2026-01-01"),
            ),
            (
                "b",
                "The Platform POD owns valve control.",
                Some("2026-06-01"),
            ),
            (
                "c",
                "The Controls POD owns valve control.",
                Some("2026-12-01"),
            ),
            (
                "unrelated",
                "The cafeteria serves lunch at noon.",
                Some("2026-03-01"),
            ),
        ]
        .into_iter()
        .map(|(id, content, ts)| (id.to_string(), sdoc(id, content, ts)))
        .collect()
    }

    #[test]
    fn content_change_in_middle_of_chain_matches_oracle() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("b")
        );

        docs.insert(
            "b".to_string(),
            sdoc(
                "b",
                "The cafeteria serves lunch at noon.",
                Some("2026-06-01"),
            ),
        );
        check(
            &mut store,
            &docs,
            &["b"],
            &[],
            &["a", "b", "c", "unrelated"],
        );
        // b no longer participates; a is now directly superseded by c.
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("c")
        );
    }

    #[test]
    fn removing_middle_of_chain_relinks_neighbors() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);

        docs.remove("b");
        check(
            &mut store,
            &docs,
            &[],
            &["b"],
            &["a", "b", "c", "unrelated"],
        );
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("c")
        );
        assert_eq!(store.document_state("b"), DocumentSemanticState::default());
    }

    #[test]
    fn new_doc_creating_conflict_matches_oracle() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);

        // Same subject/predicate, different value, no timestamp and no
        // correction cue: a visible conflict, not a supersession.
        docs.insert(
            "d".to_string(),
            sdoc("d", "The Rogue POD owns valve control.", None),
        );
        check(
            &mut store,
            &docs,
            &["d"],
            &[],
            &["a", "b", "c", "d", "unrelated"],
        );
        assert_eq!(
            store.document_state("d").status,
            Some(SemanticStatus::Conflicted)
        );
    }

    #[test]
    fn new_doc_confirming_existing_claim_matches_oracle() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);

        docs.insert(
            "d".to_string(),
            sdoc(
                "d",
                "Valve control is owned by the Controls POD.",
                Some("2027-01-01"),
            ),
        );
        check(
            &mut store,
            &docs,
            &["d"],
            &[],
            &["a", "b", "c", "d", "unrelated"],
        );
        assert!(store
            .relations()
            .iter()
            .any(|r| { r.kind == SemanticRelationKind::Confirms && r.source_doc_id == "d" }));
    }

    #[test]
    fn explicit_link_add_retarget_remove_match_oracle() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);
        let ids = &["a", "b", "c", "unrelated", "x"];

        let mut x = sdoc("x", "Completely new direction.", Some("2027-02-01"));
        x.filters.insert("supersedes_id".into(), "a".into());
        docs.insert("x".to_string(), x);
        check(&mut store, &docs, &["x"], &[], ids);
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("x")
        );

        // Retarget the link to c.
        let mut x = docs.get("x").unwrap().clone();
        x.filters.insert("supersedes_id".into(), "c".into());
        x.content = "Completely new direction v2.".to_string();
        docs.insert("x".to_string(), x);
        check(&mut store, &docs, &["x"], &[], ids);
        assert_eq!(
            store.document_state("c").superseded_by.as_deref(),
            Some("x")
        );

        // Drop the link entirely (filters change, claims identical).
        let mut x = docs.get("x").unwrap().clone();
        x.filters.remove("supersedes_id");
        docs.insert("x".to_string(), x);
        check(&mut store, &docs, &["x"], &[], ids);
        assert_ne!(
            store.document_state("c").status,
            Some(SemanticStatus::Superseded)
        );
    }

    #[test]
    fn explicit_link_target_removed_drops_relation() {
        let mut docs = chain_corpus();
        let mut x = sdoc("x", "New direction.", Some("2027-02-01"));
        x.filters.insert("supersedes_id".into(), "a".into());
        docs.insert("x".to_string(), x);
        let mut store = full_build(&docs);
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("x")
        );

        docs.remove("a");
        check(
            &mut store,
            &docs,
            &[],
            &["a"],
            &["a", "b", "c", "unrelated", "x"],
        );
        assert!(store.relations().iter().all(|r| r.method != "explicit"));
    }

    #[test]
    fn scope_change_breaking_relation_matches_oracle() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);

        // Move b into a different semantic scope: its chain relations vanish.
        let mut b = docs.get("b").unwrap().clone();
        b.filters
            .insert("semantic_scope".into(), "other-project".into());
        docs.insert("b".to_string(), b);
        check(
            &mut store,
            &docs,
            &["b"],
            &[],
            &["a", "b", "c", "unrelated"],
        );
        assert!(store
            .relations()
            .iter()
            .all(|r| r.source_doc_id != "b" && r.target_doc_id != "b"));
    }

    #[test]
    fn source_kind_change_flips_supersedes_to_conflict() {
        let mut docs = HashMap::new();
        docs.insert(
            "a".to_string(),
            sdoc(
                "a",
                "The National Park POD owns valve control.",
                Some("2026-01-01"),
            ),
        );
        docs.insert(
            "b".to_string(),
            sdoc(
                "b",
                "The Platform POD owns valve control.",
                Some("2026-06-01"),
            ),
        );
        let mut store = full_build(&docs);
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("b")
        );

        // Same content and timestamp, but b is now a memory-kind source while a
        // stays document-kind: the chronological change becomes a conflict.
        docs.insert(
            "b".to_string(),
            sdoc_with_source(
                "b",
                "The Platform POD owns valve control.",
                Some("2026-06-01"),
                "codex://project/session/b",
            ),
        );
        check(&mut store, &docs, &["b"], &[], &["a", "b"]);
        assert_eq!(
            store.document_state("a").status,
            Some(SemanticStatus::Conflicted)
        );
        assert_eq!(
            store.document_state("b").status,
            Some(SemanticStatus::Conflicted)
        );
    }

    #[test]
    fn timestamp_only_edit_reorders_chain_like_oracle() {
        let mut docs = HashMap::new();
        docs.insert(
            "a".to_string(),
            sdoc(
                "a",
                "The National Park POD owns valve control.",
                Some("2026-01-01"),
            ),
        );
        docs.insert(
            "b".to_string(),
            sdoc(
                "b",
                "The Platform POD owns valve control.",
                Some("2026-06-01"),
            ),
        );
        let mut store = full_build(&docs);
        assert_eq!(
            store.document_state("a").superseded_by.as_deref(),
            Some("b")
        );

        // Timestamp-only edit: b becomes older than a, so the chain flips and a
        // now supersedes b. Claim IDs are unchanged, only the timestamp moved.
        docs.insert(
            "b".to_string(),
            sdoc(
                "b",
                "The Platform POD owns valve control.",
                Some("2025-06-01"),
            ),
        );
        check(&mut store, &docs, &["b"], &[], &["a", "b"]);
        assert_eq!(
            store.document_state("b").superseded_by.as_deref(),
            Some("a")
        );
    }

    #[test]
    fn noop_content_update_leaves_store_identical() {
        let docs = chain_corpus();
        let mut store = full_build(&docs);
        let before_relations = store.relations().to_vec();
        let before_claims = store.claims().to_vec();

        // Re-upsert byte-identical docs: nothing may change.
        apply(&mut store, &docs, &["a", "unrelated"], &[]);
        assert_eq!(store.claims(), before_claims.as_slice());
        assert_eq!(store.relations(), before_relations.as_slice());
        assert_store_eq(&full_build(&docs), &store, &["a", "b", "c", "unrelated"]);
    }

    #[test]
    fn empty_update_is_a_noop() {
        let docs = chain_corpus();
        let mut store = full_build(&docs);
        apply(&mut store, &docs, &[], &[]);
        assert_store_eq(&full_build(&docs), &store, &["a", "b", "c", "unrelated"]);
    }

    #[test]
    fn multi_step_sequence_matches_oracle_at_every_step() {
        let mut docs = chain_corpus();
        let mut store = full_build(&docs);
        let ids = &["a", "b", "c", "d", "e", "unrelated"];

        // 1. Add a conflicting doc.
        docs.insert(
            "d".to_string(),
            sdoc("d", "The Rogue POD owns valve control.", None),
        );
        check(&mut store, &docs, &["d"], &[], ids);

        // 2. Newer doc supersedes the chain head with a correction cue.
        docs.insert(
            "e".to_string(),
            sdoc(
                "e",
                "This replaces the previous decision. The Apex POD owns valve control.",
                Some("2027-06-01"),
            ),
        );
        check(&mut store, &docs, &["e"], &[], ids);
        assert_eq!(
            store.document_state("c").superseded_by.as_deref(),
            Some("e")
        );

        // 3. Tombstone the conflict doc.
        docs.remove("d");
        check(&mut store, &docs, &[], &["d"], ids);

        // 4. Rewrite b to an unrelated topic.
        docs.insert(
            "b".to_string(),
            sdoc(
                "b",
                "The cafeteria serves lunch at noon.",
                Some("2026-06-01"),
            ),
        );
        check(&mut store, &docs, &["b"], &[], ids);

        // 5. Re-add b's original claim under a new doc id.
        docs.insert(
            "b2".to_string(),
            sdoc(
                "b2",
                "The Platform POD owns valve control.",
                Some("2026-06-01"),
            ),
        );
        check(
            &mut store,
            &docs,
            &["b2"],
            &[],
            &["a", "b", "b2", "c", "d", "e", "unrelated"],
        );
    }

    #[test]
    fn disabled_options_clears_store() {
        let docs = chain_corpus();
        let mut store = full_build(&docs);
        assert!(!store.is_empty());
        let disabled = SupersessionOptions {
            enabled: false,
            ..SupersessionOptions::default()
        };
        store
            .update_documents(&docs, &[], &[], disabled)
            .expect("disabled update must succeed");
        assert!(store.is_empty());
    }

    #[test]
    fn update_rejects_invalid_options() {
        let docs = chain_corpus();
        let mut store = full_build(&docs);
        let bad = SupersessionOptions {
            suppress_confidence: 1.5,
            ..SupersessionOptions::default()
        };
        let err = store
            .update_documents(&docs, &[], &[], bad)
            .expect_err("invalid options must be rejected");
        assert!(err.to_string().contains("confidence thresholds"));
    }

    /// Deterministic PRNG (xorshift64*) so the randomized differential test
    /// is reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Randomized differential test: random sequences of inserts, edits, and
    /// deletes must leave the incrementally-updated store identical to a
    /// fresh full rebuild after every single step.
    #[test]
    fn randomized_mutations_match_oracle() {
        let subjects = ["valve control", "budget approval", "deploy freeze"];
        let objects = [
            "National Park POD",
            "Platform POD",
            "Controls POD",
            "Apex POD",
        ];
        let dates = [
            Some("2026-01-01"),
            Some("2026-06-15"),
            Some("2027-03-20"),
            None,
        ];
        let no_claim = "The cafeteria serves lunch at noon.";
        let ids = ["d0", "d1", "d2", "d3", "d4", "d5"];

        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut docs: HashMap<String, SourceDocument> = HashMap::new();
        let mut store = full_build(&docs);
        assert!(store.is_empty());

        for step in 0..120 {
            let id = ids[rng.below(ids.len())];
            let op = rng.below(100);
            let (changed, removed) = if op < 55 || !docs.contains_key(id) {
                // Upsert with random content, date, and filters.
                let mut doc = match rng.below(100) {
                    0..=9 => sdoc(id, no_claim, dates[rng.below(dates.len())]),
                    10..=19 => sdoc(
                        id,
                        &format!(
                            "This replaces the previous decision. The {} owns {}.",
                            objects[rng.below(objects.len())],
                            subjects[rng.below(subjects.len())]
                        ),
                        dates[rng.below(dates.len())],
                    ),
                    20..=29 => sdoc(
                        id,
                        &format!(
                            "{} is owned by the {}.",
                            subjects[rng.below(subjects.len())],
                            objects[rng.below(objects.len())]
                        ),
                        dates[rng.below(dates.len())],
                    ),
                    _ => sdoc(
                        id,
                        &format!(
                            "The {} owns {}.",
                            objects[rng.below(objects.len())],
                            subjects[rng.below(subjects.len())]
                        ),
                        dates[rng.below(dates.len())],
                    ),
                };
                // Occasionally attach scope or explicit-link filters.
                match rng.below(100) {
                    0..=7 => {
                        doc.filters
                            .insert("semantic_scope".into(), "project-x".into());
                    }
                    8..=14 => {
                        let target = ids[rng.below(ids.len())];
                        if target != id {
                            doc.filters.insert("supersedes_id".into(), target.into());
                        }
                    }
                    _ => {}
                }
                docs.insert(id.to_string(), doc);
                (vec![id], vec![])
            } else if op < 80 {
                docs.remove(id);
                (vec![], vec![id])
            } else {
                // No-op re-upsert of an existing doc.
                (vec![id], vec![])
            };
            let changed_docs: Vec<&SourceDocument> = changed
                .iter()
                .map(|cid| docs.get(*cid).expect("changed doc must exist"))
                .collect();
            let removed_ids: Vec<String> = removed.iter().map(|s| s.to_string()).collect();
            store
                .update_documents(
                    &docs,
                    &changed_docs,
                    &removed_ids,
                    SupersessionOptions::default(),
                )
                .expect("incremental update must succeed");
            let oracle = full_build(&docs);
            let all_ids: Vec<&str> = docs.keys().map(String::as_str).collect();
            assert_store_eq(&oracle, &store, &all_ids);
        }
    }
}
