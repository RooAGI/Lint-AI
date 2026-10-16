# Document enrichment implementation plan

Status: implementation in progress on `feat/document-enrichment`. The shared
MCP router exposes `review_search`, `search_feedback`, and `enrich_doc`, assigns
search IDs, pages unreviewed history, and routes evidence-checked document
enrichment through `MemoryService`. Query-only project rule overlays reload automatically, and
explicit old/new comparisons search both sides. `enrich_doc` can maintain a session conclusion on the existing
`session-context` document and submit an evidence-cited, same-scope,
whole-document supersession through existing semantic relations. Automatic conclusion generation, claim-level corrections,
broader semantic reconciliation, and extraction-impact rule reload remain
future work. Rationale questions now have a typed `Explain`
intent and use built-in or project-added query expansion terms; this improves
query interpretation but does not yet guarantee rationale retrieval or resolve
every implicit reference. This is branch documentation, not a release claim.

## Objective

Improve future searches by reviewing recorded MCP searches and enriching
existing project and session documents with grounded feedback. Expose three
provider-neutral MCP tools: `review_search`, `search_feedback`, and `enrich_doc`.
`MemoryService` owns history selection, review state, validation, persistence,
and publication.

Each search review records an overall usefulness label, an explanation, and a
relevance judgment for every returned document. The service binds each
judgment to the immutable search result's document ID, original rank, and
source hash. These judgments support evaluation; they do not by themselves
change retrieval or establish what is true. Document enrichment and
query-rule changes use explicit service operations.

## Existing foundation and implementation gaps

| Enrichment | Existing types and fields | Current search use | Required work |
| --- | --- | --- | --- |
| Project terminology | `SourceDocument.key_phrases`, `KeyPhrase.text` and `kind`; derived `Tier1Entity` | Phrase/entity signals contribute to matching and routing | Admit grounded provider additions and retain their origin separately from extractor contributions. Explicit project aliases are a later extension. |
| Subject, entities, and topic | `SourceDocument.concept`, `headings`, and `content`; derived `DocRecord.probable_topic`, `key_entities`, `important_terms`, and `top_claims` | Index scoring and segment profiles consume these derived fields | Enrich source fields and rebuild affected derived records through the existing pipeline. |
| Session conclusions | Existing `session-context` source document, its `content`, `group_id`, `links`, and session filters | The document participates in ordinary retrieval | `enrich_doc` can update its managed conclusion, headings, and links when the recorded search has explicit or resident-inferred session attribution. Hook refreshes preserve these fields and retain original observations as separate documents. Automatic conclusion generation and reconciliation across all session observations remain future work. |
| Corrections | `SourceDocumentEnrichment.supersedes_document_id`, existing `filters["supersedes_id"]`, `SemanticRelation`, `SemanticStatus`, and `WriteAdjudicationReceipt` | Existing semantic policy handles explicit document supersession and reports searchable state | `enrich_doc` accepts an explicit whole-document correction when the older document is hash-cited as evidence and both documents share semantic scope and a backing index. It flushes the owning store before returning the actual adjudication. Claim-level correction that preserves unrelated facts inside one source document remains unsupported. |
| Decision and implementation connections | `SourceDocument.links`, persisted as `DocRecord.doc_links` | Resolved document-ID links can boost existing retrieval candidates | Validate targets and add useful links. Candidate expansion through links is separate work. |

Code entry points:

- [Source documents and key phrases](https://github.com/RooAGI/Membrick/blob/main/src/source.rs).
- [Derived document records, claims, and evidence](https://github.com/RooAGI/Membrick/blob/main/src/index/model.rs).
- [Source-to-record construction](https://github.com/RooAGI/Membrick/blob/main/src/pipeline/build.rs).
- [MemoryService and existing enrichment handling](https://github.com/RooAGI/Membrick/blob/main/src/memory_api.rs).
- [Semantic relationships](https://github.com/RooAGI/Membrick/blob/main/src/semantic_relations.rs).
- [Shared MCP tools](https://github.com/RooAGI/Membrick/blob/main/src/integrations/mcp_tools.rs).
- [MCP call recorder](https://github.com/RooAGI/Membrick/blob/main/src/telemetry.rs).
- [Query analysis](https://github.com/RooAGI/Membrick/blob/main/src/query_semantics.rs) and
  [shared query preparation](https://github.com/RooAGI/Membrick/blob/main/src/query_plan.rs).

`Tier1Entity` has no canonical-entity or alias field. General query expansion
uses bundled lexical resources; it does not learn project aliases from sessions.
Claude and Codex captures are incremental. The resident now retains previously
stored observation links and provider enrichments when it upserts the next
session-context document, so new captures accumulate references. Raw capture
links remain even when a feedback contribution only adds a conclusion or
headings. Feedback-owned links remain tied to their evidence and are removed
when that evidence changes or disappears. Existing sessions captured before
this behavior may not have a complete context link list, though their
observation documents remain stored.

## Document update contract

1. Keep document IDs, ownership, source identity, and original capture text.
   Do not rewrite repository files or replace raw observations with summaries.
2. Add terminology and valid links to existing source documents. Preserve
   human-authored headings and extractor-owned metadata. Generated topic
   headings belong on generated session-context documents.
3. Store a session conclusion in the existing session-context document. Its
   generated content can change; complete original observations remain stored.
   Resolve session membership through existing project, user, provider, and
   session metadata, including observations outside the latest capture batch.
4. Associate each managed contribution with its operation ID, supporting
   document IDs and content hashes, and the fields it contributed. Reuse source
   metadata for small bookkeeping values. Keep operation progress in a local
   service-owned sidecar outside the searchable corpus. Avoid introducing a
   parallel document, entity, or claim model.
5. On source changes, invalidate or recompute contributions whose dependencies
   changed. Preserve still-valid contributions across watcher updates and later
   hook captures. A hook must not overwrite an enriched context with its next
   generated header.
6. Merge provider and extractor contributions deterministically. Existing
   `key_phrase_extraction_hash` describes extractor completion for source text;
   provider enrichment must not falsely mark extraction complete or disappear
   when the extractor refreshes. Extend provenance only where existing fields
   cannot represent these ownership and invalidation requirements.
7. Route updates through MemoryService and its existing durable writer and
   staged publication. Do not write `DocRecord`, index files, semantic stores,
   or provider memory directories directly from an adapter.

An explicit `supersedes_document_id` means the reviewed source document as a
whole is being replaced. It does not identify or supersede one fact within a
multi-fact document. Keep the old source document and its observations intact;
only use this relationship when whole-document replacement is intended. A
claim-level correction that preserves other facts in the same document needs a
typed consumer at claim or chunk granularity and remains out of scope here.

One source URI can identify multiple distinct observations. Use document IDs
and versions for update identity. An old `lint-ai://` prefix alone is not a
reason to remove, suppress, or supersede a document.

## Three MCP tools: review_search, search_feedback, and enrich_doc

Keep search-history review and feedback submission as separate tools with
distinct contracts:

1. **`review_search` without `search_id`:** return a page of unreviewed search
   records from `mcp-call-history.jsonl`. Accept a page size and opaque
   continuation cursor. A page includes each search ID, query, provider, time,
   result payload, and enough referenced source-document context for review.
   Returning a page does not mark its searches reviewed.
2. **`review_search` with `search_id`:** return that exact recorded search,
   whether or not it is still in the unreviewed queue. Include its full query and
   result, plus permitted source documents needed to understand the outcome.
3. **`search_feedback`:** require one `search_id`, an overall label
   (`helpful`, `partially_helpful`, or `not_helpful`), an explanation, and one
   relevance judgment (`relevant`, `partially_relevant`, or `not_relevant`)
   for every returned document. The service resolves each judgment's original
   rank and source hash from the recorded result, rejecting missing, duplicate,
   or foreign document IDs. Judgments are evaluation data. The tool may also
   propose grounded additions to supported query-only rule lists; it does not
   accept document enrichment fields.
4. **`enrich_doc`:** require one exact `doc_id`, the expected current source
   hash, and hash-cited evidence documents. It applies supported metadata,
   session-context updates, or explicit same-scope document supersession through
   `MemoryService`; it is independent of search labels.
5. `MemoryService` verifies search identity for feedback, validates query-rule
   grounding, persists review state, and routes document changes through the
   existing staged writer and publication path. Invalid input returns a tool
   error with the validation reason; enrichment retries are idempotent.

Every ordinary `search` response exposes a stable `search_id` that
`search_feedback` can reference. Its tool guidance should tell the agent to
call `search_feedback` when the results fail to answer the question or miss
important context, and to omit feedback when the results are adequate. This
gives immediate conditional guidance; the tool does not invoke feedback or
decide whether a result is adequate. The agent can send feedback directly with
the returned ID without calling `review_search` first. `review_search` remains
available for exact lookup and paginated queued or deferred review.

The overall label describes the search's usefulness; per-document judgments
describe how each returned hit relates to the query. Both are retained as
evaluation data with the original ranks and source hashes; neither directly
trains ranking. Future retrieval changes come from validated document
enrichment or implemented query-rule changes. Use small typed request and
result contracts around existing `SourceDocument` fields.
The server selects the workspace and owner; tool arguments cannot select a
different root or bypass visibility restrictions.

Track review status by search ID in service-owned local state, not by rewriting
the append-only MCP history. Feedback retries must be idempotent; a repeated
search ID with different feedback must be rejected or explicitly versioned.
Concurrent feedback cannot overwrite newer source content. A source change
during agent review requires refreshed evidence. A pending retry may tolerate a
changed evidence hash only when that evidence is also a target carrying the
same search's already-applied enrichment. Validate the complete feedback before
mutation and use the existing writer's publication boundaries.

## Automatic execution

Use the existing project `.membrick/settings.toml` settings infrastructure and
no environment variables. After each search, include concise conditional
guidance to use `search_feedback` with that search ID if the result was
inadequate. Do not require feedback for successful searches. This guidance
prompts the agent; it does not run a second model call or submit feedback by
itself. The `review_search` queue supports paginated follow-up review and
survives provider restarts. Provider-specific MCP adapters must expose the
same tools and guidance through the shared MemoryService route.

Automatic execution while a provider is active is the first delivery target.
Clock-based execution while every provider is closed requires a configured
provider runner. MCP by itself cannot start a model turn. Document provider
capabilities accurately instead of promising an unattended schedule everywhere.

Read the growing call history incrementally with a durable cursor. Detect file
replacement, handle incomplete final lines, and retain a backlog rather than
dropping work. Batches bound processing per invocation; they do not truncate
source documents or discard recorded calls. Page large inputs through
continuation cursors when needed. Model calls and broad source collection run
outside ordinary search latency and outside the writer's critical section.

The MCP history schema includes a per-call `trace_id` and records whether a
`session_id` was explicit or inferred from the resident's active provider
session. The resident passes that attribution to the recorder as internal
metadata and removes it from the provider-facing search response. Older
entries without attribution remain readable and are treated as absent.

Record `review_search`, `search_feedback`, and `enrich_doc` invocations in the existing MCP
call history, but exclude maintenance calls from the set of searches awaiting
review. Keep the original search record immutable. Mark feedback transcript
content so it cannot be treated as independent confirmation of its proposed
enrichment. Quoted search results must retain their origin.

## How enrichment improves future queries

The production path remains:

```text
Provider proposes enrichment
          -> MemoryService validates and updates existing SourceDocuments
          -> Existing extraction, semantic relations, and staged publication
          -> Updated terms, entities, topics, claims, and segment profiles
          -> Existing query preparation and retrieval
```

For example, a session establishes how the v0.6.0 CLI was simplified. Enrich
its existing context with the established conclusion, relevant terminology,
version context, and links to the actual CLI documentation and observations.
A later query about command cleanup then has a focused existing document and
stronger matching signals available. Merely remembering that an earlier search
was poor would not produce those signals.

Reuse `QueryAnalysis` and `QueryIntent` for question operations. Preserve entity,
temporal, project, and version distinctions. `filters` remain metadata with
defined consumers; arbitrary new filter strings do not become search features
automatically. Likewise, validate that `concept`, headings, and key phrases
reach the intended scoring and routing channels before claiming an improvement.

Phase one expresses terminology through document enrichment. Explicit
project-scoped alias expansion needs additional storage and a query-time
consumer; key phrases alone do not establish equivalence. Implement that
extension only after measuring failures the first phase cannot resolve.
General NER patterns and bundled reasoning rules require separately tested
changes rather than automatic edits based on one provider's conclusion.

For corrections, reuse existing scoped confirmation/conflict/supersession
behavior. Preserve unaffected facts in multi-fact documents. A newer timestamp
or repeated assistant assertion alone does not establish a correction.
For document links, test the existing candidate boost first; following links
to introduce additional candidates is a distinct retrieval change.

## What gets added and what consumes it

Document enrichment runs when the provider has established useful project
knowledge. Query interpretation runs for each incoming question. Implement both
paths explicitly; storing a phrase in a document or rule file does not by itself
implement a retrieval operation.

| Example | Addition during enrichment | Addition to query handling | Retrieval consumer |
| --- | --- | --- | --- |
| "CLI consolidation" | Add grounded phrases such as "CLI consolidation" and "command interface" to the relevant document's key phrases and generated context heading. | Preserve those phrases as query focus/entities. Project-specific alias expansion remains a later extension. | Existing term/entity matching and segment profiles consume the rebuilt source metadata. |
| "Latest interface" | Preserve established version context and supported corrections between the interface documents. | Reloadable `query_current_state_markers` map recognized cues to the single-scope `CurrentState` intent. Interpret the applicable entity and scope; "latest interface" is not automatically an event-ordering question. | Shared semantic visibility and ranking select applicable current information. Modification time alone cannot establish applicability. |
| "Why did we change it?" | Put the established rationale in the existing session-context document and link the original discussion. | `QueryKind::Why` maps to `QueryIntent::Explain`; built-in or project-added `query_explanation_terms` enter the augmented search query. Resolving "it" through session context remains separate work. | Shared retrieval searches with rationale terms and existing semantic policy. A MemoryService regression test verifies rationale-document retrieval; broader retrieval evaluation remains open. |
| "Compare old and new commands" | Connect the previous interface, replacement, and change explanation with applicable version context. | Add general comparison language cues, a typed comparison operation, and operand extraction to existing query analysis. Explicit old/new operands request both states; comparison between two current products need not enable history. | Retrieve evidence for both operands, retain their version/state identity, and report a missing side rather than substitute an unrelated document. |

The generalized language path is:

```text
Bundled phrase/pattern rule
    -> QueryAnalysis typed interpretation
    -> PreparedQuery carries requirements and resolved context
    -> MemoryService's shared retrieval consumes those requirements
    -> Results satisfy current-state, rationale, or comparison needs
```

Extend the existing `QueryAnalysis`, `QueryIntent`, and evidence requirements
where needed. `Explain` and `CurrentState` are typed single-scope intents; their
query-only terms or markers are reloadable from the project overlay. A
MemoryService regression test confirms that a rationale query retrieves a
rationale document. Current-state markers express the request; existing
semantic visibility determines which indexed facts are applicable. The
`Compare` intent now extracts explicit old/new operands and the shared
`MemoryService` searches each side independently. Hits retain their
operand labels, and MCP reports when a side has no results. This does not yet
cover every implied comparison or make one-sided evidence complete; extend the
typed analysis and regression cases as those gaps are identified. Historical
queries must still respect ownership, deletion, expiry, and explicit access
restrictions.

Implement language cues in `data/reasoning_rules.json` together with the typed
loader, analysis, and retrieval consumers. Bundled rules remain compiled into
the binary, while workspace services now merge a validated `.membrick/rules.json`
overlay and reload supported query-only changes automatically. `search_feedback`
can add grounded terms to supported query-only rules; the existing watcher
reloads and activates valid changes for later queries.
The current reload foundation does not yet activate extraction-impacting rules
with a matching rebuilt index. An overlay cannot introduce a Rust operation or
retrieval consumer that the installed binary does not implement.

## Automatic rule reload

Add a project rule overlay at `.membrick/rules.json`, with an optional path
configured in `.membrick/settings.toml`. Resolve relative paths from the project
root through one shared resolver. Bundled `data/reasoning_rules.json` remains the
baseline. An absent overlay uses the baseline. No environment variable, manual
reload command, or resident restart is required for a supported rule edit.

The overlay extends the existing typed rule schema. Specify deterministic merge
semantics: omitted fields inherit defaults, phrase-list additions are normalized
and deduplicated, and replacement/removal must be explicit. Validate the complete
merged rules, including schema version, field types, allowed typed outputs, and
any regex. Reject unsupported fields and operations rather than silently ignore
them. A provider can propose or submit only rules the installed implementation
understands. Validation and regression checks remain required before automatic
provider proposals are written to the active overlay.

### Workspace ownership and activation

1. **Own rules in MemoryService.** Retain a shared immutable built-in default,
   and give each workspace its own immutable effective rules snapshot. The
   current process-wide `OnceLock<ReasoningRules>` must not hold mutable project
   rules; loading project A cannot change project B.
2. **Detect edits automatically.** Reuse filesystem notification infrastructure,
   watching the parent directory to detect atomic replacement as well as
   ordinary writes, removal, and recreation. Debounce notifications and perform
   a periodic recovery check for missed events. Use a content hash rather than
   modification time alone. Do not parse or scan rule files on every search.
   A settings change to the configured rule path must rebind the watcher and
   re-evaluate the effective rules.
3. **Compile a candidate outside the writer lock.** Read a complete candidate,
   merge it with defaults, validate it, and compile its matchers once. Malformed
   or partly written JSON leaves the last accepted snapshot active. Record the
   failure for status reporting and retry on subsequent changes. On first start
   without a valid prior snapshot, use bundled defaults and report the error.
4. **Identify the impact.** Maintain separate query-rule and extraction-rule
   fingerprints. Classify changed fields by their actual consumers, not by a
   provider's declaration. A field used on both paths has extraction impact.
   Unclassified changes conservatively require re-extraction. Formatting-only
   edits and identical effective rules cause no activation or index work.
5. **Activate query-only changes atomically.** A request pins one immutable rule
   snapshot from session resolution through analysis, preparation, and retrieval.
   New requests use the accepted generation; in-flight requests finish with
   their pinned generation. Include the query-rule fingerprint in affected
   cache identities and diagnostics. Restarting a process or rebuilding stored
   documents is unnecessary for a query-only edit.
6. **Publish extraction changes with their index.** Automatically schedule
   re-extraction using the new extraction fingerprint and the existing background
   publication path. Keep serving the prior compatible rules/index pair while
   it builds. Pin the candidate rules for the entire build and publish the new
   compatible pair together. Account for writes arriving during the build through
   the writer's revision barriers; no newer source update may be lost. A later
   rule edit must prevent an older build from activating over it.
7. **Persist and recover generations.** Persist the last accepted effective rules,
   fingerprints, and pending extraction progress through MemoryService. Restart
   must reconstruct a compatible rules/index pair and resume pending work.
   Deleting the overlay requests a return to bundled defaults using the same
   query-only or extraction-impact activation rules. Permission/read failures
   retain the last valid snapshot and report an error rather than masquerade as
   an intentional deletion.

Propagate the pinned rules explicitly through existing query and extraction
functions. Replace static-rule assumptions in `PreparedQuery`, query analysis,
claim extraction, and any rule-derived regex caches. Avoid process-global project
settings or thread-local scope. The public standalone convenience path can keep
using bundled defaults; all workspace calls use the snapshot selected by their
MemoryService. IPC, HTTP, filesystem fallback, hooks, and bindings must converge
on that selection.

Expose the configured path, active and pending fingerprints, last reload result,
and re-extraction progress through existing status/info reporting. Reload failure
must be observable without breaking ordinary searches. Query-only changes can
become active immediately after validation; ingestion changes are reported as
pending until the compatible index is published.

## Incremental reindexing and query-rule activation

Enriching indexed source fields is an incremental reindexing process. It updates
the existing document identity and the derived search view. It does not require
rescanning the project or reingesting every original transcript. Changes confined
to how questions are interpreted can take effect against the existing index.

All provider and MCP mutations enter through `MemoryService`. For a single
source-document update, `MemoryService::upsert` applies common enrichment
scheduling before delegating to its internal `IndexStore`; `enrich_doc` uses
`MemoryService::enrich_source_documents`, invoked by the dedicated `enrich_doc` tool, which validates the target and cited evidence before staging it through the existing writer path. Integrations do not call
`IndexStore::upsert` or stage index files directly. Those internal store
primitives are implementation details behind the service boundary.

| Change | Required index work | Activation work |
| --- | --- | --- |
| Key phrases, headings, concept, or generated context content | Rebuild changed document records and affected index structures | Publish a consistent search generation. |
| Document links or supported correction metadata | Update affected records, graph signals, and semantic dependencies | Publish the matching records and semantic state together. |
| Query-only language cues, comparison execution, or ranking policy | No document re-extraction solely for this change | Activate the new query implementation/rules and invalidate incompatible query caches. |
| NER or claim-extraction rules used to build stored records | Re-extract the affected document population; a general rule change may require the full corpus | Version the extractor and rebuild dependent indexes before claiming complete coverage. |
| A rule shared between query analysis and ingestion | Determine both query and stored-record effects | Apply both activation paths when stored interpretations can change. |

Reuse the existing source hash and staged write mechanisms. The record hash in
`src/pipeline/build.rs` already covers content, concept, headings, links, filters,
key phrases, and extraction/build identities. The content-only key-phrase
extraction stamp has a different purpose and is not the record's index version.

Implement an accepted enrichment batch as follows:

1. **Prepare outside the writer lock.** The provider reads a versioned source
   set and proposes field changes. Keep model reasoning and broad searches out
   of the publication critical section.
2. **Validate and merge under the resident writer.** Check expected document and
   dependency hashes, ownership, supported fields, and operation idempotency.
   Merge with the latest source using deterministic ordering and deduplication.
   A different current source requires a fresh proposal rather than overwriting
   concurrent captures.
3. **Remove no-op updates.** Compare effective searchable source fields and
   managed contributions. Recording another run time or receipt must not dirty
   an unchanged document. Keep scheduling and lease state in the sidecar rather
   than hashed source filters. Preserve extractor ownership and original text.
4. **Stage the batch once.** Reuse `MemoryService::stage_upsert_batch` internally
   through a service method that implements this validation contract. The
   existing batch primitive journals and stages sources; it does not itself
   supply expected-version validation or operation receipts. Do not refresh
   after every added phrase or linked document.
5. **Build the affected search view.** Reuse the existing record construction
   and changed-document publication path. Recompute changed records, affected
   segmented indexes, routing summaries, and semantic dependencies. If document
   membership changes, update both old and new segments. Shared statistics or
   a single-index layout may require broader work; measure actual work rather
   than promise that all publication cost scales only with changed documents.
6. **Publish consistently.** Reuse staged publication and its revision barriers.
   Readers use a consistent published generation. Report durable acceptance
   separately from published visibility, and do not report completion until the
   caller's required revision is visible. Invalidate or version result caches
   and derived lookup caches affected by the changed generation.
7. **Persist completion and recover.** Mark work complete only after the
   required durable/publication milestones. Recover a crash between journaling,
   publication, and receipt persistence without duplicate enrichment. Failed
   builds leave retryable work; expired provider leases do not undo accepted
   writes.

The existing segmented refresh can reuse unchanged segments. That does not imply
that every supported layout or shared statistic is already fully incremental.
Use the [write publication plan](write-publication-plan.md) for publication costs
and remaining optimization work. Enrichment should reuse that path and measure
changed documents, records rebuilt, segments rebuilt/reused, publication time,
and query latency while publication runs.

Version query interpretation separately from document extraction when possible.
A query-only rule change should not invalidate all stored record hashes merely
because a shared build version was bumped. Conversely, changes to stored entity
or claim extraction must invalidate old derived records; restarting the query
process alone cannot update those facts.

## Implementation sequence

- [x] Complete stable trace identity and reliable session attribution in search
  history. Per-call IDs and explicit/absent attribution are implemented;
  resident searches also record the resolved active session without exposing
  internal telemetry metadata in the provider-facing result. Review includes
  session documents only for explicit or resident-inferred attribution.
- [x] Define allowed source-field contributions, ownership, dependency hashes,
  invalidation, and merge behavior with existing extraction and hook updates.
  Implemented provider-owned key phrases, headings, links, conclusions, and
  explicit document supersession as per-search contributions stored on the
  existing source document. Contributions cite supporting document content
  fingerprints; hook or watcher refreshes preserve them while the cited
  evidence is unchanged and remove the contribution when evidence changes or
  disappears. Extractor-owned phrases remain separate. Regression coverage
  verifies preservation and invalidation across source refreshes. This does
  not yet cover claim-level correction or automatic extraction of conclusions.
- [x] Add MemoryService methods for selecting work, claiming operations, and
  applying idempotent enrichment through the resident writer. `review_search`
  selects exact or paginated work; `search_feedback` records overall and per-document judgments; `enrich_doc` applies
  validated document changes and records the publication receipt.
  Identical retries resume a pending review and return the stored receipt after
  completion. Provider MCP adapters route these operations through the shared
  MemoryService path.
- [ ] Generate conclusions automatically from all linked session observations;
  `enrich_doc` currently accepts an agent-authored, evidence-backed managed
  conclusion on an existing session-context document.
- [ ] Add explicit terminology aliases and supported semantic corrections.
- [ ] Connect enriched source changes to batched incremental reindexing, no-op
  detection, dependency invalidation, cache invalidation, and publication receipts.
  Partial coverage: existing batch validation and staged publication apply
  enrichments, and focused regressions cover idempotency, stale writes, evidence
  invalidation, and target-segment reuse. Full receipt and failure-recovery
  coverage remains open.
- [x] Implement the explicit old/new comparison path through shared
  `MemoryService`, preserving per-hit operands in MCP and MemoryService search
  responses and reporting missing MCP sides.
- [ ] Complete typed query interpretation and retrieval for current-state
  requests, implicit-reference resolution for change rationale, and broader
  rationale evaluation; broaden comparison coverage as needed.
- [x] Implement project query-rule overlays with workspace-owned snapshots,
  validation, watcher recovery, fingerprints, and automatic query activation.
  Effective-rule fingerprints serialize through a canonical JSON value so
  unordered map iteration cannot create false generations on equivalent edits.
- [ ] Add extraction-impact rule reload, re-extraction, and atomic publication
  of a compatible rule/index pair.
- [x] Expose shared `review_search` and `search_feedback` tools through the
  shared provider MCP router. Add conditional per-search guidance and stable
  search IDs.
- [x] Implement paginated unreviewed-search lookup, exact search lookup,
  overall and per-document relevance judgments through `search_feedback`, and evidence-checked document
  updates through the dedicated `enrich_doc` operation in `MemoryService`.
- [x] Harden continuation cursors against history replacement and make
  identical feedback retries idempotent, including recovery from a pending
  receipt. Pending retries revalidate supporting evidence and only accept an
  updated hash when the same operation already changed that target.
- [x] Validate document enrichments and proposed query rules before reserving
  a search ID as pending, so rejected proposals can be corrected and resubmitted
  with that same search ID.
- [x] Verify that a provider-added key phrase makes an otherwise unmatched
  query retrieve its source after publication, without changing the source
  text, and that repeated enrichment is a no-op.
- [x] Verify that a query-only rule submitted through `search_feedback` changes
  query interpretation immediately in the same `MemoryService` instance.
- [x] Reject stale concurrent source-document edits and require a refreshed
  evidence hash before merging a second provider's additive contribution; test
  that the first provider's phrase remains after the second update.
- [x] Preserve evidence-backed enrichment across unchanged source refreshes and
  invalidate it after a cited source changes or is deleted. Record provider
  phrase ownership in the persisted contribution and cover the lifecycle with
  a focused regression test.
- [x] Extend document enrichment to managed session conclusions, headings, and
  validated links on existing session-context documents; preserve them across
  later hook captures. Preserve raw hook observation links even when the
  feedback contribution contains no links, while continuing to invalidate
  feedback-owned links with their supporting evidence. Automatic conclusion
  generation remains future work.
- [x] Accept evidence-cited, same-scope, whole-document supersession through
  the existing semantic relation path. Both documents must belong to the same
  backing index. Publish before returning the existing adjudication receipt;
  test captured-store and workspace-store corrections, retry, cross-scope
  rejection, and cross-store rejection.
- [ ] Add claim-level correction that preserves unrelated facts within a
  multi-fact document, with typed consumers and regression coverage.
- [x] Accept evidence-grounded query-only rule additions through
  `search_feedback`; unsupported and extraction-impacting rules remain rejected.
- [ ] Measure retrieval effects and maintenance cost on controlled replays.
  Focused coverage verifies that overall and per-document judgments do not
  mutate source documents, and that `enrich_doc` changes only its selected
  document after exact target and evidence hash validation. Representative
  retrieval evaluation and maintenance-cost measurement remain open.
- [x] Document the three-tool review, search feedback, and document enrichment flow, evidence requirements,
  explicit and resident-inferred session attribution, and session-context enrichment.
- [x] Exercise the shared provider router end to end: page unreviewed searches,
  retrieve one by `search_id`, submit complete per-document judgments, then confirm the exact search
  is marked reviewed. Muse Code now also exposes the shared review and feedback
  schemas and dispatches these calls through the same router in both resident
  and in-process modes; `integrations::muse_code::tests::search_review_tools_are_listed_and_work_through_muse_mcp`
  verifies list, page, exact lookup, feedback, and reviewed state. Hermes uses
  the same Gemini MCP implementation with Hermes provider identity;
  `integrations::hermes::tests::hermes_search_review_and_feedback_use_shared_memory_service_route`
  verifies the same review lifecycle there.
- [ ] Document extraction-impact rule reload and provider capability
  differences as those behaviors are implemented.

## Acceptance checks

Correctness checks must cover:

- Raw observation text and repository files remain unchanged.
- Older persisted `KeyPhrase` values without provider/extractor provenance
  deserialize with an unspecified origin; `source::tests::legacy_key_phrases_without_origin_still_deserialize`
  verifies that adding annotation ownership does not invalidate existing source
  documents.
- Repeating enrichment is idempotent and does not duplicate terms or links.
- Two providers updating the same document cannot silently lose either update.
- Changed or deleted dependencies invalidate outdated contributions; extractor,
  watcher, and hook refreshes preserve still-valid contributions. The focused
  MemoryService regression test covers unchanged evidence preservation and
  invalidation after both evidence content changes and evidence deletion;
  broader integration coverage remains needed.
- Conclusions include earlier session observations, not only the latest batch.
- Project, user, branch, and provider provenance stay applicable. Cross-provider
  reuse within an authorized project does not allow cross-user access.
- `review_search` includes earlier documents from a search's attributed session,
  even when those documents were not in the result list, while an absent session
  attribution exposes only the result documents. The focused regression test is
  `review_search_includes_attributed_session_history_but_not_absent_sessions`.
- Corrections preserve unrelated facts and do not treat copied output as
  independent verification.
- Restart, lease expiry, publication failure, and interrupted submission recover
  without losing accepted work or acknowledging unpublished data as visible.
- Enrichment behaves consistently through IPC, configured HTTP, and filesystem
  fallback, and works with both supported index layouts.
- Persisted enrichment remains usable when an optional extraction backend is
  unavailable; implementing this feature does not require enabling beKIND.
- A key-phrase-only enrichment changes matching/routing after publication while
  retaining original source text; identical enrichment triggers no rebuild.
- Segmented enrichment rebuilds affected segments and reuses unrelated ones.
  Compare results against a full rebuild oracle, including semantic dependencies.
  `segmented_feedback_enrichment_rebuilds_only_its_segment_and_matches_full_rebuild`
  verifies the MemoryService enrichment path rebuilds the target segment, keeps
  an unrelated segment shared, and matches a full-rebuild search result for a
  focused query. `segmented_feedback_supersession_matches_full_rebuild_across_segments`
  verifies that whole-document supersession updates visibility across distinct
  segments, preserves an unrelated segment index, and matches a full rebuild.
- Query-only rule updates do not re-extract documents. Index-time rule updates
  invalidate the appropriate records and do not silently reuse stale claims.
- Current-state searches return applicable information, rationale searches can
  retrieve change explanations, and explicit old/new comparisons retrieve both
  sides or report insufficient coverage. Verify through both index layouts.
- Valid rule edits take effect without restart. Cover atomic rename, same-time
  content changes, repeated notifications, missed-event recovery, settings path
  changes, overlay deletion, and recreation.
- Invalid, unsupported, partially written, or unreadable rules retain the last
  valid snapshot and expose a useful error. Formatting-only edits do no work.
  `watcher_keeps_last_good_rules_on_invalid_edit_and_ignores_formatting_only_changes`
  verifies stable snapshot identity for equivalent content, last-good retention,
  visible parse errors, and automatic activation after repair.
- Concurrent queries pin one rule generation. Two projects keep independent
  overlays, and late reload/build completion cannot replace newer accepted work.
  `rule_reload_keeps_pinned_queries_and_other_projects_isolated` verifies that
  a prepared query keeps its original terms while new queries use the reloaded
  terms, and that the reload does not alter a second project's rules.
  `workspace_service_applies_external_query_rule_edit_without_restart` verifies
  that a query made through an already-running workspace service observes an
  external overlay edit without a restart or explicit reload call.
- Query-only reload changes behavior and cache identity without document
  re-extraction. Extraction reload automatically builds and publishes a compatible
  rules/index pair, preserves concurrent writes, and recovers after restart.

For retrieval checks, freeze a corpus and recorded queries, enrich one copy,
and compare against an unchanged control. Keep tuning queries and paraphrase
families separate from evaluation queries. Inspect ranking changes and regressions
alongside search p50/p95/p99, enrichment token use, publication cost, and index
growth. Distinguish extraction, matching, routing, ranking, and excerpt failures.

Use Recall@k only when expected relevant sources are independently established.
Otherwise report rank changes and regression outcomes. Historical search
evaluation must use the documents and conclusions available at that time;
later session answers cannot be introduced and counted as historical success.
The earlier informal MCP-history relevance estimate is not a release benchmark.

## Current verification snapshot

On 2026-10-15, `cargo test --lib --features agent-integrations` passed all
945 tests in 64.76 seconds. This includes the feedback and MemoryService
enrichment regressions, resident writer coverage, and automatic query-rule
reload tests. It does not close the open plan items above, including
automatic session conclusion generation, claim-level correction,
extraction-impact rule publication, and broader controlled retrieval and
maintenance-cost measurement.
