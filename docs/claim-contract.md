# Structured claim contract

`DocRecord.top_claims` stores provider-neutral claims as JSON. The contract is
backward compatible: older records deserialize with `status`, `event_time`,
and `evidence` unavailable.

Each claim contains:

| Field | Meaning |
| --- | --- |
| `subject` | Who or what performs or participates in the relation. |
| `predicate` | Canonical relation, such as `participated_in`. |
| `object` | The event, person, or thing related to the subject. |
| `status` | Optional `planned`, `ongoing`, `completed`, or `negated` assertion state. |
| `event_time` | Optional time asserted for the event, distinct from the source document timestamp. |
| `evidence` | Source document, optional turn/chunk identifiers, exact supporting text, and optional UTF-8 byte offsets. |

Missing optional values mean the producer could not establish them from the
source. A document timestamp must not be copied into `event_time`: it describes
when the source was recorded, not necessarily when the claimed event occurred.

beKIND evidence signals are an optional enrichment. When enabled, Membrick
accepts signal protocol versions 1 and 2, verifies that the evidence span
matches the stored document content, and stores the result in `top_claims`.
Version 2 supplies status and explicitly stated event time. The core claim
model and old indexes do not require beKIND to be present.

## Native reasoning cues

The first provider-neutral event, acquisition, and related query-language cues
are stored as readable JSON in [`data/reasoning_rules.json`](https://github.com/RooAGI/Membrick/blob/main/data/reasoning_rules.json).
Membrick parses this bundled file as its baseline. The file contains vocabulary
and phrase patterns; typed Rust code still decides how they produce claims and
query requirements, and still applies evidence and temporal checks. A workspace
can extend supported query cues in `.membrick/rules.json`; valid query-only
edits reload automatically, and `.membrick/settings.toml` can select another
rules path. Rules that change extraction of stored claims are not activated by
this query reload path and require re-extraction support before they can be
used safely. Extraction-impacting overlay changes still require a compatible
re-extraction and index publication, which is not yet implemented.

Query rules include comparison, aggregation, temporal, current-state, and
rationale terms. Recognized current-state cues map to the single-scope
`QueryIntent::CurrentState`; they express what the caller wants, while existing
semantic visibility determines which facts remain applicable. A `Why`
question is represented as `QueryIntent::Explain`; the
bundled `query_explanation_terms` (`reason`, `rationale`, `decision`, and
`because`) are added to its augmented search query. A workspace can add
project terminology to that list, and the existing rule watcher makes it
available to later queries without restarting the resident. This shapes the
query; it does not guarantee that a relevant explanation document exists or
that every implicit reference has been resolved.
