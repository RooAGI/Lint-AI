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

beKIND evidence signals are an optional enrichment. When enabled, Lint-AI
accepts signal protocol versions 1 and 2, verifies that the evidence span
matches the stored document content, and stores the result in `top_claims`.
Version 2 supplies status and explicitly stated event time. The core claim
model and old indexes do not require beKIND to be present.

## Native reasoning cues

The first provider-neutral event, acquisition, and related query-language cues
are stored as readable JSON in [`data/reasoning_rules.json`](../data/reasoning_rules.json).
Lint-AI parses this bundled file once per process. The file contains vocabulary
and phrase patterns; typed Rust code still decides how they produce claims and
query requirements, and still applies evidence and temporal checks. Changing
the bundled rules currently requires rebuilding Lint-AI. Project-specific
runtime overrides are not part of this first step.
