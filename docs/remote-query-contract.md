# Remote Query Contract

Status: the transport-neutral Rust types, validation, statistics aggregation,
and deterministic candidate reduction are implemented. The transport schema,
`lint-service` adapter, end-to-end two-phase orchestration, and operational
failure handling remain pending.

This is the wire-level contract for placing the local segmented query
coordinator behind `lint-service`. It is deliberately separate from the
transport mechanism: the current service uses gRPC, but the same messages can
be carried by HTTP or another RPC system.

The Rust data model for this contract lives in `src/remote_query.rs` and is
re-exported from the crate root. Its `aggregate_statistics` and
`reduce_candidates` helpers are the canonical validation/reduction boundary.
They are transport-neutral and intentionally do not change the legacy
`RunLint` protocol. A protobuf or other transport schema should be added with
the eventual `lint-service` adapter and generated bindings so its parity with
these types can be tested.

## Deployment model

`lint-service` is the cross-corpus coordinator. Each worker owns one corpus and
runs a local `lint-ai` coordinator, which may itself route across segments.
Worker and segment identifiers are therefore scoped independently:

```text
lint-service snapshot
  ├── worker-a snapshot ── segment-a1, segment-a2
  ├── worker-b snapshot ── segment-b1, segment-b2
  └── worker-c snapshot ── segment-c1
```

The service must never merge local scores merely because they are both called
`score`. Scores are valid for reduction only when their statistics contract and
statistics generation match the coordinator's query.

## Versioning and identity

The common request envelope used by both phases carries:

- `protocol_version`: monotonically versioned wire contract;
- `request_id`: globally unique ID used for logs and deduplication;
- `deadline_unix_ms`: absolute deadline, not an independently reset timeout;
- `coordinator_generation`: the service's worker-set/configuration snapshot;
- `query`: the original query;
- `top_k`, filters, temporal bounds, and reference date.

The statistics request additionally carries parsed query terms and field names.
The candidate request carries the frozen `statistics_generation` and aggregated
statistics payload produced after phase 1.

Every response carries the request ID, worker ID, worker snapshot generation,
statistics generation, and a protocol version. A response from a different
request, incompatible protocol, or incompatible statistics generation is a
failed worker result—not an unlabelled candidate.

## Two-phase exchange

Exact cross-corpus BM25 scoring needs corpus-wide document frequencies and token
totals before workers score candidates. The coordinator therefore uses two
bounded phases.

### Phase 1: statistics

The coordinator sends the parsed query terms and the relevant field names to
each eligible worker. Each worker returns query-scoped statistics:

```json
{
  "protocol_version": 1,
  "request_id": "q-123",
  "worker_id": "worker-a",
  "worker_generation": 42,
  "statistics_generation": "q-123:stats",
  "fields": {
    "content": {"documents": 1200, "tokens": 84000},
    "headings": {"documents": 1200, "tokens": 4100}
  },
  "terms": {
    "routing": {"content": 18, "headings": 7},
    "deployment": {"content": 91, "headings": 22}
  },
  "completeness": {
    "expected_segments": ["segment-a1", "segment-a2"],
    "successful_segments": ["segment-a1"],
    "failures": [{"segment_id": "segment-a2", "message": "deadline"}]
  }
}
```

The implemented `aggregate_statistics` helper validates request and protocol
identity, rejects duplicate workers or mismatched statistics generations, sums
successful worker statistics, and merges segment completeness. The future
service adapter must freeze that result under the shared
`statistics_generation`. Missing workers must be represented by the adapter;
partial statistics must never be presented as complete. A service may continue
with an explicitly partial scoring view or fail closed according to its request
policy.

### Phase 2: candidates

The coordinator sends the frozen statistics generation and statistics payload
to each worker. A worker scores only its own segment-local postings, returning
stable document IDs, score components, and local completeness:

```json
{
  "protocol_version": 1,
  "request_id": "q-123",
  "worker_id": "worker-a",
  "worker_generation": 42,
  "statistics_generation": "q-123:stats",
  "results": [{
    "doc_id": "worker-a:doc-7",
    "source": "docs/deploy.md",
    "group_id": "release-42",
    "score": 4.81,
    "score_breakdown": {"lexical_score": 2.9, "semantic_score": 1.91}
  }],
  "completeness": {
    "expected_segments": ["segment-a1", "segment-a2"],
    "successful_segments": ["segment-a1"],
    "failures": [{"segment_id": "segment-a2", "message": "deadline"}]
  }
}
```

The worker must reject a statistics generation it did not receive or cannot
apply. The implemented `reduce_candidates` helper validates the generation,
rejects duplicate worker responses, deduplicates candidates by stable document
ID, sorts by score and then document ID, and truncates only after merging.
Cross-worker group/session aggregation is not part of the current remote
reducer; workers must return results at the ranking granularity expected by the
coordinator.

## Required service behavior

The local contract helpers validate messages and preserve segment-level
completeness, but they do not run discovery, authentication, RPCs, retries, or
deadline cancellation. The future `lint-service` adapter must provide the
following behavior:

- The deadline is shared across discovery, both phases, and reduction.
- A worker failure is represented in completeness and does not erase successful
  results.
- A failed statistics phase must not be represented as a fully complete answer.
- If every worker fails, the service returns an unavailable error.
- No automatic retry may duplicate a non-idempotent operation; statistics and
  candidate requests must be read-only and safe to retry only with the same
  request ID.
- Authentication and tenant scope are checked before either phase.

## Compatibility with current `lint-service`

The existing `RunLint` RPC carries arbitrary CLI arguments and returns stdout,
stderr, exit code, and worker results. It remains useful for generic lint-job
federation, but it cannot prove the contract above because it has no typed
statistics phase, generation fields, or structured segment completeness.

The migration should add a new RPC rather than reinterpret `RunLint`. Existing
CLI dispatch keeps its behavior; typed distributed query clients opt into the
new protocol and can be rolled out worker-by-worker.

## Implementation status and acceptance criteria

Local unit tests currently cover wire serialization and request validation,
statistics aggregation, generation mismatch rejection, completeness validation,
and order-independent candidate reduction. The remote service implementation is
ready only when integration or contract tests additionally prove:

1. identical query-scoped statistics produce comparable scores across two
   workers;
2. a stale or mismatched generation is rejected and reported across the RPC
   boundary;
3. one failed worker/segment yields partial results with explicit completeness;
4. the same request ID and deadline are preserved through both phases;
5. service response ordering remains deterministic across worker arrival orders;
6. all-worker failure returns an unavailable result;
7. the legacy `RunLint` path remains unchanged.
