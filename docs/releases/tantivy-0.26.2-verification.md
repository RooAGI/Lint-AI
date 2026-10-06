# Tantivy 0.26.2 verification

Measured 2026-10-04 before accepting the dependency upgrade for Lint-AI 0.3.0.

## Results

| Check | Result |
|---|---|
| Existing indexes | 0.25-created indexes opened; lexical file hashes unchanged on all 20 restarts |
| Search correctness | Targeted records preserved in all runs; eight varied-corpus queries preserve ordered IDs in both layouts |
| Rust suite | 803 passed, zero failures, with agent integrations enabled |
| Hermes plugin suite | 39 passed, zero failures |
| Compilation | All features and all targets passed |
| Throughput | No observed regression in either layout at C=1 or C=10 |
| Remaining advisory | Trigger unreachable in the inspected Tantivy cache usage; warning remains in cargo audit |

The upgrade is applied with explicit `.order_by_score()` collectors at the
query entry points. No index migration or forced rebuild was needed.

## Throughput protocol

Both binaries were built from the same working tree, with only dependency
resolution and the required collector API migration differing. Release builds
used `--features agent-integrations`. Each layout contained 23,366 records
spread across five sessions. Each version ran five repetitions at C=1 and C=10,
with 1,000 measured requests and 100 warmup requests per cell. Version order
alternated between repetitions. Every server started on a fresh copy of an
index created by 0.25. Builds and tests finished before timing started.

The HTTP client uses urllib and a thread pool; throughput includes HTTP and
client overhead. Medians below describe this workload and machine, rather than
a universal capacity estimate. This protocol differs from the historical
100-request throughput runs. The five-segment corpus exercises distributed
search work, unlike a layout where routing finds only one relevant segment.

| Layout | Concurrency | 0.25.0 req/s | 0.26.2 req/s | Change |
|---|---:|---:|---:|---:|
| Single index | 1 | 376.70 | 601.71 | +59.7% |
| Single index | 10 | 2,246.69 | 2,769.53 | +23.3% |
| Routed segments | 1 | 332.79 | 354.91 | +6.6% |
| Routed segments | 10 | 888.86 | 901.29 | +1.4% |

The small routed C=10 difference should be read as no observed regression,
not evidence of a meaningful improvement.

Raw measurements and binary hashes are in
`comparison/results/throughput-tantivy-upgrade-2026-10-04.json`.

## Compatibility scope and score changes

The large old-index fixtures preserve the expected top document for records
42, 123 and 23365. An absent term remains empty. Separately, ten varied documents
cover deployments, databases, dependencies, concurrency and recipes. All eight
queries preserve the entire ordered result-ID list in both single and routed
layouts. The raw responses are in
`comparison/results/search-tantivy-upgrade-2026-10-04.json`.

Numeric scores are **not identical** across versions. For example, the single
index query `decision 42` retains the same top record but its score changes from
26.70757 to 13.43378. Broad searches over the repetitive 23,366-record corpus
also change tied ordering, including on restarts of 0.25 itself. The artifacts
retain these differences; byte-identical HTTP responses are not a passed check.
These tests establish compatibility for the exercised indexes and queries,
not unchanged ranking for every corpus. A follow-up full LongMemEval rerun on 2026-10-05 found lower quality metrics
with Tantivy 0.26.2. See the versioned rerun tables in
docs/benchmark-results.md; the pre-upgrade headline does not describe 0.26.2.

## Remaining lru advisory

The local resolution is Tantivy 0.26.2 with lru 0.16.4.
[RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002.html)
is fixed. Cargo audit still reports
[RUSTSEC-2026-0253](https://rustsec.org/advisories/RUSTSEC-2026-0253.html),
which concerns `LruCache::pop` and a key destructor panicking, followed by
continued cache use after unwinding.

Inspection of Tantivy 0.26.2 source finds its only LruCache in
`src/store/reader.rs`: `Mutex<LruCache<usize, Block>>`, with `Block = OwnedBytes`.
Production methods use `get`, `put`, and construction; they do not call `pop`
or `iter_mut`. Integer keys have no custom destructor and cannot trigger the
advisory's key-drop panic. Therefore this specific trigger is unreachable in
that usage. This is a source assessment, not a patched dependency or a clean
audit. Reassess if Tantivy changes cache key types or operations, or another
consumer introduces lru. The repository ignores Cargo.lock, so the exact local
resolution is not a committed dependency pin.

## Reproduction

Save the 0.25 release binary before upgrading. After compiling the 0.26.2 binary:

```bash
python3 comparison/tantivy_upgrade.py prepare \
  --baseline /path/to/server-0.25.0 \
  --work-dir target/tantivy-verification
python3 comparison/tantivy_upgrade.py compare \
  --baseline /path/to/server-0.25.0 \
  --candidate /path/to/server-0.26.2 \
  --work-dir target/tantivy-verification \
  --output comparison/results/throughput-tantivy-upgrade.json
python3 comparison/tantivy_upgrade.py verify \
  --baseline /path/to/server-0.25.0 \
  --candidate /path/to/server-0.26.2 \
  --work-dir target/tantivy-search-verification \
  --output comparison/results/search-tantivy-upgrade.json
cargo test --all-targets --features agent-integrations
cargo check --all-features --all-targets
python3 -m unittest discover -s src/integrations/hermes/plugin/tests
cargo audit
```

Use fresh work directories for preparation and varied-corpus verification.
Existing fixtures are never overwritten.
