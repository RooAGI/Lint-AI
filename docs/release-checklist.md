# Release checklist

Use this checklist for every Membrick release. Complete the preparation and
benchmark work before opening the release PR. Merge the PR before creating its
`vX.Y.Z` tag. Check each item or record why it does not apply.

## Before opening the release PR

Current v0.5.0 candidate: PR #110, branch `release/v0.5.0-ipc-architecture`. Checked items below have evidence in the linked release notes, metadata, and benchmark pages. Remaining boxes are open gates.

- [x] Choose the version using semantic versioning: patch for fixes, minor for
  compatible features, major for breaking changes.
- [ ] Confirm the release scope and identify migrations, configuration changes,
  compatibility limits, and known issues.
- [x] Confirm version `0.5.0` is unused in GitHub Releases, crates.io, PyPI,
  and npm. Checked the direct registry endpoints and release tag on 2026-10-14;
  all report not found. Check each platform package immediately before publish.
- [x] Confirm GitHub publishing prerequisites: `NPM_TOKEN` is present; the
  `pypi` and `crates-io` environments exist; and the required publish actions
  are allowed by the repository Actions policy.
- [ ] Verify account-side PyPI pending trusted publisher and crates.io Trusted
  Publisher settings for `RooAGI/Membrick` and the matching workflow names.
- [x] Update the Rust crate, Python package, Node package and platform packages,
  OpenClaw plugin and its npm manifest, Helm chart, and application version to
  `0.5.0`.
- [x] Add release notes at `docs/releases/X.Y.Z.md` covering user-visible
  changes, upgrade steps, compatibility, and known limitations.
- [x] Update installation, configuration, architecture, and website docs that
  the release changes.
- [x] Check state migration against existing user data: a release-binary
  startup moved `.lint-ai` to `.membrick`, preserved non-conflicting data,
  archived a conflicting legacy file, and left no `.lint-ai` directory.
- [ ] Keep unrelated workspace changes out of the release PR.
- [x] Select release benchmarks: full LongMemEval-S, corpus scale, repeated
  IPC/HTTP read-write-mixed workloads, lifecycle startup, and affected provider
  integrations. The benchmark runs were started before explicit numerical
  regression limits were recorded; document that process exception and review
  results before merge.
- [ ] Run the required benchmarks on a clean, optimized candidate build. Compare
  with the previous tagged release using the same machine, dataset, settings,
  query set, result depth, and harness wherever possible. Record the commit SHA,
  build features, configuration, command, host, dataset identity, raw output,
  and summary. Keep dated artifacts; do not replace prior results.
- [x] **LongMemEval-S retrieval quality:** run all 500 questions through the
  production retrieval settings, including structured-fact retrieval enabled
  and the documented release NER backend. The v0.4.0 tag and exact v0.5.0 PR
  head runs have matching retrieval metrics; dependency-lock and timing limits
  are documented in `benchmark-results.md`.
- [x] **Per-question retrieval diff:** the v0.5.0 candidate has three changed
  top-20 lists versus v0.4.0, with unchanged MRR and NDCG, no empty results,
  and no Any-hit@5 regressions. Full reports are linked in
  `benchmark-results.md`.
- [x] **Temporal and multi-session subsets:** not required for v0.5.0. The
  release retrieval gate uses all 500 LongMemEval-S questions; the 133-question
  subsets are optional diagnostics and are excluded from release claims.
- [x] **Corpus-scale indexing and search:** compared v0.4.0 and v0.5.0 on
  1k/5k/10k/19,829 LongMemEval-S sessions with 20 or 100 eligible queries;
  recall matched at every size. Build and p50/p95 query timing are recorded in
  `benchmark-results.md`. Memory use was unavailable. The v0.4.0 tag lacks a
  Cargo.lock, so its dependency resolution and timing are diagnostic.
- [x] **Transport performance:** when the server, IPC, or HTTP path changes,
  compare IPC and HTTP with the same read, write, and mixed read/write workload.
  Three 10-second runs per transport used the same optimized release binary,
  four search callers, and one writer. Throughput, p50/p95/p99, startup time,
  errors, and post-ack write visibility are recorded in
  `comparison/results/release-v0.5.0-ipc-http-mcp-18339c0.json` and summarized
  in `http-server-benchmarks.md`. Run-to-run spread was substantial; no speedup
  claim is made.
- [ ] **Cold-start and lifecycle:** when resident startup or agent hooks change,
  measure first MCP call/startup separately from warm calls and confirm that
  concurrent clients share one owner without losing writes.
- [ ] **Agent integration scenario:** when adapters, hooks, context injection,
  session capture, or memory behavior changes, run the affected real-runtime
  scenarios. For shared behavior, run the same scenario on Claude Code, Codex,
  and AGY and report retrieved documents, injected context, correctness,
  latency, tokens, and tool calls.
- [ ] Review benchmark results against the limits set above. Investigate every
  material regression; document accepted trade-offs and the reason for them.
- [ ] Save commands, per-question outputs, summaries, and runtime transcripts
  where applicable under a dated benchmark-results path. Redact secrets and
  obtain consent before retaining external users' transcripts.
- [ ] Review the release changes and confirm every included change belongs in
  this release.

## In the release PR

- [ ] Full Rust test matrix passes, including supported agent integration
  features.
- [ ] Agent install contract and Unix and Windows installer checks pass.
- [ ] Python binding tests and wheel or source package build checks pass.
- [ ] Node binding tests and native builds pass on every supported target.
- [ ] Security audit passes, or each finding has a documented disposition.
- [x] Strict MkDocs documentation build and link checks pass locally; confirm
  the docs workflow on the updated PR head as well.
- [x] Version metadata matches across release components and matches the
  proposed `vX.Y.Z` tag.
- [x] `cargo package --locked --list` and `cargo publish --dry-run --locked`
  succeed for the Rust crate.
- [ ] Package contents include the expected CLI, native libraries, licenses,
  and metadata for each published package.
- [x] Include benchmark commands, artifacts, and a concise results summary in
  the PR. Explain any waived benchmark and its reason. The repeated IPC/HTTP
  benchmark script and raw artifact are committed in `5d23499`.
- [ ] All required reviews are complete and the release PR is merged to `main`.

## After the release PR is merged

- [ ] Confirm `main` contains the merged release commit and required checks pass
  on that commit.
- [ ] Create an annotated `vX.Y.Z` tag on the merged commit. Do not tag an
  unmerged release branch.
- [ ] Push the tag once. Confirm the release, crate, Python, and Node workflows
  all started from that tag and each workflow validates the same version.
- [ ] Wait for all build jobs to pass before treating the release as available.
- [ ] Confirm GitHub release binaries and `SHA256SUMS` are present and match the
  built artifacts.
- [ ] Confirm registry publication succeeded for every package included in the
  release. Check all Node platform packages as well as the root package.
- [ ] Install the CLI from the release instructions on supported operating
  systems and verify its reported version.
- [ ] Install the published Python and Node packages in clean environments and
  verify imports, CLI availability, and package versions.
- [ ] Verify the Rust crate can be installed from crates.io when it is part of
  the release.
- [ ] Run a short smoke test for the released default transport and the
  documented setup path.
- [ ] Confirm the website has deployed the release notes and updated guides.
- [ ] Record the tag, commit, workflow runs, registry versions, and any
  exceptions in the release record.

## If a release job fails

- [ ] Identify whether the version was published to any registry before
  retrying.
- [ ] For an unpublished artifact, fix the workflow or package issue and rerun
  the job for the same tag after checking its version guard.
- [ ] For an already published artifact, do not try to overwrite that version.
  Verify its contents, complete other registries only when they can use the
  same source, or prepare a new patch release if the artifact is wrong.
- [ ] Do not move or recreate a release tag to point at different source.
- [ ] Update the release record with the partial publication and recovery
  outcome.
