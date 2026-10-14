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
- [ ] Confirm the version is unused on GitHub Releases, crates.io, PyPI, npm,
  and any other registry being published to.
- [ ] Confirm publishing access and required GitHub environments are configured.
- [x] Update the Rust crate, Python package, Node package and platform packages,
  OpenClaw plugin, Helm chart, and application version to the intended release
  version. If a component is versioned independently, document that exception.
- [x] Add release notes at `docs/releases/X.Y.Z.md` covering user-visible
  changes, upgrade steps, compatibility, and known limitations.
- [x] Update installation, configuration, architecture, and website docs that
  the release changes.
- [ ] Check migration behavior against existing user data and configuration.
- [ ] Keep unrelated workspace changes out of the release PR.
- [ ] Decide which benchmark tracks are required based on the release changes.
  Set acceptable regression limits before running them.
- [ ] Run the required benchmarks on a clean, optimized candidate build. Compare
  with the previous tagged release using the same machine, dataset, settings,
  query set, result depth, and harness wherever possible. Record the commit SHA,
  build features, configuration, command, host, dataset identity, raw output,
  and summary. Keep dated artifacts; do not replace prior results.
- [ ] **LongMemEval-S retrieval quality:** run all 500 questions through the
  production retrieval settings, including structured-fact retrieval enabled
  and the documented release NER backend. Compare Any-hit and Fractional
  Recall at @5/@10/@20, MRR, NDCG@10, and latency with the prior release.
- [ ] **Per-question retrieval diff:** report questions whose retrieved IDs or
  ranks changed, inspect regressions and misses, and confirm there are no
  unexpected empty retrievals. Aggregate scores alone are insufficient.
- [ ] **Temporal and multi-session correctness:** run the existing temporal
  and multi-session benchmark track when retrieval, claim reconciliation,
  indexing, or memory publication changes. Record the exact subset and mode;
  keep beKIND-on and beKIND-off results separate.
- [ ] **Corpus-scale indexing and search:** run the representative corpus scale
  benchmark when indexing, persistence, segmentation, or startup changes.
  Record build/reopen time, memory use if available, and query latency/quality.
- [ ] **Transport performance:** when the server, IPC, or HTTP path changes,
  compare IPC and HTTP with the same read, write, and mixed read/write workload.
  Use release binaries, repeated runs, and report throughput plus p50/p95/p99,
  errors, and write visibility after acknowledgement.
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
- [ ] Documentation build and link checks pass.
- [ ] Version metadata matches across release components and matches the
  proposed `vX.Y.Z` tag.
- [ ] `cargo package --locked --list` and `cargo publish --dry-run --locked`
  succeed when publishing the Rust crate.
- [ ] Package contents include the expected CLI, native libraries, licenses,
  and metadata for each published package.
- [ ] Include benchmark commands, artifacts, and a concise results summary in
  the PR. Explain any waived benchmark and its reason.
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
