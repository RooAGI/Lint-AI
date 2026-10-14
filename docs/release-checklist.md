# Release checklist

Use this checklist for every Membrick release. Complete the preparation and
benchmark work before opening the release PR. Merge the PR before creating its
`vX.Y.Z` tag. Check each item or record why it does not apply.

## Release files and what they control

Update only the files for components included in the release. Keep package
versions aligned with the tag, and regenerate each lockfile with its package
manager.

| Release area | Files to update | Purpose |
|---|---|---|
| Rust CLI and library | `Cargo.toml`, `Cargo.lock` | Crate version and locked dependency/package metadata |
| Python package | `pyproject.toml`, `uv.lock` | Python distribution version and lockfile |
| Node package | `bindings/node/package.json`, `bindings/node/package-lock.json`, `bindings/node/Cargo.toml`, `bindings/node/Cargo.lock` | npm package and native binding versions |
| Node platform packages | `bindings/node/package.json` (`napi.targets` and `optionalDependencies`) and `.github/workflows/node-publish.yml` | The workflow builds and publishes the five platform packages named in the root package manifest |
| OpenClaw plugin | `src/integrations/openclaw/plugin/package.json`, `src/integrations/openclaw/plugin/openclaw.plugin.json` | Plugin package version and discovery metadata |
| Helm chart | `charts/membrick/Chart.yaml` | Chart version and application version |
| Release notes | `docs/releases/X.Y.Z.md` | User-visible changes, upgrade steps, compatibility, and known issues |
| Website content | `docs/index.md` for homepage claims; relevant `docs/*.md` pages for guides and benchmarks; `mkdocs.yml` for navigation | Published documentation site. `README.md` updates the GitHub repository page; it does not update the website homepage |
| Benchmark evidence | `comparison/results/<run>.json` or `.json.gz`, plus `docs/benchmark-results.md` | Reproducible report artifacts and the human-readable summary |
| Publishing workflows, when needed | `.github/workflows/release.yml`, `crate-publish.yml`, `python-publish.yml`, `node-publish.yml`, `docs.yml` | Build and publish behavior. Change these only when the workflow itself needs a fix |

GitHub Actions permissions, environments, npm credentials, and registry trusted
publisher settings are configured in GitHub or the package registries, not in
these repository files. Verify them as separate release prerequisites.

## Publish sequence

1. Update the applicable version files, release notes, benchmark evidence, and
   website pages in one release PR. Update `README.md` separately when its
   repository overview or benchmark summary needs to change.
2. Review the file map above, generated manifests, lockfiles, package contents,
   and version consistency. Run the required checks and benchmarks, then merge
   the PR to `main`.
3. Create and push one annotated `vX.Y.Z` tag on the merged commit. The tag
   starts the GitHub release, crates.io, PyPI, and npm publication workflows.
4. Confirm every workflow validates the same tag version and finishes
   successfully. Check the GitHub release assets and every registry package,
   including platform-specific npm packages.
5. Confirm the documentation workflow deploys the merged `docs/` changes.
   README changes alone do not trigger or update the website homepage.
6. Record the merged commit, tag, workflow run links, published package
   versions, and any exceptions in the release record.

## Before opening the release PR

The v0.5.0 tag has been pushed. Use the checklist below to track publication
and post-release verification; do not treat a pushed tag as proof that every
package and website deployment is available.

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
- [x] Run the required benchmarks on a clean, optimized candidate build. The
  current PR head `ac0d23d` was evaluated in release mode with locked dependencies
  on the same host, dataset, and settings as the prior v0.4.0 comparison. The
  command, feature set, host, dataset hash, per-question results, and summary
  are recorded in `benchmark-results.md`; the dated raw report is preserved.
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

### v0.5.0 publication status

The `v0.5.0` tag points to merged commit `b204b7e`. At the latest check on
2026-10-14, the GitHub binary release and Node publication were running, the
Python publication was queued, and the Rust crate workflow failed at startup
before creating a job. The failed Rust run cannot be retried in place. Do not
mark v0.5.0 publication complete until the available workflows and registry
versions have been checked, and the Rust publication has a documented recovery.

- [GitHub release](https://github.com/RooAGI/Membrick/actions/runs/38640167122)
- [Node publish](https://github.com/RooAGI/Membrick/actions/runs/38640167124)
- [Python publish](https://github.com/RooAGI/Membrick/actions/runs/38640167375)
- [Rust crate publish](https://github.com/RooAGI/Membrick/actions/runs/38640168831)

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
