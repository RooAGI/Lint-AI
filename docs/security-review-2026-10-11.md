# Security review, 2026-10-11

Status: in progress. This review does not yet establish that all security issues are resolved.

## Confirmed findings

- Workspace MCP endpoints accepted a matching tenant identity but dispatched workspace-wide retrieval and mutation. A behavioral HTTP regression reproduced HTTP 200 where HTTP 403 was required. Block both MCP endpoints in tenant mode, consistent with the existing workspace-hook restriction. The test failed before the fix, passed after the fix, failed again after reverting only the fix, and passed again after restoration. Logs: `/tmp/lint-ai-tenant-mcp-{red,green,revert,restored}.log`.
- Provider-memory search clears the requested user identity for server-token requests even when tenant mode is configured. The tenant restriction must remain applied to retrieval. A behavioral HTTP regression exposed Bob's secret to Alice. The fix preserves the tenant user filter and binds conversation scope to the tenant. Red, green, reverted red, and restored green completed. Logs: `/tmp/lint-ai-tenant-search-{red,green,revert,restored}.log`.
- The browser dashboard's unauthenticated local API accepted requests without adequate same-origin validation. The API now checks the Host/Origin relationship and `Sec-Fetch-Site`; browser regression tests passed red/green/revert/restored.
- Conversation-state reads, locks, expiration cleanup, and writes now reject symlink redirection and use private file permissions. The targeted `conversation_state::tests` suite passed (15 tests).
- Opt-in session recording now rejects symlinked session-directory paths and reads manifest/event files through symlink-resistant file opens. `event_recording_rejects_symlinked_session_directory` passed after the fix and failed when the guard alone was removed. The fixed source was restored and the targeted test passed again. Logs: `/tmp/lint-ai-recording-symlink-{final,revert}.log`.
- Vendored `lru` backports address the two unsoundness advisories without waiting for Tantivy's dependency update. Regression tests passed red/green/revert/restored; the iterator case also passed Miri. Details and provenance are in `vendor/lru/SECURITY-PATCHES.md`.

## Dependencies

`cargo audit --json` completed. No entries were reported in the vulnerabilities category. It still reports the two informational unsoundness advisories for transitive `lru` 0.12.5 through Tantivy 0.25.0, plus unmaintained `number_prefix` and `paste` packages. The `lru` fixes are backported into `vendor/lru` and selected through the root Cargo patch, but the audit warning remains because the dependency version is unchanged.

RustSec reports `lru` 0.18.2 as patched for RUSTSEC-2026-0253 and 0.16.3 as patched for RUSTSEC-2026-0002. See `vendor/lru/SECURITY-PATCHES.md` for the backport details. Prior assessments of unreachable triggers do not count as patching the dependency.

## Remaining review and verification

The full feature suite ran with `claude-code,codex,gemini-cli,agy`: 844 passed and 5 failed. The failures are two `session_start_does_not_create_memory_store` checks, two compact/session-end document-type checks, and the OpenClaw bootstrap recall check. I do not have a verified baseline run for these five failures, so they are unresolved and should not be labeled baseline failures. The separate server test run passed 41 tests. `git diff --check` passed. A whole-worktree `cargo fmt --check` remains noisy because of pre-existing formatting differences in unrelated files.

## Benchmark rerun

Release benchmark binaries were built from the working tree. Raw outputs are retained under `comparison/results/security-review-*-2026-10-11.json`.

- LongMemEval-S, 500 questions: any-hit Recall@1/3/5/10/20 = 82.0%/92.6%/95.8%/98.2%/98.8%; MRR 88.0%; NDCG@10 87.2%; mean measured search time 3.75 ms. Artifact: `security-review-longmemeval-2026-10-11.json`.
- Segmented HTTP search, 23,366 records, 3 repetitions, 100 measured requests per cell after 25 warmups: concurrency 1 median p50 1.94 ms and p99 2.28 ms; concurrency 10 median p50 8.60 ms and p99 15.02 ms, at median throughput about 1,149 requests/s. Artifact: `security-review-throughput-segmented-2026-10-11.json`. Single/global layouts were rejected by the current resident segmented-store requirement, so this is the supported segmented layout only.
- Write-only HTTP load on the same corpus, one short repetition per batch size: accepted-record rates were 224/s (batch 1), 1,498/s (batch 8), 4,446/s (batch 32), and 9,701/s (batch 128). These are staged-write acknowledgement rates. Published rates after flush were 105/s, 621/s, 1,570/s, and 2,987/s respectively. Every cell confirmed the last written record became searchable. This is a short burst measurement, not a sustained throughput claim. Artifact: `security-review-write-only-2026-10-11.json`.
- Mixed load, 23,366 initial records, 10 readers and 1 writer, 5 seconds: readers completed 17,602 searches at 3,519/s during the write phase (p50 1.97 ms, p99 13.25 ms); the writer accepted 7,240 records at 4.65 ms p50. Publication and visibility verification took a further 7.84 seconds, and the written record was searchable afterward. This single run is noisy and should not be compared directly with the paired read-only control. Artifact: `security-review-mixed-load-2026-10-11.json`.
- Corpus-scale build/search (saved output from the final run): 1,000 sessions built in 1.25 s (p50/p95 0.97/1.56 ms), 5,000 in 6.36 s (2.34/3.76 ms), 10,000 in 12.95 s (2.88/4.80 ms), and the full 19,829-session dataset in 27.82 s (2.52/4.17 ms). The dataset contains 19,829 unique sessions from 500 questions, so it did not reach the requested 25,000-session size. Recall-any@10 was 80% on the full dataset in this run. Output: `comparison/results/security-review-corpus-scale-2026-10-11.txt`.

The mixed-load harness previously omitted `--project-root`, causing server startup to use the repository as its root instead of the temporary benchmark corpus. `comparison/mixed_load.py` now passes the temporary root explicitly; the repaired run completed and verified post-write visibility.

`cargo install --path . --features agent-integrations --force` installed the working-tree `lint-ai` and `server` binaries (v0.3.0). The already-running backend on port 8080 was not restarted, so that process still uses its prior in-memory executable until restarted.
