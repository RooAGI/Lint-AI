# Membrick security backports

This directory vendors the MIT-licensed crates.io `lru` 0.12.5 source so Tantivy 0.25 keeps its existing API and retrieval behavior.

Backports from the upstream lru repository:

- [PR 238](https://github.com/jeromefroe/lru-rs/pull/238), RUSTSEC-2026-0253: detach a popped node before its key destructor can panic.
- [PR 224](https://github.com/jeromefroe/lru-rs/pull/224), RUSTSEC-2026-0002: mutable iteration creates shared key references, preserving the map's key pointers.

The package keeps version 0.12.5. Version-based audit tools may still flag it; do not describe this as upgrading to the patched registry release. Cargo's root `[patch.crates-io]` selects this source. Future consumer builds must retain that root patch or upgrade the dependency upstream.

Tests:

```sh
cargo test --manifest-path vendor/lru/Cargo.toml
cargo +nightly miri test --manifest-path vendor/lru/Cargo.toml iter_mut_preserves_shared_map_key_borrows
```

Both security regressions were observed failing on the original code, passing with the backport, failing after reverting only the fix, and passing after restoration. The mutable-iterator test requires Miri to detect the original pointer invalidation. The key-drop test inspects the surviving neighbours without traversing freed nodes; it intentionally forgets the cache so the failing assertion cannot run a corrupt destructor.
