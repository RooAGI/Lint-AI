//! Guard: production memory access goes through [`lint_ai::memory_api::MemoryService`].
//!
//! Direct `IndexStore` use is allowed only in:
//! - `src/pipeline/*` — the core implementation the service owns,
//! - `src/memory_api.rs` — the service itself,
//! - `#[cfg(test)]` modules,
//! - `src/bin/*benchmark*.rs` — benchmarks,
//! - `inspect_index_store` in `src/engine/run.rs` — the documented low-level
//!   CLI diagnostic (see its doc comment).
//!
//! Everything else (integrations, hooks, recall, server) must go through the
//! service's constructors and methods.

use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Line ranges (inclusive, 0-based) covered by `#[cfg(test)] mod ...` blocks.
fn test_regions(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "#[cfg(test)]" {
            let mut j = i + 1;
            while j < lines.len() && !lines[j].contains("mod ") {
                j += 1;
            }
            if j < lines.len() {
                let mut depth: i32 = 0;
                let mut started = false;
                let mut k = j;
                while k < lines.len() {
                    for ch in lines[k].chars() {
                        if ch == '{' {
                            depth += 1;
                            started = true;
                        } else if ch == '}' {
                            depth -= 1;
                        }
                    }
                    if started && depth == 0 {
                        break;
                    }
                    k += 1;
                }
                regions.push((j, k.min(lines.len().saturating_sub(1))));
                i = k + 1;
                continue;
            }
        }
        i += 1;
    }
    regions
}

/// Name of the `fn` whose body contains `line_idx`, via brace tracking.
fn enclosing_fn(lines: &[&str], line_idx: usize) -> Option<String> {
    let mut depth: i32 = 0;
    let mut current: Option<(String, i32)> = None;
    for (idx, line) in lines.iter().enumerate() {
        if idx > line_idx {
            break;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub(crate) fn ")
        {
            let name = trimmed
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub ")
                .trim_start_matches("fn ")
                .split(['(', '<'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            // Only treat it as a body start if the signature opens a brace
            // on this line or a later line before the next fn.
            current = Some((name, depth));
        }
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if let Some((_, fn_depth)) = &current {
                    if depth < *fn_depth {
                        current = None;
                    }
                }
            }
        }
    }
    current.map(|(name, _)| name)
}

fn in_test_region(regions: &[(usize, usize)], idx: usize) -> bool {
    regions.iter().any(|(s, e)| idx >= *s && idx <= *e)
}

/// True when `IndexStore` appears as its own word (not `IndexStoreInspection`).
fn contains_index_store_word(code: &str) -> bool {
    let mut rest = code;
    while let Some(pos) = rest.find("IndexStore") {
        let after = rest[pos + "IndexStore".len()..].chars().next();
        let is_word_char = after
            .map(|c| c.is_alphanumeric() || c == '_')
            .unwrap_or(false);
        if !is_word_char {
            return true;
        }
        rest = &rest[pos + "IndexStore".len()..];
    }
    false
}

fn is_path_allowed(relative: &str) -> bool {
    relative.starts_with("pipeline/")
        || relative == "memory_api.rs"
        || (relative.starts_with("bin/") && relative.contains("benchmark"))
}

fn check_file(path: &Path, relative: &str) -> Vec<String> {
    let content = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let regions = test_regions(&lines);
    let mut violations = Vec::new();
    let mut in_block_comment = false;
    let mut in_lib_use_stmt = false;
    let mut skip_next_item = false;
    for (idx, line) in lines.iter().enumerate() {
        if in_test_region(&regions, idx) {
            continue;
        }
        let trimmed_start = line.trim_start();
        // `#[cfg(test)]` directly on an item (import, fn, ...) exempts it.
        if trimmed_start == "#[cfg(test)]" {
            skip_next_item = true;
            continue;
        }
        if skip_next_item {
            if !trimmed_start.is_empty() {
                skip_next_item = false;
                continue;
            }
            continue;
        }
        // Strip block comments (approximate) and line comments.
        let mut code = String::new();
        let mut chars = line.chars().peekable();
        let mut block = in_block_comment;
        while let Some(ch) = chars.next() {
            if block {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    block = false;
                }
                continue;
            }
            if ch == '/' && chars.peek() == Some(&'*') {
                chars.next();
                block = true;
                continue;
            }
            if ch == '/' && chars.peek() == Some(&'/') {
                break;
            }
            code.push(ch);
        }
        in_block_comment = block;
        // Skip string literals mentioning IndexStore (test data, doc text).
        // Approximate: drop double-quoted spans.
        let mut stripped = String::new();
        let mut in_string = false;
        let mut str_chars = code.chars().peekable();
        while let Some(ch) = str_chars.next() {
            if ch == '"' && !in_string {
                in_string = true;
                continue;
            }
            if ch == '"' && in_string {
                in_string = false;
                continue;
            }
            if ch == '\\' && in_string {
                str_chars.next();
                continue;
            }
            if !in_string {
                stripped.push(ch);
            }
        }
        // Public API re-export tracking (curated in PR #66); the type stays
        // public for downstream crates, but in-crate production code goes
        // through the service. Runs before the word check so multi-line
        // `pub use` continuations are tracked on every line.
        if relative == "lib.rs" {
            let t = stripped.trim_start();
            if t.starts_with("pub use") || t.starts_with("use ") {
                in_lib_use_stmt = !stripped.contains(';');
                continue;
            }
            if in_lib_use_stmt {
                in_lib_use_stmt = !stripped.contains(';');
                continue;
            }
        }
        if !contains_index_store_word(&stripped) {
            continue;
        }
        // Documented low-level diagnostic exception.
        if relative == "engine/run.rs"
            && (stripped.trim_start().starts_with("use ")
                || enclosing_fn(&lines, idx).as_deref() == Some("inspect_index_store"))
        {
            continue;
        }
        violations.push(format!("{}:{}: {}", relative, idx + 1, line.trim()));
    }
    violations
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}

#[test]
fn production_code_does_not_bypass_memory_service() {
    let src = manifest_dir().join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files);
    files.sort();
    let mut violations = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(&src)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if is_path_allowed(&relative) {
            continue;
        }
        violations.extend(check_file(&path, &relative));
    }
    assert!(
        violations.is_empty(),
        "direct IndexStore use outside MemoryService (allowed: src/pipeline/*, \
         src/memory_api.rs, #[cfg(test)] modules, benchmarks, documented diagnostics):\n{}",
        violations.join("\n")
    );
}
