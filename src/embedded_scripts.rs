//! Runtime Python scripts embedded in the binary.
//!
//! The spaCy NER script and the relation extractor are part of the
//! product, not of the source checkout. Resolving them through
//! `env!("CARGO_MANIFEST_DIR")` baked the *build machine's* checkout
//! path into every distributed binary (`/home/runner/work/...` in CI
//! builds), so on any end-user machine the file was missing and the
//! spaCy ranker silently degraded to the heuristic ranker.
//!
//! Instead, both scripts are compiled into the binary with
//! `include_str!` and materialized once per version into a scratch
//! directory under the OS temp dir. The subprocess interfaces
//! (`python3 -I <script>`, the `--serve` daemon) keep taking a real
//! file path, unchanged.

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

const SPACY_NER_PY: &str = include_str!("../scripts/spacy_ner.py");
const SPACY_RELATIONS_PY: &str = include_str!("../scripts/spacy_relations.py");

/// Directory holding the materialized scripts for this crate version.
/// Materialization is best-effort: if the temp dir is not writable the
/// returned paths simply do not exist, and callers fail open exactly as
/// they did when the script file was missing.
fn scripts_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!(
            "membrick-embedded-scripts-{}",
            env!("CARGO_PKG_VERSION")
        ));
        let _ = fs::create_dir_all(&dir);
        write_if_changed(&dir.join("spacy_ner.py"), SPACY_NER_PY);
        write_if_changed(&dir.join("spacy_relations.py"), SPACY_RELATIONS_PY);
        dir
    })
}

fn write_if_changed(path: &std::path::Path, contents: &str) {
    if fs::read(path).is_ok_and(|bytes| bytes == contents.as_bytes()) {
        return;
    }
    // Write to a sibling temp file, then rename, so a concurrent process
    // never observes a half-written script.
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    if fs::write(&tmp, contents).is_ok() {
        if fs::rename(&tmp, path).is_err() {
            let _ = fs::remove_file(path);
            let _ = fs::rename(&tmp, path);
        }
    }
}

/// Path of the materialized spaCy NER script (`scripts/spacy_ner.py`).
pub(crate) fn spacy_ner_script_path() -> PathBuf {
    scripts_dir().join("spacy_ner.py")
}

/// Path of the materialized relation extractor (`scripts/spacy_relations.py`).
/// Lives in the same directory as the NER script, mirroring the source
/// layout the scripts were written against.
pub(crate) fn spacy_relations_script_path() -> PathBuf {
    scripts_dir().join("spacy_relations.py")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materialized_ner_script_matches_embedded_source() {
        let path = spacy_ner_script_path();
        let on_disk = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("NER script not materialized at {path:?}: {e}"));
        assert_eq!(on_disk, SPACY_NER_PY);
        assert!(on_disk.contains("ALLOWED_MODELS"));
    }

    #[test]
    fn materialized_relations_script_matches_embedded_source() {
        let path = spacy_relations_script_path();
        let on_disk = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("relations script not materialized at {path:?}: {e}"));
        assert_eq!(on_disk, SPACY_RELATIONS_PY);
    }

    #[test]
    fn both_scripts_share_one_directory() {
        assert_eq!(
            spacy_ner_script_path().parent(),
            spacy_relations_script_path().parent()
        );
    }

    #[test]
    fn default_spacy_script_path_is_the_materialized_one() {
        assert_eq!(
            crate::tier1::default_spacy_script_path(),
            spacy_ner_script_path()
        );
        assert!(crate::tier1::default_spacy_script_path().is_file());
    }
}
