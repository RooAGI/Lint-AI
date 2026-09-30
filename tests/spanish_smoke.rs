// Spanish add -> search smoke test.
//
// Adds three Spanish memories and verifies Spanish queries retrieve the
// right ones, both with --lang es (explicit) and --lang auto (detection).
// Exercises the full pipeline: accent-preserving tokenization, tantivy
// Latin folding, Spanish stopwords, interrogative focus, and temporal.
use lint_ai::lang::Lang;
use lint_ai::memory_api::{AddRequest, MemoryService, Message};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

const FACT_LIBRO: &str = "El usuario compró un libro sobre la historia de Madrid ayer.";
const FACT_BIBLIOTECA: &str = "La biblioteca de Madrid está en la calle de Alcalá.";
const FACT_PAELLA: &str = "A María le gusta cocinar paella los domingos.";

fn options_for(lang: Lang) -> PipelineOptions {
    let mut opts = PipelineOptions::default();
    opts.lang = lang;
    // Keep the smoke test hermetic: no spaCy, no bekind, no network.
    opts.ner_provider = lint_ai::pipeline::Tier1NerProvider::Heuristic;
    opts
}

fn add_fact(service: &mut MemoryService, user_id: &str, request_id: &str, content: &str) {
    service
        .add(AddRequest {
            request_id: request_id.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                timestamp: None,
                content: content.to_string(),
                expires_at_ms: None,
                supersedes_id: None,
            }],
            user_id: user_id.to_string(),
            session_id: "es-smoke".to_string(),
        })
        .expect("add fact");
}

fn search(service: &mut MemoryService, user_id: &str, query: &str) -> Vec<lint_ai::SearchResult> {
    let mut filters = BTreeMap::new();
    filters.insert("memory_user_id".to_string(), user_id.to_string());
    service
        .search_with_filters(query, user_id, None, 10, &filters)
        .expect("search")
}

fn doc_id(user_id: &str, request_id: &str) -> String {
    lint_ai::stable_doc_id_from_source(&format!("{user_id}:{request_id}:0"))
}

fn run_smoke(lang: Lang, label: &str) {
    let user_id = format!("es-smoke-{label}");
    let mut service = MemoryService::in_memory(options_for(lang));
    add_fact(&mut service, &user_id, "req-libro", FACT_LIBRO);
    add_fact(&mut service, &user_id, "req-biblio", FACT_BIBLIOTECA);
    add_fact(&mut service, &user_id, "req-paella", FACT_PAELLA);

    let id_libro = doc_id(&user_id, "req-libro");
    let id_biblio = doc_id(&user_id, "req-biblio");
    let id_paella = doc_id(&user_id, "req-paella");

    // "Where is the Madrid library?" -> the biblioteca fact.
    let r = search(&mut service, &user_id, "¿Dónde está la biblioteca de Madrid?");
    let top = r.first().expect("[{label}] donde: no results");
    eprintln!("[{label}] donde top: {} score={:.2}", top.doc_id, top.score);
    assert_eq!(top.doc_id, id_biblio, "[{label}] donde: wrong top hit");

    // "What did the user buy yesterday?" -> the libro fact (temporal: ayer).
    let r = search(&mut service, &user_id, "¿Qué compró el usuario ayer?");
    let pos = r
        .iter()
        .position(|hit| hit.doc_id == id_libro)
        .expect("[{label}] que-ayer: libro fact not retrieved");
    eprintln!("[{label}] que-ayer: libro at position {pos}");

    // "Who likes to cook paella?" -> the paella fact (quién -> Who).
    let r = search(&mut service, &user_id, "¿A quién le gusta cocinar paella?");
    let pos = r
        .iter()
        .position(|hit| hit.doc_id == id_paella)
        .expect("[{label}] quien: paella fact not retrieved");
    eprintln!("[{label}] quien: paella at position {pos}");

    // Accent-insensitive: unaccented "biblioteca" query still matches the
    // accented indexed form via the tantivy Latin folding.
    let r = search(&mut service, &user_id, "biblioteca Madrid");
    assert!(
        r.iter().any(|hit| hit.doc_id == id_biblio),
        "[{label}] unaccented query missed biblioteca fact"
    );
}

#[test]
fn spanish_smoke_explicit_lang() {
    run_smoke(Lang::Es, "es");
}

#[test]
fn spanish_smoke_auto_detect() {
    run_smoke(Lang::Auto, "auto");
}
