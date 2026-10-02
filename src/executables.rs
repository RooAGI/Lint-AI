//! Private implementations of bundled command-line programs.

#[path = "bin/corpus_scale_benchmark.rs"]
mod corpus_scale_benchmark;
#[path = "bin/haystack_indexstore_benchmark.rs"]
mod haystack_indexstore_benchmark;
#[path = "bin/haystack_scoped_benchmark.rs"]
mod haystack_scoped_benchmark;
#[path = "bin/jevbench_case_classifier.rs"]
mod jevbench_case_classifier;
#[path = "bin/locomo_benchmark.rs"]
mod locomo_benchmark;
#[cfg(feature = "agent-integrations")]
#[path = "bin/locomo_benchmark_search_server.rs"]
mod locomo_benchmark_search_server;
#[path = "bin/project_memory_recency_benchmark.rs"]
mod project_memory_recency_benchmark;
#[path = "bin/recency_benchmark.rs"]
mod recency_benchmark;
#[path = "bin/refresh_scaling_benchmark.rs"]
mod refresh_scaling_benchmark;
#[cfg(feature = "experimental")]
#[path = "bin/segment_scoped_benchmark.rs"]
mod segment_scoped_benchmark;
#[path = "bin/server.rs"]
mod server;
#[path = "bin/session_ab_benchmark.rs"]
mod session_ab_benchmark;
#[path = "bin/validate_corpus.rs"]
mod validate_corpus;

pub(crate) fn run(name: &str) -> anyhow::Result<()> {
    match name {
        "lint-ai" => crate::cli::run(),
        "corpus_scale_benchmark" => corpus_scale_benchmark::main(),
        "haystack_indexstore_benchmark" => haystack_indexstore_benchmark::main(),
        "haystack_scoped_benchmark" => haystack_scoped_benchmark::main(),
        "jevbench_case_classifier" => jevbench_case_classifier::main(),
        "locomo_benchmark" => locomo_benchmark::main(),
        #[cfg(feature = "agent-integrations")]
        "locomo_benchmark_search_server" => locomo_benchmark_search_server::main(),
        "project_memory_recency_benchmark" => project_memory_recency_benchmark::main(),
        "recency_benchmark" => recency_benchmark::main(),
        "refresh_scaling_benchmark" => refresh_scaling_benchmark::main(),
        #[cfg(feature = "experimental")]
        "segment_scoped_benchmark" => segment_scoped_benchmark::main(),
        "server" => server::main(),
        "session_ab_benchmark" => session_ab_benchmark::main(),
        "validate_corpus" => {
            validate_corpus::main();
            Ok(())
        }
        _ => anyhow::bail!("executable is unavailable in this build: {name}"),
    }
}
