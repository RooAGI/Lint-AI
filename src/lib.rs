//! Lint AI — semantic linting for Markdown documentation.
//!
//! This crate provides the core pipeline for building a concept inventory from
//! a Markdown corpus, matching mentions, and reporting missing cross-references
//! and orphan/unreachable pages.
//!
//! Basic usage:
//! ```no_run
//! use lint_ai::graph::Graph;
//! use lint_ai::report::Report;
//! use lint_ai::rules::{cross_refs::check_cross_refs, orphan_pages::check_orphans};
//! use lint_ai::config::Config;
//!
//! let graph = Graph::build("docs", 5_000_000, 50_000, 20, 100_000_000).unwrap();
//! let mut report = Report::new();
//! let cfg = Config::default();
//! check_orphans(&graph, &mut report);
//! check_cross_refs(&graph, &mut report, &cfg);
//! ```

pub mod cli;
pub mod config;
pub mod engine;
pub mod filters;
pub mod graph;
pub mod index;
pub mod report;
pub mod rules;
pub mod tier1;
