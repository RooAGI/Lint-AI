//! False-positive (over-suppression) audit for semantic supersession.
//!
//! Run with: cargo run --example false_positive_audit
//!
//! Each case writes documents through the public MemoryService API and checks
//! that true facts are not wrongly retired. Cases expecting supersession are
//! true-positive controls that keep the harness honest.

use lint_ai::semantic_audit;

fn main() {
    let results = semantic_audit::run_all();
    let mut passed = 0;
    println!("{:<38} {:<7} {}", "case", "result", "detail");
    println!("{}", "-".repeat(90));
    for result in &results {
        if result.passed {
            passed += 1;
            println!("{:<38} {:<7}", result.name, "PASS");
        } else {
            println!("{:<38} {:<7}", result.name, "FAIL");
            for failure in &result.failures {
                println!("  - {failure}");
            }
        }
    }
    println!("{}", "-".repeat(90));
    println!("{passed}/{} cases passed", results.len());
    if passed != results.len() {
        std::process::exit(1);
    }
}
