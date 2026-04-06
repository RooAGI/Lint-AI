use anyhow::Result;

fn main() -> Result<()> {
    let args = lint_ai::cli::parse();
    lint_ai::engine::run(args)?;
    Ok(())
}
