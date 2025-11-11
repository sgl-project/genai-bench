//! GenAI Bench CLI
//!
//! Command-line interface for running LLM benchmarks.

use anyhow::Result;
use clap::Parser;
use genai_bench_rs::cli::Cli;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Run the benchmark
    cli.run().await?;

    Ok(())
}
