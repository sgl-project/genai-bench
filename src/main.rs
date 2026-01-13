//! genai-bench - High-performance GenAI benchmarking tool

use anyhow::Result;
use clap::Parser;

mod cli;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let _cli = cli::Cli::parse();

    tracing::info!("genai-bench starting...");

    // TODO: Implement command dispatch

    Ok(())
}
