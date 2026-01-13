//! CLI argument parsing and command dispatch

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "genai-bench")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run a benchmark experiment
    Run {
        /// Path to configuration file
        #[arg(short, long)]
        config: String,
    },
    /// Generate a report from results
    Report {
        /// Path to results file
        #[arg(short, long)]
        input: String,
        /// Output format
        #[arg(short, long, default_value = "excel")]
        format: String,
    },
    /// Validate a configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long)]
        config: String,
    },
}
