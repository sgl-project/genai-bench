//! GenAI Bench - High-performance LLM benchmarking tool
//!
//! This library provides the core functionality for benchmarking various LLM providers
//! including OpenAI, Azure, GCP, AWS, and Anthropic.
//!
//! # Architecture
//!
//! - **Providers**: HTTP clients for different LLM APIs
//! - **Scenarios**: Request patterns (Deterministic, Normal, Uniform)
//! - **Metrics**: Performance measurement and collection
//! - **Sampling**: Request sampling from datasets
//! - **Runner**: Orchestrates benchmark execution
//!
//! # Example
//!
//! ```rust,no_run
//! use genai_bench_rs::providers::openai::OpenAIProvider;
//! use genai_bench_rs::metrics::RequestMetrics;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let provider = OpenAIProvider::new(
//!         "your-api-key".to_string(),
//!         "https://api.openai.com/v1".to_string(),
//!     );
//!
//!     // Run a single request and get metrics
//!     // let metrics = provider.chat(&request).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod cli;
pub mod metrics;
pub mod output;
pub mod providers;
pub mod runner;
pub mod sampling;
pub mod scenarios;
pub mod ui;
pub mod visualization;

// Re-export commonly used types
pub use metrics::{MetricsCollector, RequestMetrics};
pub use output::{CsvExporter, ExcelExporter, JsonExporter};
pub use providers::Provider;
pub use scenarios::{DeterministicScenario, Scenario};
pub use visualization::{HistogramPlotter, PercentilePlotter, ThroughputPlotter};
