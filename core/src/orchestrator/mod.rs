//! Orchestrator for experiment lifecycle management
//!
//! The Orchestrator coordinates the complete benchmark experiment:
//! - Spawning and managing worker tasks
//! - Handling concurrency via semaphores
//! - Managing graceful shutdown via broadcast channels
//! - Collecting results from all workers
//!
//! # Example
//!
//! ```ignore
//! use genai_bench_core::{OrchestratorBuilder, StopCondition};
//!
//! let (orchestrator, metrics_rx) = OrchestratorBuilder::new()
//!     .concurrency(10)
//!     .stop_condition(StopCondition::RequestCount(1000))
//!     .vendor(vendor)
//!     .sampler(sampler)
//!     .build()?;
//!
//! let stats = orchestrator.run_with_signal_handling().await?;
//! ```

mod aggregator;
mod builder;
mod executor;

pub use aggregator::{aggregate_worker_stats, AggregatedStats};
pub use builder::OrchestratorBuilder;
pub use executor::Orchestrator;

#[cfg(test)]
mod tests;
