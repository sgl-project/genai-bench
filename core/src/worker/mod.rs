//! Worker module for executing benchmark requests
//!
//! The Worker is the core execution unit in genai-bench, responsible for the
//! simple but critical loop: **sample -> execute -> report -> repeat**.
//!
//! Unlike traditional load testing tools that simulate "users" with complex
//! lifecycle hooks, think times, and session state, our Worker is intentionally
//! minimal. Each Worker is a stateless tokio task that:
//!
//! 1. Samples the next request from a Sampler
//! 2. Executes the request via a VendorClient
//! 3. Streams the response and computes metrics
//! 4. Sends metrics to the aggregator via channel
//! 5. Optionally applies rate limiting
//! 6. Repeats until stop condition is met
//!
//! # Example
//!
//! ```ignore
//! use genai_bench_core::worker::{Worker, WorkerBuilder, WorkerStats};
//! use genai_bench_core::traits::StopCondition;
//!
//! let worker = WorkerBuilder::new(0)
//!     .vendor(vendor)
//!     .sampler(sampler)
//!     .metrics_tx(tx)
//!     .semaphore(semaphore)
//!     .stop_condition(StopCondition::RequestCount(100))
//!     .concurrency(10)
//!     .build()?;
//!
//! let stats = worker.run(shutdown_rx).await?;
//! println!("Completed: {}", stats.completed);
//! ```

mod builder;
mod executor;
mod rate_limiter;
mod stats;

pub use builder::WorkerBuilder;
pub use executor::Worker;
pub use rate_limiter::RequestRateLimiter;
pub use stats::WorkerStats;

#[cfg(test)]
mod tests;
