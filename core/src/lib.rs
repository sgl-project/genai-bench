//! Core types, metrics collection, and orchestration for genai-bench
//!
//! This crate provides the foundational types and abstractions used throughout
//! the genai-bench tool, including:
//!
//! - Protocol data structures (requests, responses, events)
//! - Metrics collection and aggregation
//! - Experiment orchestration
//! - Worker implementation

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;

/// Core error types
pub mod prelude {
    pub use crate::error::{Error, Result};
}
