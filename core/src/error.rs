//! Error types for genai-bench-core

use thiserror::Error;

/// Core error type
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// Worker error
    #[error("worker error: {0}")]
    Worker(String),

    /// Metrics error
    #[error("metrics error: {0}")]
    Metrics(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
