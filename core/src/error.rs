//! Error types for genai-bench-core

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Error classification for benchmark failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorKind {
    /// Request timed out
    Timeout,
    /// Rate limited by vendor
    RateLimited,
    /// Invalid request format
    InvalidRequest,
    /// Authentication failure
    AuthenticationFailed,
    /// Server-side error (5xx)
    ServerError,
    /// Network/connection error
    ConnectionError,
    /// Streaming parse error
    StreamingError,
    /// Content filter triggered
    ContentFiltered,
    /// Model not found or unavailable
    ModelNotFound,
    /// Quota exceeded
    QuotaExceeded,
    /// Unknown error
    Unknown,
}

impl ErrorKind {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorKind::Timeout
                | ErrorKind::RateLimited
                | ErrorKind::ServerError
                | ErrorKind::ConnectionError
                | ErrorKind::StreamingError
        )
    }

    /// Get the recommended backoff for this error type
    pub fn recommended_backoff(&self) -> Option<Duration> {
        match self {
            ErrorKind::RateLimited => Some(Duration::from_secs(60)),
            ErrorKind::Timeout => Some(Duration::from_secs(5)),
            ErrorKind::ServerError => Some(Duration::from_secs(10)),
            ErrorKind::ConnectionError => Some(Duration::from_secs(2)),
            ErrorKind::StreamingError => Some(Duration::from_secs(1)),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Timeout => write!(f, "timeout"),
            ErrorKind::RateLimited => write!(f, "rate_limited"),
            ErrorKind::InvalidRequest => write!(f, "invalid_request"),
            ErrorKind::AuthenticationFailed => write!(f, "authentication_failed"),
            ErrorKind::ServerError => write!(f, "server_error"),
            ErrorKind::ConnectionError => write!(f, "connection_error"),
            ErrorKind::StreamingError => write!(f, "streaming_error"),
            ErrorKind::ContentFiltered => write!(f, "content_filtered"),
            ErrorKind::ModelNotFound => write!(f, "model_not_found"),
            ErrorKind::QuotaExceeded => write!(f, "quota_exceeded"),
            ErrorKind::Unknown => write!(f, "unknown"),
        }
    }
}

/// Main error type for benchmark operations
#[derive(Debug)]
pub struct BenchError {
    /// Error kind classification
    pub kind: BenchErrorKind,
    /// Human-readable error message
    pub message: String,
    /// Optional source error
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl BenchError {
    /// Create a new error
    pub fn new(kind: BenchErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Create an error with a source
    pub fn with_source<E>(kind: BenchErrorKind, message: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Configuration, message)
    }

    /// Create a vendor error
    pub fn vendor(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Vendor(kind), message)
    }

    /// Create an I/O error
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Io, message)
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Serialization, message)
    }

    /// Create a sampler error
    pub fn sampler(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Sampler, message)
    }

    /// Create an orchestration error
    pub fn orchestration(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Orchestration, message)
    }

    /// Create a metrics error
    pub fn metrics(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Metrics, message)
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(BenchErrorKind::Internal, message)
    }

    /// Create a shutdown error
    pub fn shutdown() -> Self {
        Self::new(
            BenchErrorKind::Shutdown,
            "Operation cancelled due to shutdown",
        )
    }

    /// Create a missing config error
    pub fn missing_config(field: &'static str) -> Self {
        Self::new(
            BenchErrorKind::MissingConfig(field),
            format!("Missing required configuration: {}", field),
        )
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match &self.kind {
            BenchErrorKind::Vendor(kind) => kind.is_retryable(),
            _ => false,
        }
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.message)
    }
}

impl std::error::Error for BenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Error kind classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchErrorKind {
    /// Configuration error
    Configuration,
    /// Missing required configuration field
    MissingConfig(&'static str),
    /// Vendor/API error
    Vendor(ErrorKind),
    /// I/O error
    Io,
    /// Serialization/deserialization error
    Serialization,
    /// Sampler error
    Sampler,
    /// Orchestration error
    Orchestration,
    /// Metrics calculation error
    Metrics,
    /// Shutdown signal received
    Shutdown,
    /// Internal error
    Internal,
}

impl fmt::Display for BenchErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchErrorKind::Configuration => write!(f, "CONFIG"),
            BenchErrorKind::MissingConfig(field) => write!(f, "MISSING:{}", field),
            BenchErrorKind::Vendor(kind) => write!(f, "VENDOR:{}", kind),
            BenchErrorKind::Io => write!(f, "IO"),
            BenchErrorKind::Serialization => write!(f, "SERDE"),
            BenchErrorKind::Sampler => write!(f, "SAMPLER"),
            BenchErrorKind::Orchestration => write!(f, "ORCHESTRATOR"),
            BenchErrorKind::Metrics => write!(f, "METRICS"),
            BenchErrorKind::Shutdown => write!(f, "SHUTDOWN"),
            BenchErrorKind::Internal => write!(f, "INTERNAL"),
        }
    }
}

/// Result type alias for benchmark operations
pub type BenchResult<T> = Result<T, BenchError>;

// Implement From for common error types

impl From<std::io::Error> for BenchError {
    fn from(err: std::io::Error) -> Self {
        Self::with_source(BenchErrorKind::Io, err.to_string(), err)
    }
}

impl From<serde_json::Error> for BenchError {
    fn from(err: serde_json::Error) -> Self {
        Self::with_source(BenchErrorKind::Serialization, err.to_string(), err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_retryable() {
        assert!(ErrorKind::RateLimited.is_retryable());
        assert!(ErrorKind::Timeout.is_retryable());
        assert!(ErrorKind::ServerError.is_retryable());
        assert!(ErrorKind::ConnectionError.is_retryable());
        assert!(ErrorKind::StreamingError.is_retryable());

        assert!(!ErrorKind::InvalidRequest.is_retryable());
        assert!(!ErrorKind::AuthenticationFailed.is_retryable());
        assert!(!ErrorKind::ContentFiltered.is_retryable());
        assert!(!ErrorKind::ModelNotFound.is_retryable());
        assert!(!ErrorKind::QuotaExceeded.is_retryable());
        assert!(!ErrorKind::Unknown.is_retryable());
    }

    #[test]
    fn test_error_kind_backoff() {
        assert!(ErrorKind::RateLimited.recommended_backoff().is_some());
        assert!(ErrorKind::Timeout.recommended_backoff().is_some());
        assert!(ErrorKind::InvalidRequest.recommended_backoff().is_none());
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ErrorKind::Timeout.to_string(), "timeout");
        assert_eq!(ErrorKind::RateLimited.to_string(), "rate_limited");
        assert_eq!(ErrorKind::InvalidRequest.to_string(), "invalid_request");
    }

    #[test]
    fn test_bench_error_display() {
        let err = BenchError::config("Invalid configuration");
        assert_eq!(err.to_string(), "[CONFIG] Invalid configuration");
    }

    #[test]
    fn test_bench_error_vendor() {
        let err = BenchError::vendor(ErrorKind::RateLimited, "Too many requests");
        assert_eq!(err.to_string(), "[VENDOR:rate_limited] Too many requests");
        assert!(err.is_retryable());
    }

    #[test]
    fn test_bench_error_not_retryable() {
        let err = BenchError::config("Bad config");
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_bench_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let bench_err: BenchError = io_err.into();
        assert_eq!(bench_err.kind, BenchErrorKind::Io);
    }

    #[test]
    fn test_bench_error_kind_display() {
        assert_eq!(BenchErrorKind::Configuration.to_string(), "CONFIG");
        assert_eq!(BenchErrorKind::Io.to_string(), "IO");
        assert_eq!(
            BenchErrorKind::Vendor(ErrorKind::Timeout).to_string(),
            "VENDOR:timeout"
        );
    }
}
