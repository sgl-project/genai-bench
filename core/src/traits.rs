//! Core traits for vendor clients and samplers
//!
//! These traits are defined in core to avoid circular dependencies.
//! Implementations live in their respective crates (vendors/, samplers/).

use crate::request::{BenchmarkRequest, Task};
use crate::response::BenchmarkResponse;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::time::Duration;

// ============================================================================
// Vendor Client Trait
// ============================================================================

/// Core vendor client trait for executing benchmark requests
///
/// Implementations handle vendor-specific API details (OpenAI, Azure, Bedrock, etc.)
/// while presenting a unified interface to the Worker.
#[async_trait]
pub trait VendorClient: Send + Sync {
    /// Vendor identifier (e.g., "openai", "azure", "bedrock")
    fn vendor_name(&self) -> &str;

    /// Model identifier (e.g., "gpt-4", "claude-3-opus")
    fn model_name(&self) -> &str;

    /// Supported task types for this vendor/model combination
    fn supported_tasks(&self) -> &[Task];

    /// Execute a request and return the complete response
    ///
    /// This method handles streaming internally and returns metrics.
    async fn execute(&self, request: &BenchmarkRequest) -> Result<BenchmarkResponse, VendorError>;

    /// Execute with detailed streaming (returns stream of chunks)
    ///
    /// Use this for fine-grained control over streaming responses.
    async fn execute_streaming(
        &self,
        request: &BenchmarkRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, VendorError>> + Send>>, VendorError>;

    /// Validate client configuration
    fn validate(&self) -> Result<(), VendorError>;
}

/// Reason why a response finished generating
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of generation (stop token)
    Stop,
    /// Maximum token limit reached
    Length,
    /// Content filtered by safety systems
    ContentFilter,
    /// Tool/function call requested
    ToolCalls,
}

/// Token usage statistics from vendor response
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    /// Number of tokens in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    /// Total tokens (prompt + completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<usize>,
}

/// A chunk from a streaming response
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Timestamp when this chunk was received
    pub timestamp: std::time::Instant,

    /// The text content of this chunk (None for metadata-only chunks)
    pub content: Option<String>,

    /// Token count for this chunk (0 if not available from API)
    pub token_count: usize,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Reason for completion (only in final chunk)
    pub finish_reason: Option<FinishReason>,

    /// Usage statistics (typically only in final chunk)
    pub usage: Option<Usage>,
}

/// Vendor-specific errors
#[derive(Debug, thiserror::Error)]
pub enum VendorError {
    /// HTTP/network error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Rate limited by vendor
    #[error("Rate limited: retry after {retry_after:?}")]
    RateLimited {
        /// Suggested retry delay
        retry_after: Option<Duration>,
    },

    /// Invalid request format
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Model not available
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Content filtered by safety systems
    #[error("Content filtered: {0}")]
    ContentFiltered(String),

    /// Server error (5xx)
    #[error("Server error: {status} - {message}")]
    ServerError {
        /// HTTP status code
        status: u16,
        /// Error message
        message: String,
    },

    /// Streaming parse error
    #[error("Streaming error: {0}")]
    StreamingError(String),

    /// Request timeout
    #[error("Request timed out after {0:?}")]
    Timeout(Duration),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

impl VendorError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            VendorError::Http(_)
                | VendorError::RateLimited { .. }
                | VendorError::ServerError { .. }
                | VendorError::Timeout(_)
                | VendorError::StreamingError(_)
        )
    }

    /// Get recommended backoff duration for retry
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            VendorError::RateLimited { retry_after } => {
                retry_after.or(Some(Duration::from_secs(60)))
            }
            VendorError::ServerError { .. } => Some(Duration::from_secs(10)),
            VendorError::Timeout(_) => Some(Duration::from_secs(5)),
            VendorError::StreamingError(_) => Some(Duration::from_secs(1)),
            VendorError::Http(_) => Some(Duration::from_secs(2)),
            _ => None,
        }
    }

    /// Convert to ErrorKind for metrics/error classification
    pub fn to_error_kind(&self) -> crate::ErrorKind {
        match self {
            VendorError::Http(_) => crate::ErrorKind::ConnectionError,
            VendorError::Auth(_) => crate::ErrorKind::AuthenticationFailed,
            VendorError::RateLimited { .. } => crate::ErrorKind::RateLimited,
            VendorError::InvalidRequest(_) => crate::ErrorKind::InvalidRequest,
            VendorError::ModelNotFound(_) => crate::ErrorKind::ModelNotFound,
            VendorError::ContentFiltered(_) => crate::ErrorKind::ContentFiltered,
            VendorError::ServerError { .. } => crate::ErrorKind::ServerError,
            VendorError::StreamingError(_) => crate::ErrorKind::StreamingError,
            VendorError::Timeout(_) => crate::ErrorKind::Timeout,
            VendorError::Config(_) => crate::ErrorKind::Unknown,
        }
    }
}

// ============================================================================
// Sampler Trait
// ============================================================================

/// Sampler generates benchmark requests
///
/// Implementations can generate requests from distributions (synthetic workloads)
/// or from datasets (realistic workloads).
pub trait Sampler: Send + Sync {
    /// Sampler name for identification
    fn name(&self) -> &str;

    /// Generate a single benchmark request
    ///
    /// Returns `SamplerError::Exhausted` if the dataset is depleted
    /// and no more requests can be generated.
    fn sample(&self) -> Result<BenchmarkRequest, SamplerError>;

    /// Get the scenario name for the current/last sample
    ///
    /// Used for grouping metrics by scenario (e.g., "input_512_output_128")
    fn scenario_name(&self) -> &str;

    /// Supported task types for this sampler
    fn supported_tasks(&self) -> &[Task];
}

/// Sampler-specific errors
#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    /// Dataset exhausted, no more samples available
    #[error("Dataset exhausted")]
    Exhausted,

    /// Invalid sampler configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Tokenization error during sample generation
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// IO error (e.g., reading dataset file)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ============================================================================
// Stop Condition
// ============================================================================

/// Experiment stop condition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum StopCondition {
    /// Run exactly N requests total (divided among workers)
    RequestCount(usize),

    /// Run for the specified duration
    Duration(Duration),

    /// Run indefinitely until explicitly stopped (Ctrl+C)
    Indefinite,
}

impl Default for StopCondition {
    fn default() -> Self {
        StopCondition::RequestCount(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_error_retryable() {
        assert!(VendorError::RateLimited { retry_after: None }.is_retryable());
        assert!(VendorError::Timeout(Duration::from_secs(30)).is_retryable());
        assert!(VendorError::ServerError {
            status: 500,
            message: "Internal error".into()
        }
        .is_retryable());

        assert!(!VendorError::Auth("bad token".into()).is_retryable());
        assert!(!VendorError::InvalidRequest("bad format".into()).is_retryable());
        assert!(!VendorError::ModelNotFound("gpt-5".into()).is_retryable());
    }

    #[test]
    fn test_vendor_error_retry_after() {
        let rate_limited = VendorError::RateLimited {
            retry_after: Some(Duration::from_secs(30)),
        };
        assert_eq!(rate_limited.retry_after(), Some(Duration::from_secs(30)));

        let rate_limited_no_hint = VendorError::RateLimited { retry_after: None };
        assert_eq!(
            rate_limited_no_hint.retry_after(),
            Some(Duration::from_secs(60))
        );

        let auth_error = VendorError::Auth("bad".into());
        assert_eq!(auth_error.retry_after(), None);
    }

    #[test]
    fn test_stop_condition_default() {
        let default = StopCondition::default();
        assert!(matches!(default, StopCondition::RequestCount(100)));
    }

    #[test]
    fn test_vendor_error_to_error_kind() {
        use crate::ErrorKind;

        assert_eq!(
            VendorError::Auth("bad".into()).to_error_kind(),
            ErrorKind::AuthenticationFailed
        );
        assert_eq!(
            VendorError::RateLimited { retry_after: None }.to_error_kind(),
            ErrorKind::RateLimited
        );
        assert_eq!(
            VendorError::InvalidRequest("bad".into()).to_error_kind(),
            ErrorKind::InvalidRequest
        );
        assert_eq!(
            VendorError::ModelNotFound("gpt-5".into()).to_error_kind(),
            ErrorKind::ModelNotFound
        );
        assert_eq!(
            VendorError::ContentFiltered("blocked".into()).to_error_kind(),
            ErrorKind::ContentFiltered
        );
        assert_eq!(
            VendorError::ServerError {
                status: 500,
                message: "error".into()
            }
            .to_error_kind(),
            ErrorKind::ServerError
        );
        assert_eq!(
            VendorError::StreamingError("parse fail".into()).to_error_kind(),
            ErrorKind::StreamingError
        );
        assert_eq!(
            VendorError::Timeout(Duration::from_secs(30)).to_error_kind(),
            ErrorKind::Timeout
        );
        assert_eq!(
            VendorError::Config("bad config".into()).to_error_kind(),
            ErrorKind::Unknown
        );
    }
}
