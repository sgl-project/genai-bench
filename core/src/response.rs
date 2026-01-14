//! Response types and timing structures

// Re-export ErrorKind from error module for backward compatibility
pub use crate::error::ErrorKind;

use crate::request::RequestId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Response from a benchmark request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    /// Corresponding request ID
    pub request_id: RequestId,

    /// Response status
    pub status: ResponseStatus,

    /// Generated content (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Token counts
    pub tokens: TokenCounts,

    /// Timing information
    pub timing: ResponseTiming,

    /// Raw vendor response (for debugging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<serde_json::Value>,
}

impl BenchmarkResponse {
    /// Check if the response was successful
    pub fn is_success(&self) -> bool {
        self.status == ResponseStatus::Success
    }

    /// Check if the response was an error
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }

    /// Get the error kind if this is an error response
    pub fn error_kind(&self) -> Option<ErrorKind> {
        match self.status {
            ResponseStatus::Error(kind) => Some(kind),
            ResponseStatus::Success => None,
        }
    }
}

/// Response status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseStatus {
    /// Request completed successfully
    Success,
    /// Request failed with error
    Error(ErrorKind),
}

impl ResponseStatus {
    /// Check if this status indicates success
    pub fn is_success(&self) -> bool {
        matches!(self, ResponseStatus::Success)
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            ResponseStatus::Success => false,
            ResponseStatus::Error(kind) => kind.is_retryable(),
        }
    }
}

/// Token counts for a request/response
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCounts {
    /// Input tokens (from tokenizer or vendor response)
    pub input: usize,

    /// Output tokens generated
    pub output: usize,

    /// Total tokens (input + output)
    pub total: usize,
}

impl TokenCounts {
    /// Create new token counts
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            total: input + output,
        }
    }

    /// Create token counts for input only (embeddings, etc.)
    pub fn input_only(input: usize) -> Self {
        Self {
            input,
            output: 0,
            total: input,
        }
    }
}

impl std::ops::Add for TokenCounts {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input: self.input + rhs.input,
            output: self.output + rhs.output,
            total: self.total + rhs.total,
        }
    }
}

impl std::ops::AddAssign for TokenCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.input += rhs.input;
        self.output += rhs.output;
        self.total += rhs.total;
    }
}

/// Response timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTiming {
    /// When request was sent
    pub request_start: chrono::DateTime<chrono::Utc>,

    /// When first byte was received
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_byte: Option<chrono::DateTime<chrono::Utc>>,

    /// When first token was received
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token: Option<chrono::DateTime<chrono::Utc>>,

    /// When response completed
    pub response_end: chrono::DateTime<chrono::Utc>,

    /// Time to first byte (TTFB) in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttfb_ms: Option<f64>,

    /// Time to first token (TTFT) in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,

    /// End-to-end latency in milliseconds
    pub e2e_latency_ms: f64,

    /// Per-token timings in milliseconds (for TPOT calculation)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_timings_ms: Vec<f64>,
}

impl ResponseTiming {
    /// Create timing from timestamps
    pub fn from_timestamps(
        request_start: chrono::DateTime<chrono::Utc>,
        first_byte: Option<chrono::DateTime<chrono::Utc>>,
        first_token: Option<chrono::DateTime<chrono::Utc>>,
        response_end: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        let ttfb_ms =
            first_byte.map(|t| (t - request_start).num_microseconds().unwrap_or(0) as f64 / 1000.0);
        let ttft_ms = first_token
            .map(|t| (t - request_start).num_microseconds().unwrap_or(0) as f64 / 1000.0);
        let e2e_latency_ms = (response_end - request_start)
            .num_microseconds()
            .unwrap_or(0) as f64
            / 1000.0;

        Self {
            request_start,
            first_byte,
            first_token,
            response_end,
            ttfb_ms,
            ttft_ms,
            e2e_latency_ms,
            token_timings_ms: Vec::new(),
        }
    }

    /// Calculate time per output token (TPOT) in milliseconds
    pub fn tpot_ms(&self) -> Option<f64> {
        if self.token_timings_ms.is_empty() {
            return None;
        }

        // Skip the first token (TTFT) and calculate mean of remaining
        if self.token_timings_ms.len() <= 1 {
            return None;
        }

        let inter_token_times: Vec<f64> = self
            .token_timings_ms
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        if inter_token_times.is_empty() {
            return None;
        }

        Some(inter_token_times.iter().sum::<f64>() / inter_token_times.len() as f64)
    }

    /// Get end-to-end latency as Duration
    pub fn e2e_latency(&self) -> Duration {
        Duration::from_secs_f64(self.e2e_latency_ms / 1000.0)
    }

    /// Get TTFT as Duration
    pub fn ttft(&self) -> Option<Duration> {
        self.ttft_ms.map(|ms| Duration::from_secs_f64(ms / 1000.0))
    }

    /// Get TTFB as Duration
    pub fn ttfb(&self) -> Option<Duration> {
        self.ttfb_ms.map(|ms| Duration::from_secs_f64(ms / 1000.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counts_arithmetic() {
        let a = TokenCounts::new(100, 50);
        let b = TokenCounts::new(200, 100);
        let sum = a + b;

        assert_eq!(sum.input, 300);
        assert_eq!(sum.output, 150);
        assert_eq!(sum.total, 450);
    }

    #[test]
    fn test_token_counts_add_assign() {
        let mut a = TokenCounts::new(100, 50);
        let b = TokenCounts::new(200, 100);
        a += b;

        assert_eq!(a.input, 300);
        assert_eq!(a.output, 150);
        assert_eq!(a.total, 450);
    }

    #[test]
    fn test_response_timing_tpot() {
        let mut timing = ResponseTiming::from_timestamps(
            chrono::Utc::now(),
            None,
            Some(chrono::Utc::now()),
            chrono::Utc::now(),
        );
        timing.token_timings_ms = vec![100.0, 120.0, 140.0, 160.0, 180.0];

        let tpot = timing.tpot_ms().unwrap();
        assert!((tpot - 20.0).abs() < 0.01); // Mean of inter-token times
    }

    #[test]
    fn test_response_timing_empty_tpot() {
        let timing =
            ResponseTiming::from_timestamps(chrono::Utc::now(), None, None, chrono::Utc::now());

        assert!(timing.tpot_ms().is_none());
    }

    #[test]
    fn test_response_status_is_success() {
        assert!(ResponseStatus::Success.is_success());
        assert!(!ResponseStatus::Error(ErrorKind::Timeout).is_success());
    }
}
