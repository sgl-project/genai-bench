//! Metrics aggregation and percentile calculation

use crate::request::{RequestId, Task};
use crate::response::ResponseStatus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Aggregated metrics for an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// Experiment metadata
    pub metadata: ExperimentMetadata,

    /// Summary statistics
    pub summary: MetricsSummary,

    /// Per-scenario breakdown
    pub scenarios: HashMap<String, ScenarioMetrics>,

    /// Time-series data (for plotting)
    pub timeseries: Vec<TimeseriesPoint>,

    /// Individual request records
    pub requests: Vec<RequestRecord>,
}

impl ExperimentMetrics {
    /// Create new metrics with metadata
    pub fn new(metadata: ExperimentMetadata) -> Self {
        Self {
            metadata,
            summary: MetricsSummary::default(),
            scenarios: HashMap::new(),
            timeseries: Vec::new(),
            requests: Vec::new(),
        }
    }
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    /// Unique experiment identifier
    pub experiment_id: String,
    /// Vendor name (e.g., "openai", "azure")
    pub vendor: String,
    /// Model name
    pub model: String,
    /// Task type
    pub task: Task,
    /// When the experiment started
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// When the experiment ended
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Experiment configuration
    pub config: ExperimentConfig,
}

/// Experiment configuration (subset of full config relevant for metrics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Concurrency level
    pub concurrency: usize,
    /// Number of requests to run
    pub num_requests: usize,
    /// Duration limit in seconds
    pub duration_secs: Option<u64>,
    /// Additional configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            num_requests: 100,
            duration_secs: None,
            extra: HashMap::new(),
        }
    }
}

/// Summary statistics for an experiment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSummary {
    /// Total requests made
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,

    /// Total input tokens
    pub total_input_tokens: usize,
    /// Total output tokens
    pub total_output_tokens: usize,

    /// Requests per second
    pub requests_per_second: f64,
    /// Input tokens per second
    pub input_tokens_per_second: f64,
    /// Output tokens per second
    pub output_tokens_per_second: f64,

    /// Time to first token percentiles (milliseconds)
    pub ttft: LatencyPercentiles,
    /// Time per output token percentiles (milliseconds)
    pub tpot: LatencyPercentiles,
    /// End-to-end latency percentiles (milliseconds)
    pub e2e: LatencyPercentiles,

    /// Total duration in seconds
    pub total_duration_secs: f64,
}

impl MetricsSummary {
    /// Calculate summary from request records
    pub fn from_records(records: &[RequestRecord], duration: Duration) -> Self {
        let total_requests = records.len();
        let successful_requests = records.iter().filter(|r| r.status.is_success()).count();
        let failed_requests = total_requests - successful_requests;
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };

        let total_input_tokens: usize = records.iter().map(|r| r.input_tokens).sum();
        let total_output_tokens: usize = records.iter().map(|r| r.output_tokens).sum();

        let duration_secs = duration.as_secs_f64();
        let requests_per_second = if duration_secs > 0.0 {
            total_requests as f64 / duration_secs
        } else {
            0.0
        };
        let input_tokens_per_second = if duration_secs > 0.0 {
            total_input_tokens as f64 / duration_secs
        } else {
            0.0
        };
        let output_tokens_per_second = if duration_secs > 0.0 {
            total_output_tokens as f64 / duration_secs
        } else {
            0.0
        };

        // Calculate latency percentiles
        let ttft_values: Vec<f64> = records.iter().filter_map(|r| r.ttft_ms).collect();
        let tpot_values: Vec<f64> = records.iter().filter_map(|r| r.tpot_ms).collect();
        let e2e_values: Vec<f64> = records.iter().map(|r| r.e2e_ms).collect();

        Self {
            total_requests,
            successful_requests,
            failed_requests,
            error_rate,
            total_input_tokens,
            total_output_tokens,
            requests_per_second,
            input_tokens_per_second,
            output_tokens_per_second,
            ttft: LatencyPercentiles::from_values(&ttft_values),
            tpot: LatencyPercentiles::from_values(&tpot_values),
            e2e: LatencyPercentiles::from_values(&e2e_values),
            total_duration_secs: duration_secs,
        }
    }
}

/// Latency percentiles (all values in milliseconds)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct LatencyPercentiles {
    /// Minimum value
    pub min: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 75th percentile
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub stddev: f64,
}

impl LatencyPercentiles {
    /// Calculate percentiles from a slice of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let mean = sorted.iter().sum::<f64>() / len as f64;

        let variance = if len > 1 {
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (len - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();

        Self {
            min: sorted[0],
            p50: percentile(&sorted, 0.50),
            p75: percentile(&sorted, 0.75),
            p90: percentile(&sorted, 0.90),
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            max: sorted[len - 1],
            mean,
            stddev,
        }
    }
}

/// Calculate percentile from sorted values using linear interpolation
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// In-memory histogram for efficient percentile calculation
/// Uses HdrHistogram for memory-efficient storage of large datasets
pub struct LatencyHistogram {
    histogram: hdrhistogram::Histogram<u64>,
}

impl LatencyHistogram {
    /// Create a new histogram
    /// Configured for microsecond precision with max 1 hour latency
    pub fn new() -> Self {
        // Histogram with microsecond precision, max 1 hour (3,600,000,000 microseconds)
        let histogram = hdrhistogram::Histogram::new_with_bounds(1, 3_600_000_000, 3)
            .expect("Failed to create histogram");
        Self { histogram }
    }

    /// Record a duration
    pub fn record(&mut self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        let _ = self.histogram.record(micros);
    }

    /// Record a value in milliseconds
    pub fn record_ms(&mut self, ms: f64) {
        let micros = (ms * 1000.0) as u64;
        let _ = self.histogram.record(micros);
    }

    /// Get the number of recorded values
    pub fn len(&self) -> u64 {
        self.histogram.len()
    }

    /// Check if the histogram is empty
    pub fn is_empty(&self) -> bool {
        self.histogram.is_empty()
    }

    /// Calculate percentiles from the histogram
    pub fn percentiles(&self) -> LatencyPercentiles {
        if self.histogram.is_empty() {
            return LatencyPercentiles::default();
        }

        LatencyPercentiles {
            min: self.histogram.min() as f64 / 1000.0,
            p50: self.histogram.value_at_quantile(0.50) as f64 / 1000.0,
            p75: self.histogram.value_at_quantile(0.75) as f64 / 1000.0,
            p90: self.histogram.value_at_quantile(0.90) as f64 / 1000.0,
            p95: self.histogram.value_at_quantile(0.95) as f64 / 1000.0,
            p99: self.histogram.value_at_quantile(0.99) as f64 / 1000.0,
            max: self.histogram.max() as f64 / 1000.0,
            mean: self.histogram.mean() / 1000.0,
            stddev: self.histogram.stdev() / 1000.0,
        }
    }

    /// Reset the histogram
    pub fn reset(&mut self) {
        self.histogram.reset();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-scenario metrics breakdown
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScenarioMetrics {
    /// Scenario name
    pub scenario_name: String,
    /// Number of requests
    pub request_count: usize,
    /// Number of successful requests
    pub success_count: usize,
    /// Number of failed requests
    pub error_count: usize,
    /// Total input tokens
    pub input_tokens: usize,
    /// Total output tokens
    pub output_tokens: usize,
    /// Time to first token percentiles
    pub ttft: LatencyPercentiles,
    /// Time per output token percentiles
    pub tpot: LatencyPercentiles,
    /// End-to-end latency percentiles
    pub e2e: LatencyPercentiles,
}

/// Time-series data point for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesPoint {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Cumulative requests completed
    pub requests_completed: usize,
    /// Requests per second at this point
    pub requests_per_second: f64,
    /// Tokens per second at this point
    pub tokens_per_second: f64,
    /// Average TTFT at this point (ms)
    pub avg_ttft_ms: f64,
    /// Average TPOT at this point (ms)
    pub avg_tpot_ms: f64,
    /// Cumulative error count
    pub error_count: usize,
}

/// Individual request record for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    /// Request ID
    pub request_id: RequestId,
    /// Scenario name
    pub scenario: String,
    /// Input tokens
    pub input_tokens: usize,
    /// Output tokens
    pub output_tokens: usize,
    /// Time to first token (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Time per output token (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    /// End-to-end latency (ms)
    pub e2e_ms: f64,
    /// Response status
    pub status: ResponseStatus,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 1.0);
        assert_eq!(percentiles.max, 10.0);
        assert!((percentiles.p50 - 5.5).abs() < 0.01);
        assert!((percentiles.mean - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile_single_value() {
        let values = vec![42.0];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 42.0);
        assert_eq!(percentiles.max, 42.0);
        assert_eq!(percentiles.p50, 42.0);
        assert_eq!(percentiles.mean, 42.0);
        assert_eq!(percentiles.stddev, 0.0);
    }

    #[test]
    fn test_percentile_empty() {
        let values: Vec<f64> = vec![];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 0.0);
        assert_eq!(percentiles.max, 0.0);
        assert_eq!(percentiles.mean, 0.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut histogram = LatencyHistogram::new();

        for i in 1..=100 {
            histogram.record_ms(i as f64);
        }

        let percentiles = histogram.percentiles();
        assert!((percentiles.min - 1.0).abs() < 0.1);
        assert!((percentiles.max - 100.0).abs() < 0.1);
        assert!((percentiles.p50 - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_empty_histogram() {
        let histogram = LatencyHistogram::new();
        let percentiles = histogram.percentiles();

        assert_eq!(percentiles.min, 0.0);
        assert_eq!(percentiles.max, 0.0);
        assert_eq!(percentiles.mean, 0.0);
    }

    #[test]
    fn test_histogram_record_duration() {
        let mut histogram = LatencyHistogram::new();
        histogram.record(Duration::from_millis(100));
        histogram.record(Duration::from_millis(200));

        assert_eq!(histogram.len(), 2);
        assert!(!histogram.is_empty());
    }

    #[test]
    fn test_metrics_summary_from_records() {
        let records = vec![
            RequestRecord {
                request_id: 1u64.into(),
                scenario: "test".to_string(),
                input_tokens: 100,
                output_tokens: 50,
                ttft_ms: Some(100.0),
                tpot_ms: Some(10.0),
                e2e_ms: 500.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            },
            RequestRecord {
                request_id: 2u64.into(),
                scenario: "test".to_string(),
                input_tokens: 200,
                output_tokens: 100,
                ttft_ms: Some(150.0),
                tpot_ms: Some(15.0),
                e2e_ms: 750.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            },
        ];

        let summary = MetricsSummary::from_records(&records, Duration::from_secs(10));

        assert_eq!(summary.total_requests, 2);
        assert_eq!(summary.successful_requests, 2);
        assert_eq!(summary.failed_requests, 0);
        assert_eq!(summary.error_rate, 0.0);
        assert_eq!(summary.total_input_tokens, 300);
        assert_eq!(summary.total_output_tokens, 150);
        assert!((summary.requests_per_second - 0.2).abs() < 0.01);
    }
}
