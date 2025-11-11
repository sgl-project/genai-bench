//! Metrics collection and aggregation

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Metrics for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Time to first token in milliseconds
    pub ttft_ms: u64,
    /// Total request time in milliseconds
    pub total_time_ms: u64,
    /// Number of prompt tokens
    pub prompt_tokens: u32,
    /// Number of completion tokens
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
    /// Whether the request succeeded
    pub success: bool,
    /// HTTP status code
    pub status_code: u16,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Aggregated metrics across multiple requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub avg_ttft_ms: f64,
    pub p50_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub avg_total_time_ms: f64,
    pub p50_total_time_ms: f64,
    pub p95_total_time_ms: f64,
    pub p99_total_time_ms: f64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_tokens: u64,
    pub avg_tokens_per_second: f64,
}

/// Collector for aggregating metrics from multiple requests
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    metrics: Vec<RequestMetrics>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Add a request metric
    pub fn add(&mut self, metric: RequestMetrics) {
        self.metrics.push(metric);
    }

    /// Get the number of collected metrics
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if the collector is empty
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Get an iterator over the metrics
    pub fn iter(&self) -> impl Iterator<Item = &RequestMetrics> {
        self.metrics.iter()
    }

    /// Compute aggregated statistics
    pub fn aggregate(&self) -> AggregatedMetrics {
        if self.metrics.is_empty() {
            return AggregatedMetrics::default();
        }

        let total_requests = self.metrics.len();
        let successful_requests = self.metrics.iter().filter(|m| m.success).count();
        let failed_requests = total_requests - successful_requests;

        // Collect all TTFT and total times
        let mut ttft_values: Vec<u64> = self.metrics.iter().map(|m| m.ttft_ms).collect();
        let mut total_time_values: Vec<u64> = self.metrics.iter().map(|m| m.total_time_ms).collect();

        ttft_values.sort_unstable();
        total_time_values.sort_unstable();

        // Calculate percentiles
        let p50_ttft_ms = percentile(&ttft_values, 50.0);
        let p95_ttft_ms = percentile(&ttft_values, 95.0);
        let p99_ttft_ms = percentile(&ttft_values, 99.0);
        let p50_total_time_ms = percentile(&total_time_values, 50.0);
        let p95_total_time_ms = percentile(&total_time_values, 95.0);
        let p99_total_time_ms = percentile(&total_time_values, 99.0);

        // Calculate averages
        let avg_ttft_ms = ttft_values.iter().sum::<u64>() as f64 / total_requests as f64;
        let avg_total_time_ms = total_time_values.iter().sum::<u64>() as f64 / total_requests as f64;

        // Token statistics
        let total_prompt_tokens: u64 = self.metrics.iter().map(|m| m.prompt_tokens as u64).sum();
        let total_completion_tokens: u64 = self.metrics.iter().map(|m| m.completion_tokens as u64).sum();
        let total_tokens: u64 = self.metrics.iter().map(|m| m.total_tokens as u64).sum();

        // Tokens per second (completion tokens / total time in seconds)
        let total_time_seconds = total_time_values.iter().sum::<u64>() as f64 / 1000.0;
        let avg_tokens_per_second = if total_time_seconds > 0.0 {
            total_completion_tokens as f64 / total_time_seconds
        } else {
            0.0
        };

        AggregatedMetrics {
            total_requests,
            successful_requests,
            failed_requests,
            avg_ttft_ms,
            p50_ttft_ms,
            p95_ttft_ms,
            p99_ttft_ms,
            avg_total_time_ms,
            p50_total_time_ms,
            p95_total_time_ms,
            p99_total_time_ms,
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            avg_tokens_per_second,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_ttft_ms: 0.0,
            p50_ttft_ms: 0.0,
            p95_ttft_ms: 0.0,
            p99_ttft_ms: 0.0,
            avg_total_time_ms: 0.0,
            p50_total_time_ms: 0.0,
            p95_total_time_ms: 0.0,
            p99_total_time_ms: 0.0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            total_tokens: 0,
            avg_tokens_per_second: 0.0,
        }
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[u64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let index = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)] as f64
}

/// Type alias for shared metrics collector
pub type SharedMetricsCollector = Arc<Mutex<MetricsCollector>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());

        collector.add(RequestMetrics {
            ttft_ms: 100,
            total_time_ms: 200,
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            success: true,
            status_code: 200,
            error_message: None,
        });

        assert_eq!(collector.len(), 1);
        assert!(!collector.is_empty());
    }

    #[test]
    fn test_aggregation() {
        let mut collector = MetricsCollector::new();

        for i in 0..10 {
            collector.add(RequestMetrics {
                ttft_ms: (i + 1) * 100,
                total_time_ms: (i + 1) * 200,
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                success: true,
                status_code: 200,
                error_message: None,
            });
        }

        let agg = collector.aggregate();
        assert_eq!(agg.total_requests, 10);
        assert_eq!(agg.successful_requests, 10);
        assert_eq!(agg.failed_requests, 0);
        assert_eq!(agg.total_prompt_tokens, 100);
        assert_eq!(agg.total_completion_tokens, 200);
    }
}
