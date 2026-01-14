//! Result aggregation from multiple workers

use std::time::Duration;

use crate::worker::WorkerStats;

/// Aggregated statistics from all workers
#[derive(Debug, Clone, Default)]
pub struct AggregatedStats {
    /// Number of workers that completed
    pub total_workers: usize,

    /// Total successful requests
    pub total_completed: usize,

    /// Total failed requests
    pub total_errors: usize,

    /// Total input tokens processed
    pub total_input_tokens: usize,

    /// Total output tokens generated
    pub total_output_tokens: usize,

    /// Maximum duration across all workers
    pub total_duration: Duration,

    /// Overall requests per second
    pub requests_per_second: f64,

    /// Overall tokens per second (output)
    pub tokens_per_second: f64,
}

impl AggregatedStats {
    /// Get the total number of requests (completed + errors)
    pub fn total_requests(&self) -> usize {
        self.total_completed + self.total_errors
    }

    /// Get the success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests();
        if total > 0 {
            self.total_completed as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the error rate (0.0 - 1.0)
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

/// Aggregate statistics from multiple workers
pub fn aggregate_worker_stats(stats: &[WorkerStats]) -> AggregatedStats {
    if stats.is_empty() {
        return AggregatedStats::default();
    }

    let total_completed: usize = stats.iter().map(|s| s.completed).sum();
    let total_errors: usize = stats.iter().map(|s| s.errors).sum();
    let total_input_tokens: usize = stats.iter().map(|s| s.input_tokens).sum();
    let total_output_tokens: usize = stats.iter().map(|s| s.output_tokens).sum();

    // Use the maximum elapsed time across all workers
    let total_duration = stats
        .iter()
        .filter_map(|s| s.elapsed())
        .max()
        .unwrap_or(Duration::ZERO);

    let secs = total_duration.as_secs_f64();
    let rate_multiplier = if secs > 0.0 { 1.0 / secs } else { 0.0 };
    let requests_per_second = total_completed as f64 * rate_multiplier;
    let tokens_per_second = total_output_tokens as f64 * rate_multiplier;

    AggregatedStats {
        total_workers: stats.len(),
        total_completed,
        total_errors,
        total_input_tokens,
        total_output_tokens,
        total_duration,
        requests_per_second,
        tokens_per_second,
    }
}
