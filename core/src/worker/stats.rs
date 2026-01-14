//! Worker statistics tracking

use std::time::Instant;

/// Statistics tracked by each worker
#[derive(Debug, Default, Clone)]
pub struct WorkerStats {
    /// Number of successfully completed requests
    pub completed: usize,

    /// Number of failed requests
    pub errors: usize,

    /// Total input tokens processed
    pub input_tokens: usize,

    /// Total output tokens generated
    pub output_tokens: usize,

    /// Worker start time
    pub started_at: Option<Instant>,

    /// Worker end time
    pub ended_at: Option<Instant>,
}

impl WorkerStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Start tracking (records start time)
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
    }

    /// Stop tracking (records end time)
    pub fn stop(&mut self) {
        self.ended_at = Some(Instant::now());
    }

    /// Get total number of requests (completed + errors)
    pub fn total_requests(&self) -> usize {
        self.completed + self.errors
    }

    /// Get success rate as a percentage (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_requests() == 0 {
            0.0
        } else {
            self.completed as f64 / self.total_requests() as f64
        }
    }

    /// Get error rate as a percentage (0.0 - 1.0)
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    /// Get total tokens (input + output)
    pub fn total_tokens(&self) -> usize {
        self.input_tokens + self.output_tokens
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.started_at.map(|start| {
            self.ended_at
                .map(|end| end.duration_since(start))
                .unwrap_or_else(|| start.elapsed())
        })
    }

    /// Get requests per second
    pub fn requests_per_second(&self) -> f64 {
        self.elapsed()
            .map(|d| {
                let secs = d.as_secs_f64();
                if secs > 0.0 {
                    self.total_requests() as f64 / secs
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Get tokens per second (output tokens)
    pub fn tokens_per_second(&self) -> f64 {
        self.elapsed()
            .map(|d| {
                let secs = d.as_secs_f64();
                if secs > 0.0 {
                    self.output_tokens as f64 / secs
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Record a successful request
    pub fn record_success(&mut self, input_tokens: usize, output_tokens: usize) {
        self.completed += 1;
        self.input_tokens += input_tokens;
        self.output_tokens += output_tokens;
    }

    /// Record a failed request
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Merge stats from another worker
    pub fn merge(&mut self, other: &WorkerStats) {
        self.completed += other.completed;
        self.errors += other.errors;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_stats_defaults() {
        let stats = WorkerStats::default();
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
        assert!(stats.started_at.is_none());
        assert!(stats.ended_at.is_none());
    }

    #[test]
    fn test_worker_stats_total_requests() {
        let mut stats = WorkerStats::new();
        stats.completed = 10;
        stats.errors = 2;
        assert_eq!(stats.total_requests(), 12);
    }

    #[test]
    fn test_worker_stats_success_rate() {
        let mut stats = WorkerStats::new();
        stats.completed = 8;
        stats.errors = 2;
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_worker_stats_success_rate_zero_requests() {
        let stats = WorkerStats::new();
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_worker_stats_record_success() {
        let mut stats = WorkerStats::new();
        stats.record_success(100, 50);
        stats.record_success(200, 100);

        assert_eq!(stats.completed, 2);
        assert_eq!(stats.input_tokens, 300);
        assert_eq!(stats.output_tokens, 150);
        assert_eq!(stats.total_tokens(), 450);
    }

    #[test]
    fn test_worker_stats_record_error() {
        let mut stats = WorkerStats::new();
        stats.record_error();
        stats.record_error();

        assert_eq!(stats.errors, 2);
        assert_eq!(stats.completed, 0);
    }

    #[test]
    fn test_worker_stats_merge() {
        let mut stats1 = WorkerStats::new();
        stats1.completed = 10;
        stats1.errors = 1;
        stats1.input_tokens = 1000;
        stats1.output_tokens = 500;

        let mut stats2 = WorkerStats::new();
        stats2.completed = 5;
        stats2.errors = 2;
        stats2.input_tokens = 500;
        stats2.output_tokens = 250;

        stats1.merge(&stats2);

        assert_eq!(stats1.completed, 15);
        assert_eq!(stats1.errors, 3);
        assert_eq!(stats1.input_tokens, 1500);
        assert_eq!(stats1.output_tokens, 750);
    }

    #[test]
    fn test_worker_stats_start_stop() {
        let mut stats = WorkerStats::new();
        assert!(stats.elapsed().is_none());

        stats.start();
        assert!(stats.started_at.is_some());
        assert!(stats.elapsed().is_some());

        std::thread::sleep(std::time::Duration::from_millis(10));
        stats.stop();

        let elapsed = stats.elapsed().unwrap();
        assert!(elapsed >= std::time::Duration::from_millis(10));
    }
}
