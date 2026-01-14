//! Builder pattern for Worker construction

use crate::error::{BenchError, BenchResult};
use crate::metrics::RequestRecord;
use crate::traits::{Sampler, StopCondition, VendorClient};

use super::executor::Worker;
use super::rate_limiter::RequestRateLimiter;

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Builder for creating Worker instances
///
/// Provides ergonomic construction with validation.
///
/// # Example
/// ```ignore
/// let worker = WorkerBuilder::new(0)
///     .vendor(vendor)
///     .sampler(sampler)
///     .metrics_tx(tx)
///     .semaphore(semaphore)
///     .stop_condition(StopCondition::RequestCount(100))
///     .concurrency(10)
///     .build()?;
/// ```
pub struct WorkerBuilder {
    id: usize,
    vendor: Option<Arc<dyn VendorClient>>,
    sampler: Option<Arc<dyn Sampler>>,
    metrics_tx: Option<mpsc::Sender<RequestRecord>>,
    semaphore: Option<Arc<Semaphore>>,
    rate_limit: Option<f64>,
    stop_condition: Option<StopCondition>,
    concurrency: Option<usize>,
    request_counter: Option<Arc<AtomicUsize>>,
    total_requests: Option<usize>,
}

impl WorkerBuilder {
    /// Create a new builder with the given worker ID
    pub fn new(id: usize) -> Self {
        Self {
            id,
            vendor: None,
            sampler: None,
            metrics_tx: None,
            semaphore: None,
            rate_limit: None,
            stop_condition: None,
            concurrency: None,
            request_counter: None,
            total_requests: None,
        }
    }

    /// Set the vendor client
    pub fn vendor(mut self, vendor: Arc<dyn VendorClient>) -> Self {
        self.vendor = Some(vendor);
        self
    }

    /// Set the sampler
    pub fn sampler(mut self, sampler: Arc<dyn Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Set the metrics channel sender
    pub fn metrics_tx(mut self, tx: mpsc::Sender<RequestRecord>) -> Self {
        self.metrics_tx = Some(tx);
        self
    }

    /// Set the concurrency semaphore
    pub fn semaphore(mut self, semaphore: Arc<Semaphore>) -> Self {
        self.semaphore = Some(semaphore);
        self
    }

    /// Set the rate limit (requests per second)
    pub fn rate_limit(mut self, rps: Option<f64>) -> Self {
        self.rate_limit = rps;
        self
    }

    /// Set the stop condition
    pub fn stop_condition(mut self, condition: StopCondition) -> Self {
        self.stop_condition = Some(condition);
        self
    }

    /// Set the concurrency level
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// Set a shared request counter for fair work distribution
    pub fn request_counter(mut self, counter: Arc<AtomicUsize>, total: usize) -> Self {
        self.request_counter = Some(counter);
        self.total_requests = Some(total);
        self
    }

    /// Build the Worker
    ///
    /// # Errors
    /// Returns an error if any required field is missing.
    pub fn build(self) -> BenchResult<Worker> {
        let vendor = self.vendor.ok_or(BenchError::missing_config("vendor"))?;
        let sampler = self.sampler.ok_or(BenchError::missing_config("sampler"))?;
        let metrics_tx = self
            .metrics_tx
            .ok_or(BenchError::missing_config("metrics_tx"))?;
        let semaphore = self
            .semaphore
            .ok_or(BenchError::missing_config("semaphore"))?;
        let stop_condition = self
            .stop_condition
            .ok_or(BenchError::missing_config("stop_condition"))?;
        let concurrency = self
            .concurrency
            .ok_or(BenchError::missing_config("concurrency"))?;

        let rate_limiter = RequestRateLimiter::new(self.rate_limit);

        let mut worker = Worker::new(
            self.id,
            vendor,
            sampler,
            metrics_tx,
            semaphore,
            rate_limiter,
            stop_condition,
            concurrency,
        );

        if let (Some(counter), Some(total)) = (self.request_counter, self.total_requests) {
            worker = worker.with_request_counter(counter, total);
        }

        Ok(worker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_missing_vendor() {
        let result = WorkerBuilder::new(0)
            .concurrency(1)
            .stop_condition(StopCondition::RequestCount(10))
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("vendor"));
    }

    #[test]
    fn test_builder_missing_sampler() {
        let result = WorkerBuilder::new(0)
            .concurrency(1)
            .stop_condition(StopCondition::RequestCount(10))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_metrics_tx() {
        let result = WorkerBuilder::new(0)
            .concurrency(1)
            .stop_condition(StopCondition::RequestCount(10))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_semaphore() {
        let result = WorkerBuilder::new(0)
            .concurrency(1)
            .stop_condition(StopCondition::RequestCount(10))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_stop_condition() {
        let result = WorkerBuilder::new(0).concurrency(1).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_concurrency() {
        let result = WorkerBuilder::new(0)
            .stop_condition(StopCondition::RequestCount(10))
            .build();

        assert!(result.is_err());
    }
}
