//! Worker execution loop

use crate::error::{BenchError, BenchResult};
use crate::metrics::RequestRecord;
use crate::traits::{Sampler, StopCondition, VendorClient};

use super::rate_limiter::RequestRateLimiter;
use super::stats::WorkerStats;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc, Semaphore};

/// Worker executes requests in a loop: sample -> execute -> report -> repeat
///
/// Workers are stateless tokio tasks managed by the Orchestrator.
/// They share a Sampler and VendorClient via Arc, and send metrics
/// through an mpsc channel.
pub struct Worker {
    /// Unique worker identifier
    id: usize,

    /// Vendor client (shared across workers via Arc)
    vendor: Arc<dyn VendorClient>,

    /// Sampler (shared across workers via Arc)
    sampler: Arc<dyn Sampler>,

    /// Channel sender for metrics/request records
    metrics_tx: mpsc::Sender<RequestRecord>,

    /// Concurrency limiter (shared semaphore)
    semaphore: Arc<Semaphore>,

    /// Rate limiter (per-worker or shared)
    rate_limiter: RequestRateLimiter,

    /// Stop condition
    stop_condition: StopCondition,

    /// Concurrency level (for calculating per-worker request count)
    concurrency: usize,

    /// Shared request counter for fair distribution
    request_counter: Option<Arc<AtomicUsize>>,

    /// Total requests (when using counter-based distribution)
    total_requests: Option<usize>,
}

impl Worker {
    /// Create a new worker
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        vendor: Arc<dyn VendorClient>,
        sampler: Arc<dyn Sampler>,
        metrics_tx: mpsc::Sender<RequestRecord>,
        semaphore: Arc<Semaphore>,
        rate_limiter: RequestRateLimiter,
        stop_condition: StopCondition,
        concurrency: usize,
    ) -> Self {
        Self {
            id,
            vendor,
            sampler,
            metrics_tx,
            semaphore,
            rate_limiter,
            stop_condition,
            concurrency,
            request_counter: None,
            total_requests: None,
        }
    }

    /// Set a shared request counter for fair work distribution
    ///
    /// When set, workers will use atomic increment to claim requests,
    /// ensuring all requests are executed even when total doesn't divide evenly.
    pub fn with_request_counter(
        mut self,
        counter: Arc<AtomicUsize>,
        total_requests: usize,
    ) -> Self {
        self.request_counter = Some(counter);
        self.total_requests = Some(total_requests);
        self
    }

    /// Run the worker loop
    ///
    /// Returns WorkerStats when complete (either via stop condition or shutdown signal).
    pub async fn run(self, mut shutdown: broadcast::Receiver<()>) -> BenchResult<WorkerStats> {
        let mut stats = WorkerStats::new();
        stats.start();

        tracing::debug!(worker_id = self.id, "Worker started");

        loop {
            // Check stop condition BEFORE claiming work
            if self.should_stop(&stats) {
                tracing::debug!(
                    worker_id = self.id,
                    completed = stats.completed,
                    errors = stats.errors,
                    "Worker reached stop condition"
                );
                break;
            }

            // Try to claim a request slot (for counter-based distribution)
            if !self.try_claim_request() {
                tracing::debug!(
                    worker_id = self.id,
                    "No more requests to claim, worker stopping"
                );
                break;
            }

            tokio::select! {
                biased;

                // Check for shutdown signal (highest priority)
                _ = shutdown.recv() => {
                    tracing::debug!(worker_id = self.id, "Worker received shutdown signal");
                    break;
                }

                // Execute next request
                result = self.execute_one() => {
                    match result {
                        Ok(record) => {
                            stats.record_success(record.input_tokens, record.output_tokens);

                            // Send metrics to aggregator (ignore send errors on shutdown)
                            if self.metrics_tx.send(record).await.is_err() {
                                tracing::debug!(
                                    worker_id = self.id,
                                    "Metrics channel closed, worker stopping"
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            stats.record_error();
                            tracing::warn!(
                                worker_id = self.id,
                                error = %e,
                                "Request failed"
                            );
                            // NOTE: Error details are logged but not sent to metrics channel.
                            // The Orchestrator aggregates error counts from WorkerStats.
                            // Future: Consider ErrorRecord type for detailed error analytics.
                        }
                    }
                }
            }
        }

        stats.stop();
        tracing::debug!(
            worker_id = self.id,
            completed = stats.completed,
            errors = stats.errors,
            elapsed_ms = ?stats.elapsed().map(|d| d.as_millis()),
            "Worker finished"
        );

        Ok(stats)
    }

    /// Execute a single request
    async fn execute_one(&self) -> BenchResult<RequestRecord> {
        // 1. Apply rate limiting (waits if necessary)
        self.rate_limiter.wait().await;

        // 2. Acquire semaphore permit (for concurrency control)
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| BenchError::shutdown())?;

        // 3. Sample request from sampler
        let request = self
            .sampler
            .sample()
            .map_err(|e| BenchError::sampler(e.to_string()))?;
        let scenario = self.sampler.scenario_name().to_string();

        // 4. Execute and measure
        let start = Instant::now();
        let response = self
            .vendor
            .execute(&request)
            .await
            .map_err(|e| BenchError::vendor(e.to_error_kind(), e.to_string()))?;
        let e2e_latency = start.elapsed();

        // 5. Build and return request record
        Ok(RequestRecord {
            request_id: request.id,
            scenario,
            input_tokens: request.input_tokens,
            output_tokens: response.tokens.output,
            ttft_ms: response.timing.ttft_ms,
            tpot_ms: response.timing.tpot_ms(),
            e2e_ms: e2e_latency.as_secs_f64() * 1000.0,
            status: response.status,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Try to claim a request slot from the shared counter
    ///
    /// Returns `true` if a slot was claimed, `false` if no more slots available.
    /// Always returns `true` if not using counter-based distribution.
    fn try_claim_request(&self) -> bool {
        if let (Some(counter), Some(total)) = (&self.request_counter, self.total_requests) {
            // Atomically try to claim the next request slot
            let claimed = counter.fetch_add(1, Ordering::SeqCst);
            if claimed >= total {
                // Rollback: we over-claimed due to concurrent access near the limit.
                // This ensures the counter stays accurate for other workers still checking.
                counter.fetch_sub(1, Ordering::SeqCst);
                return false;
            }
            true
        } else {
            // Not using counter-based distribution, always allow
            true
        }
    }

    /// Check if the worker should stop based on stop condition
    fn should_stop(&self, stats: &WorkerStats) -> bool {
        match &self.stop_condition {
            StopCondition::RequestCount(total) => {
                // When using a shared counter, try_claim_request is responsible for
                // enforcing the request limit via atomic increment. We only check
                // stats-based stopping for the legacy per-worker division mode.
                if self.request_counter.is_some() {
                    false
                } else {
                    // Fall back to per-worker division
                    let per_worker = total / self.concurrency;
                    stats.total_requests() >= per_worker
                }
            }
            StopCondition::Duration(duration) => stats
                .started_at
                .map(|start| start.elapsed() >= *duration)
                .unwrap_or(false),
            StopCondition::Indefinite => false, // Only stops via shutdown signal
        }
    }

    /// Get the worker ID
    pub fn id(&self) -> usize {
        self.id
    }
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("id", &self.id)
            .field("vendor", &self.vendor.vendor_name())
            .field("sampler", &self.sampler.name())
            .field("rate_limiter", &self.rate_limiter)
            .field("stop_condition", &self.stop_condition)
            .field("concurrency", &self.concurrency)
            .finish()
    }
}
