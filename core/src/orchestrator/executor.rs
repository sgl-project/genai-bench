//! Orchestrator execution logic

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc, Semaphore};

use crate::config::ExperimentConfig;
use crate::error::BenchResult;
use crate::metrics::RequestRecord;
use crate::traits::{Sampler, StopCondition, VendorClient};
use crate::worker::{WorkerBuilder, WorkerStats};

use super::aggregator::aggregate_worker_stats;

/// Orchestrator manages the experiment lifecycle
///
/// Responsible for spawning workers, coordinating shutdown,
/// and collecting results.
pub struct Orchestrator {
    /// Experiment configuration
    pub(crate) config: ExperimentConfig,

    /// Vendor client (shared across workers)
    pub(crate) vendor: Arc<dyn VendorClient>,

    /// Sampler (shared across workers)
    pub(crate) sampler: Arc<dyn Sampler>,

    /// Metrics sender (cloned for each worker)
    pub(crate) metrics_tx: mpsc::Sender<RequestRecord>,

    /// Concurrency limiter
    pub(crate) semaphore: Arc<Semaphore>,

    /// Shutdown signal sender
    pub(crate) shutdown_tx: broadcast::Sender<()>,

    /// Shared request counter for fair work distribution
    pub(crate) request_counter: Arc<AtomicUsize>,
}

impl Orchestrator {
    /// Create a new orchestrator
    ///
    /// Use `OrchestratorBuilder` for a more ergonomic construction.
    pub fn new(
        config: ExperimentConfig,
        vendor: Arc<dyn VendorClient>,
        sampler: Arc<dyn Sampler>,
        metrics_tx: mpsc::Sender<RequestRecord>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.concurrency));
        let (shutdown_tx, _) = broadcast::channel(1);
        let request_counter = Arc::new(AtomicUsize::new(0));

        Self {
            config,
            vendor,
            sampler,
            metrics_tx,
            semaphore,
            shutdown_tx,
            request_counter,
        }
    }

    /// Get a shutdown signal receiver
    pub fn shutdown_receiver(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Trigger shutdown of all workers
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Get the experiment configuration
    pub fn config(&self) -> &ExperimentConfig {
        &self.config
    }

    /// Run the experiment
    ///
    /// Spawns worker tasks, waits for completion, and returns aggregated stats.
    pub async fn run(&self) -> BenchResult<Vec<WorkerStats>> {
        let start = Instant::now();
        let mut handles = Vec::with_capacity(self.config.concurrency);

        // Determine total request count for counter-based distribution
        let total_requests = match self.config.stop_condition {
            StopCondition::RequestCount(n) => Some(n),
            _ => None,
        };

        tracing::info!(
            concurrency = self.config.concurrency,
            stop_condition = ?self.config.stop_condition,
            rate_limit = ?self.config.rate_limit,
            "Starting experiment"
        );

        // Spawn worker tasks
        for worker_id in 0..self.config.concurrency {
            let mut builder = WorkerBuilder::new(worker_id)
                .vendor(Arc::clone(&self.vendor))
                .sampler(Arc::clone(&self.sampler))
                .metrics_tx(self.metrics_tx.clone())
                .semaphore(Arc::clone(&self.semaphore))
                .stop_condition(self.config.stop_condition.clone())
                .concurrency(self.config.concurrency)
                .rate_limit(self.config.rate_limit);

            // Use shared counter for RequestCount stop condition
            if let Some(total) = total_requests {
                builder = builder.request_counter(Arc::clone(&self.request_counter), total);
            }

            let worker = builder.build()?;
            let shutdown_rx = self.shutdown_tx.subscribe();

            handles.push(tokio::spawn(async move { worker.run(shutdown_rx).await }));
        }

        // Wait for all workers to complete
        let mut results = Vec::with_capacity(handles.len());
        let mut worker_failures = 0;
        for (idx, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(stats)) => {
                    tracing::debug!(
                        worker_id = idx,
                        completed = stats.completed,
                        errors = stats.errors,
                        "Worker completed"
                    );
                    results.push(stats);
                }
                Ok(Err(e)) => {
                    worker_failures += 1;
                    tracing::error!(worker_id = idx, error = %e, "Worker returned error");
                    // Continue collecting other results
                }
                Err(e) => {
                    worker_failures += 1;
                    tracing::error!(worker_id = idx, error = %e, "Worker task panicked");
                    // Continue collecting other results
                }
            }
        }

        // If all workers failed, return an error
        if results.is_empty() && worker_failures > 0 {
            return Err(crate::error::BenchError::orchestration(format!(
                "All {} workers failed to complete",
                worker_failures
            )));
        }

        let elapsed = start.elapsed();
        let aggregated = aggregate_worker_stats(&results);
        tracing::info!(
            elapsed_secs = elapsed.as_secs_f64(),
            total_completed = aggregated.total_completed,
            total_errors = aggregated.total_errors,
            rps = aggregated.requests_per_second,
            "Experiment completed"
        );

        Ok(results)
    }

    /// Run with Ctrl+C signal handling
    ///
    /// Automatically triggers graceful shutdown on Ctrl+C.
    pub async fn run_with_signal_handling(&self) -> BenchResult<Vec<WorkerStats>> {
        let shutdown_tx = self.shutdown_tx.clone();

        // Spawn signal handler task
        let signal_handle = tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    tracing::info!("Received Ctrl+C, initiating graceful shutdown...");
                    let _ = shutdown_tx.send(());
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to listen for Ctrl+C");
                }
            }
        });

        // Run the experiment
        let result = self.run().await;

        // Abort signal handler if still running
        signal_handle.abort();

        result
    }

    /// Run with a timeout
    ///
    /// Automatically triggers shutdown when timeout is reached.
    pub async fn run_with_timeout(&self, timeout: Duration) -> BenchResult<Vec<WorkerStats>> {
        let shutdown_tx = self.shutdown_tx.clone();

        // Spawn timeout task
        let timeout_handle = tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            tracing::info!("Timeout reached, initiating shutdown...");
            let _ = shutdown_tx.send(());
        });

        // Run the experiment
        let result = self.run().await;

        // Abort timeout task if still running
        timeout_handle.abort();

        result
    }
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("config", &self.config)
            .field("vendor", &self.vendor.vendor_name())
            .field("sampler", &self.sampler.name())
            .finish()
    }
}
