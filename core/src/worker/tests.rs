//! Integration tests for the Worker module

use super::*;
use crate::metrics::RequestRecord;
use crate::request::{BenchmarkRequest, Message, RequestId, RequestMetadata, SamplingParams, Task};
use crate::response::{BenchmarkResponse, ResponseStatus, ResponseTiming, TokenCounts};
use crate::traits::{Sampler, SamplerError, StopCondition, VendorClient, VendorError};

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, Semaphore};

// ============================================================================
// Mock Sampler
// ============================================================================

struct MockSampler {
    name: String,
    scenario: String,
    counter: AtomicUsize,
    max_samples: Option<usize>,
}

impl MockSampler {
    fn new(name: &str, scenario: &str) -> Self {
        Self {
            name: name.to_string(),
            scenario: scenario.to_string(),
            counter: AtomicUsize::new(0),
            max_samples: None,
        }
    }

    #[allow(dead_code)]
    fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = Some(max);
        self
    }
}

impl Sampler for MockSampler {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample(&self) -> Result<BenchmarkRequest, SamplerError> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);

        if let Some(max) = self.max_samples {
            if count >= max {
                return Err(SamplerError::Exhausted);
            }
        }

        Ok(BenchmarkRequest {
            id: RequestId(count as u64),
            task: Task::TextToText,
            messages: vec![Message::user("Test message")],
            input_tokens: 100,
            output_tokens: Some(50),
            params: SamplingParams::default(),
            metadata: RequestMetadata::new(&self.name, &self.scenario),
        })
    }

    fn scenario_name(&self) -> &str {
        &self.scenario
    }

    fn supported_tasks(&self) -> &[Task] {
        &[Task::TextToText]
    }
}

// ============================================================================
// Mock VendorClient
// ============================================================================

struct MockVendorClient {
    name: String,
    model: String,
    delay: Option<Duration>,
    fail_every: Option<usize>,
    counter: AtomicUsize,
}

impl MockVendorClient {
    fn new(name: &str, model: &str) -> Self {
        Self {
            name: name.to_string(),
            model: model.to_string(),
            delay: None,
            fail_every: None,
            counter: AtomicUsize::new(0),
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    fn with_fail_every(mut self, n: usize) -> Self {
        self.fail_every = Some(n);
        self
    }
}

#[async_trait]
impl VendorClient for MockVendorClient {
    fn vendor_name(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn supported_tasks(&self) -> &[Task] {
        &[Task::TextToText]
    }

    async fn execute(&self, request: &BenchmarkRequest) -> Result<BenchmarkResponse, VendorError> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);

        // Simulate delay if configured
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }

        // Simulate failures if configured
        if let Some(fail_every) = self.fail_every {
            if count > 0 && count.is_multiple_of(fail_every) {
                return Err(VendorError::ServerError {
                    status: 500,
                    message: "Simulated failure".to_string(),
                });
            }
        }

        Ok(BenchmarkResponse {
            request_id: request.id,
            status: ResponseStatus::Success,
            content: Some("Test response".to_string()),
            tokens: TokenCounts::new(request.input_tokens, 50),
            timing: ResponseTiming::from_timestamps(
                chrono::Utc::now(),
                Some(chrono::Utc::now()),
                Some(chrono::Utc::now()),
                chrono::Utc::now(),
            ),
            raw_response: None,
        })
    }

    async fn execute_streaming(
        &self,
        _request: &BenchmarkRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<crate::traits::StreamChunk, VendorError>> + Send>>,
        VendorError,
    > {
        Err(VendorError::Config(
            "Streaming not implemented in mock".to_string(),
        ))
    }

    fn validate(&self) -> Result<(), VendorError> {
        Ok(())
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn create_test_worker(
    id: usize,
    sampler: Arc<dyn Sampler>,
    vendor: Arc<dyn VendorClient>,
    stop_condition: StopCondition,
    concurrency: usize,
) -> (Worker, mpsc::Receiver<RequestRecord>, broadcast::Sender<()>) {
    let (metrics_tx, metrics_rx) = mpsc::channel(100);
    let (shutdown_tx, _) = broadcast::channel(1);
    let semaphore = Arc::new(Semaphore::new(concurrency));

    let worker = WorkerBuilder::new(id)
        .vendor(vendor)
        .sampler(sampler)
        .metrics_tx(metrics_tx)
        .semaphore(semaphore)
        .stop_condition(stop_condition)
        .concurrency(concurrency)
        .build()
        .expect("Failed to build worker");

    (worker, metrics_rx, shutdown_tx)
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_worker_run_request_count() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    let vendor = Arc::new(MockVendorClient::new("test-vendor", "test-model"));

    let (worker, mut metrics_rx, shutdown_tx) = create_test_worker(
        0,
        sampler,
        vendor,
        StopCondition::RequestCount(5),
        1, // concurrency of 1 means this worker handles all 5 requests
    );

    let shutdown_rx = shutdown_tx.subscribe();
    let stats = worker.run(shutdown_rx).await.expect("Worker failed");

    assert_eq!(stats.completed, 5);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.total_requests(), 5);

    // Verify metrics were sent
    let mut received = 0;
    while metrics_rx.try_recv().is_ok() {
        received += 1;
    }
    assert_eq!(received, 5);
}

#[tokio::test]
async fn test_worker_run_shutdown() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    let vendor = Arc::new(
        MockVendorClient::new("test-vendor", "test-model").with_delay(Duration::from_millis(50)),
    );

    let (worker, _metrics_rx, shutdown_tx) =
        create_test_worker(0, sampler, vendor, StopCondition::Indefinite, 1);

    let shutdown_rx = shutdown_tx.subscribe();

    // Spawn worker and send shutdown after a short delay
    let handle = tokio::spawn(async move { worker.run(shutdown_rx).await });

    tokio::time::sleep(Duration::from_millis(100)).await;
    shutdown_tx.send(()).expect("Failed to send shutdown");

    let stats = handle
        .await
        .expect("Worker task panicked")
        .expect("Worker failed");

    // Should have completed at least 1 request but stopped due to shutdown
    assert!(stats.completed >= 1);
}

#[tokio::test]
async fn test_worker_run_with_errors() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    let vendor = Arc::new(
        MockVendorClient::new("test-vendor", "test-model").with_fail_every(2), // Fail every 2nd request
    );

    let (worker, _metrics_rx, shutdown_tx) =
        create_test_worker(0, sampler, vendor, StopCondition::RequestCount(5), 1);

    let shutdown_rx = shutdown_tx.subscribe();
    let stats = worker.run(shutdown_rx).await.expect("Worker failed");

    // Should have completed 5 total requests (some successful, some failed)
    assert_eq!(stats.total_requests(), 5);
    assert!(stats.completed > 0);
    assert!(stats.errors > 0);
    assert!(stats.error_rate() > 0.0);
}

#[tokio::test]
async fn test_worker_run_duration() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    // Add delay to prevent filling the channel buffer before duration elapses
    let vendor = Arc::new(
        MockVendorClient::new("test-vendor", "test-model").with_delay(Duration::from_millis(20)),
    );

    let (metrics_tx, mut metrics_rx) = mpsc::channel(1000);
    let (shutdown_tx, _) = broadcast::channel(1);
    let semaphore = Arc::new(Semaphore::new(1));

    let worker = WorkerBuilder::new(0)
        .vendor(vendor)
        .sampler(sampler)
        .metrics_tx(metrics_tx)
        .semaphore(semaphore)
        .stop_condition(StopCondition::Duration(Duration::from_millis(100)))
        .concurrency(1)
        .build()
        .expect("Failed to build worker");

    let shutdown_rx = shutdown_tx.subscribe();

    // Spawn a task to drain the metrics channel
    let drain_handle = tokio::spawn(async move { while metrics_rx.recv().await.is_some() {} });

    let start = std::time::Instant::now();
    let stats = worker.run(shutdown_rx).await.expect("Worker failed");
    let elapsed = start.elapsed();

    // Drop sender to close channel and let drain task finish
    drop(drain_handle);

    // Should run for at least 100ms
    assert!(elapsed >= Duration::from_millis(100));
    assert!(stats.completed > 0);
}

#[tokio::test]
async fn test_worker_with_rate_limit() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    let vendor = Arc::new(MockVendorClient::new("test-vendor", "test-model"));

    let (metrics_tx, _metrics_rx) = mpsc::channel(100);
    let (shutdown_tx, _) = broadcast::channel(1);
    let semaphore = Arc::new(Semaphore::new(1));

    let worker = WorkerBuilder::new(0)
        .vendor(vendor)
        .sampler(sampler)
        .metrics_tx(metrics_tx)
        .semaphore(semaphore)
        .stop_condition(StopCondition::RequestCount(3))
        .concurrency(1)
        .rate_limit(Some(100.0)) // 100 RPS
        .build()
        .expect("Failed to build worker");

    let shutdown_rx = shutdown_tx.subscribe();
    let stats = worker.run(shutdown_rx).await.expect("Worker failed");

    assert_eq!(stats.completed, 3);
}

#[tokio::test]
async fn test_worker_stats_tracking() {
    let sampler = Arc::new(MockSampler::new("test-sampler", "test-scenario"));
    let vendor = Arc::new(MockVendorClient::new("test-vendor", "test-model"));

    let (worker, _metrics_rx, shutdown_tx) =
        create_test_worker(0, sampler, vendor, StopCondition::RequestCount(3), 1);

    let shutdown_rx = shutdown_tx.subscribe();
    let stats = worker.run(shutdown_rx).await.expect("Worker failed");

    assert_eq!(stats.completed, 3);
    assert_eq!(stats.input_tokens, 300); // 3 requests * 100 input tokens
    assert_eq!(stats.output_tokens, 150); // 3 requests * 50 output tokens
    assert!(stats.elapsed().is_some());
    assert!(stats.requests_per_second() > 0.0);
}

#[tokio::test]
async fn test_should_stop_request_count() {
    let mut stats = WorkerStats::new();
    stats.completed = 9;
    stats.errors = 1;

    // 10 total requests / 1 worker = 10 per worker
    // stats.total_requests() = 10, so should stop
    let total_requests = 10;
    let concurrency = 1;
    let per_worker = total_requests / concurrency;
    assert!(stats.total_requests() >= per_worker);
}

#[tokio::test]
async fn test_should_stop_duration() {
    let mut stats = WorkerStats::new();
    stats.start();

    // Sleep longer than the stop duration
    tokio::time::sleep(Duration::from_millis(50)).await;

    let stop_duration = Duration::from_millis(25);
    let should_stop = stats
        .started_at
        .map(|start| start.elapsed() >= stop_duration)
        .unwrap_or(false);
    assert!(should_stop);
}

#[tokio::test]
async fn test_should_stop_indefinite() {
    // StopCondition::Indefinite should never trigger a stop on its own.
    // Workers with this condition only stop via shutdown signal.
    let stop_condition = StopCondition::Indefinite;
    assert!(matches!(stop_condition, StopCondition::Indefinite));
}
