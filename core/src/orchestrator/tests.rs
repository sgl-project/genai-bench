//! Tests for the Orchestrator module

use super::aggregator::{aggregate_worker_stats, AggregatedStats};
use super::builder::OrchestratorBuilder;
use crate::request::{BenchmarkRequest, Message, RequestId, RequestMetadata, SamplingParams, Task};
use crate::response::{BenchmarkResponse, ResponseStatus, ResponseTiming, TokenCounts};
use crate::traits::{Sampler, SamplerError, StopCondition, StreamChunk, VendorClient, VendorError};
use crate::worker::WorkerStats;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    #[allow(dead_code)]
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

        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }

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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, VendorError>> + Send>>, VendorError>
    {
        Err(VendorError::Config(
            "Streaming not implemented in mock".to_string(),
        ))
    }

    fn validate(&self) -> Result<(), VendorError> {
        Ok(())
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[test]
fn test_aggregated_stats_default() {
    let stats = AggregatedStats::default();
    assert_eq!(stats.total_workers, 0);
    assert_eq!(stats.total_completed, 0);
    assert_eq!(stats.total_errors, 0);
    assert_eq!(stats.success_rate(), 0.0);
}

#[test]
fn test_aggregated_stats_success_rate() {
    let stats = AggregatedStats {
        total_completed: 90,
        total_errors: 10,
        ..Default::default()
    };
    assert!((stats.success_rate() - 0.9).abs() < 0.001);
    assert!((stats.error_rate() - 0.1).abs() < 0.001);
}

#[test]
fn test_aggregate_worker_stats_empty() {
    let stats = aggregate_worker_stats(&[]);
    assert_eq!(stats.total_workers, 0);
    assert_eq!(stats.total_completed, 0);
}

#[test]
fn test_aggregate_worker_stats() {
    let mut s1 = WorkerStats::new();
    s1.completed = 50;
    s1.errors = 5;
    s1.input_tokens = 5000;
    s1.output_tokens = 2500;
    s1.start();
    std::thread::sleep(Duration::from_millis(10));
    s1.stop();

    let mut s2 = WorkerStats::new();
    s2.completed = 50;
    s2.errors = 5;
    s2.input_tokens = 5000;
    s2.output_tokens = 2500;
    s2.start();
    std::thread::sleep(Duration::from_millis(10));
    s2.stop();

    let aggregated = aggregate_worker_stats(&[s1, s2]);

    assert_eq!(aggregated.total_workers, 2);
    assert_eq!(aggregated.total_completed, 100);
    assert_eq!(aggregated.total_errors, 10);
    assert_eq!(aggregated.total_input_tokens, 10000);
    assert_eq!(aggregated.total_output_tokens, 5000);
    assert!(aggregated.total_duration >= Duration::from_millis(10));
}

#[test]
fn test_builder_missing_vendor() {
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let result = OrchestratorBuilder::new()
        .sampler(sampler)
        .concurrency(1)
        .build();

    assert!(result.is_err());
}

#[test]
fn test_builder_missing_sampler() {
    let vendor = Arc::new(MockVendorClient::new("test", "model"));

    let result = OrchestratorBuilder::new()
        .vendor(vendor)
        .concurrency(1)
        .build();

    assert!(result.is_err());
}

#[test]
fn test_builder_invalid_config() {
    let vendor = Arc::new(MockVendorClient::new("test", "model"));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let result = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(0) // Invalid
        .build();

    assert!(result.is_err());
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_orchestrator_run_basic() {
    let vendor = Arc::new(MockVendorClient::new("test", "model"));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let (orchestrator, mut metrics_rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(2)
        .stop_condition(StopCondition::RequestCount(10))
        .build()
        .expect("Failed to build orchestrator");

    // Spawn a task to drain the metrics channel
    let drain_handle = tokio::spawn(async move {
        let mut count = 0;
        while metrics_rx.recv().await.is_some() {
            count += 1;
        }
        count
    });

    let stats = orchestrator.run().await.expect("Run failed");

    // Drop orchestrator to close metrics channel
    drop(orchestrator);

    let metrics_count = drain_handle.await.expect("Drain task failed");

    // Should have 2 worker stats
    assert_eq!(stats.len(), 2);

    // Total completed should be 10
    let total: usize = stats.iter().map(|s| s.completed).sum();
    assert_eq!(total, 10);

    // Metrics count should match completed requests
    assert_eq!(metrics_count, 10);
}

#[tokio::test]
async fn test_orchestrator_shutdown() {
    let vendor =
        Arc::new(MockVendorClient::new("test", "model").with_delay(Duration::from_millis(50)));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let (orchestrator, _metrics_rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(2)
        .stop_condition(StopCondition::Indefinite)
        .build()
        .expect("Failed to build orchestrator");

    // Get shutdown handle before running
    let shutdown_tx = orchestrator.shutdown_tx.clone();

    // Spawn the orchestrator run
    let run_handle = tokio::spawn(async move { orchestrator.run().await });

    // Wait a bit then trigger shutdown
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _ = shutdown_tx.send(());

    let stats = run_handle
        .await
        .expect("Run task panicked")
        .expect("Run failed");

    // Should have stats from both workers
    assert_eq!(stats.len(), 2);

    // At least some requests should have completed
    let total: usize = stats.iter().map(|s| s.completed).sum();
    assert!(total > 0);
}

#[tokio::test]
async fn test_orchestrator_with_timeout() {
    let vendor =
        Arc::new(MockVendorClient::new("test", "model").with_delay(Duration::from_millis(20)));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let (orchestrator, mut metrics_rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(2)
        .stop_condition(StopCondition::Indefinite)
        .build()
        .expect("Failed to build orchestrator");

    // Spawn drain task
    let drain_handle = tokio::spawn(async move { while metrics_rx.recv().await.is_some() {} });

    let start = Instant::now();
    let stats = orchestrator
        .run_with_timeout(Duration::from_millis(100))
        .await
        .expect("Run failed");
    let elapsed = start.elapsed();

    drop(orchestrator);
    let _ = drain_handle.await;

    // Should complete around the timeout
    assert!(elapsed >= Duration::from_millis(100));
    assert!(elapsed < Duration::from_millis(300)); // Allow some slack

    // Should have completed some requests
    let total: usize = stats.iter().map(|s| s.completed).sum();
    assert!(total > 0);
}

#[tokio::test]
async fn test_orchestrator_concurrency() {
    let vendor =
        Arc::new(MockVendorClient::new("test", "model").with_delay(Duration::from_millis(50)));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let (orchestrator, mut metrics_rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(5)
        .stop_condition(StopCondition::RequestCount(10))
        .build()
        .expect("Failed to build orchestrator");

    // Drain metrics
    let drain_handle = tokio::spawn(async move { while metrics_rx.recv().await.is_some() {} });

    let start = Instant::now();
    let stats = orchestrator.run().await.expect("Run failed");
    let elapsed = start.elapsed();

    drop(orchestrator);
    let _ = drain_handle.await;

    // With 5 workers doing 10 requests at 50ms each, should take ~100ms
    // (2 batches of 5 requests each)
    assert!(elapsed < Duration::from_millis(300)); // Much less than 500ms (serial)

    // All 5 workers should have participated
    assert_eq!(stats.len(), 5);
}

#[tokio::test]
async fn test_orchestrator_rate_limit() {
    let vendor = Arc::new(MockVendorClient::new("test", "model"));
    let sampler = Arc::new(MockSampler::new("test", "scenario"));

    let (orchestrator, mut metrics_rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(1)
        .stop_condition(StopCondition::RequestCount(5))
        .rate_limit(Some(100.0)) // 100 RPS
        .build()
        .expect("Failed to build orchestrator");

    // Drain metrics
    let drain_handle = tokio::spawn(async move { while metrics_rx.recv().await.is_some() {} });

    let stats = orchestrator.run().await.expect("Run failed");

    drop(orchestrator);
    let _ = drain_handle.await;

    // Should complete all requests
    let total: usize = stats.iter().map(|s| s.completed).sum();
    assert_eq!(total, 5);
}

#[tokio::test]
async fn test_orchestrator_debug_format() {
    let vendor = Arc::new(MockVendorClient::new("test-vendor", "test-model"));
    let sampler = Arc::new(MockSampler::new("test-sampler", "scenario"));

    let (orchestrator, _rx) = OrchestratorBuilder::new()
        .vendor(vendor)
        .sampler(sampler)
        .concurrency(1)
        .build()
        .expect("Failed to build");

    let debug = format!("{:?}", orchestrator);
    assert!(debug.contains("Orchestrator"));
    assert!(debug.contains("test-vendor"));
    assert!(debug.contains("test-sampler"));
}
