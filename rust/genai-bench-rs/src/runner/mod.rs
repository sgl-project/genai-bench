//! Benchmark runner orchestration
//!
//! The runner coordinates execution of benchmarks, managing:
//! - Request scheduling based on scenarios
//! - Progress tracking
//! - Metrics collection

use crate::metrics::MetricsCollector;
use crate::providers::{ChatRequest, Message, Provider};
use crate::sampling::PromptSampler;
use crate::scenarios::Scenario;
use crate::ui::UiMessage;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::time::sleep;

/// Configuration for benchmark execution
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub num_requests: usize,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: bool,
}

/// Runner for executing benchmarks
pub struct BenchmarkRunner<P: Provider> {
    provider: Arc<P>,
    sampler: Arc<PromptSampler>,
    scenario: Arc<Mutex<Box<dyn Scenario>>>,
    config: BenchmarkConfig,
}

impl<P: Provider + 'static> BenchmarkRunner<P> {
    /// Create a new benchmark runner
    pub fn new(
        provider: P,
        sampler: PromptSampler,
        scenario: Box<dyn Scenario>,
        config: BenchmarkConfig,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            sampler: Arc::new(sampler),
            scenario: Arc::new(Mutex::new(scenario)),
            config,
        }
    }

    /// Run the benchmark sequentially (for MVP)
    pub async fn run_sequential(&self) -> Result<MetricsCollector> {
        let mut collector = MetricsCollector::new();

        let pb = ProgressBar::new(self.config.num_requests as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for i in 0..self.config.num_requests {
            // Sample a prompt
            let prompt = self.sampler.sample();

            // Create request
            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }],
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                stream: Some(self.config.stream),
            };

            // Execute request
            let metrics = self.provider.chat(&request).await?;
            collector.add(metrics);

            pb.inc(1);

            // Apply scenario delay (except for last request)
            if i < self.config.num_requests - 1 {
                let delay = self.scenario.lock().await.next_delay();
                sleep(delay).await;
            }
        }

        pb.finish_with_message("Benchmark complete");

        Ok(collector)
    }

    /// Run the benchmark concurrently
    pub async fn run_concurrent(&self, concurrency: usize) -> Result<MetricsCollector> {
        let collector = Arc::new(Mutex::new(MetricsCollector::new()));
        let semaphore = Arc::new(Semaphore::new(concurrency));

        let pb = ProgressBar::new(self.config.num_requests as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut tasks = vec![];

        for i in 0..self.config.num_requests {
            let provider = self.provider.clone();
            let sampler = self.sampler.clone();
            let collector = collector.clone();
            let semaphore = semaphore.clone();
            let pb = pb.clone();
            let model = self.config.model.clone();
            let max_tokens = self.config.max_tokens;
            let temperature = self.config.temperature;
            let stream = self.config.stream;
            let scenario = self.scenario.clone();

            let task = tokio::spawn(async move {
                // Acquire semaphore permit to limit concurrency
                let _permit = semaphore.acquire().await.unwrap();

                // Apply scenario delay before sending request
                if i > 0 {
                    let delay = scenario.lock().await.next_delay();
                    sleep(delay).await;
                }

                // Sample a prompt
                let prompt = sampler.sample();

                // Create request
                let request = ChatRequest {
                    model: model.clone(),
                    messages: vec![Message {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    }],
                    max_tokens,
                    temperature,
                    stream: Some(stream),
                };

                // Execute request
                match provider.chat(&request).await {
                    Ok(metrics) => {
                        collector.lock().await.add(metrics);
                        pb.inc(1);
                    }
                    Err(e) => {
                        tracing::error!("Request {} failed: {}", i + 1, e);
                        pb.inc(1);
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            let _ = task.await;
        }

        pb.finish_with_message("Benchmark complete");

        // Extract collector from Arc<Mutex>
        let collector = match Arc::try_unwrap(collector) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.blocking_lock().clone(),
        };

        Ok(collector)
    }

    /// Run the benchmark concurrently with UI updates
    pub async fn run_concurrent_with_ui(
        &self,
        concurrency: usize,
        ui_sender: mpsc::Sender<UiMessage>,
    ) -> Result<MetricsCollector> {
        let collector = Arc::new(Mutex::new(MetricsCollector::new()));
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let completed = Arc::new(Mutex::new(0usize));

        // Send initial progress
        let _ = ui_sender
            .send(UiMessage::Progress {
                completed: 0,
                total: self.config.num_requests,
            })
            .await;

        let mut tasks = vec![];

        for i in 0..self.config.num_requests {
            let provider = self.provider.clone();
            let sampler = self.sampler.clone();
            let collector = collector.clone();
            let semaphore = semaphore.clone();
            let completed = completed.clone();
            let ui_sender = ui_sender.clone();
            let model = self.config.model.clone();
            let max_tokens = self.config.max_tokens;
            let temperature = self.config.temperature;
            let stream = self.config.stream;
            let scenario = self.scenario.clone();
            let total_requests = self.config.num_requests;

            let task = tokio::spawn(async move {
                // Acquire semaphore permit to limit concurrency
                let _permit = semaphore.acquire().await.unwrap();

                // Apply scenario delay before sending request
                if i > 0 {
                    let delay = scenario.lock().await.next_delay();
                    sleep(delay).await;
                }

                // Sample a prompt
                let prompt = sampler.sample();

                // Create request
                let request = ChatRequest {
                    model: model.clone(),
                    messages: vec![Message {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    }],
                    max_tokens,
                    temperature,
                    stream: Some(stream),
                };

                // Execute request
                match provider.chat(&request).await {
                    Ok(metrics) => {
                        // Send data point for charts
                        let input_throughput = if metrics.ttft_ms > 0 {
                            (metrics.prompt_tokens as f64 / metrics.ttft_ms as f64) * 1000.0
                        } else {
                            0.0
                        };
                        let output_throughput = if metrics.total_time_ms > metrics.ttft_ms {
                            let output_time = metrics.total_time_ms - metrics.ttft_ms;
                            (metrics.completion_tokens as f64 / output_time as f64) * 1000.0
                        } else {
                            0.0
                        };

                        let _ = ui_sender
                            .send(UiMessage::DataPoint {
                                ttft_ms: metrics.ttft_ms as f64,
                                total_time_ms: metrics.total_time_ms as f64,
                                input_throughput,
                                output_throughput,
                            })
                            .await;

                        collector.lock().await.add(metrics);

                        // Update completed count
                        let mut count = completed.lock().await;
                        *count += 1;
                        let current = *count;
                        drop(count);

                        // Send progress update
                        let _ = ui_sender
                            .send(UiMessage::Progress {
                                completed: current,
                                total: total_requests,
                            })
                            .await;

                        // Send metrics update every 10 requests
                        if current % 10 == 0 || current == total_requests {
                            let agg = collector.lock().await.aggregate();
                            let _ = ui_sender.send(UiMessage::Metrics(agg)).await;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Request {} failed: {}", i + 1, e);
                        let _ = ui_sender
                            .send(UiMessage::Error(format!("Request failed: {}", e)))
                            .await;
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            let _ = task.await;
        }

        // Send final metrics and completion
        let collector_guard = collector.lock().await;
        let agg = collector_guard.aggregate();
        drop(collector_guard);

        let _ = ui_sender.send(UiMessage::Metrics(agg)).await;
        let _ = ui_sender.send(UiMessage::Complete).await;

        // Extract collector from Arc<Mutex>
        let collector = match Arc::try_unwrap(collector) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.blocking_lock().clone(),
        };

        Ok(collector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::RequestMetrics;
    use crate::providers::{ChatRequest, Provider};
    use crate::scenarios::DeterministicScenario;
    use async_trait::async_trait;

    // Mock provider for testing
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn chat(&self, _request: &ChatRequest) -> Result<RequestMetrics> {
            Ok(RequestMetrics {
                ttft_ms: 100,
                total_time_ms: 200,
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                success: true,
                status_code: 200,
                error_message: None,
            })
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_runner_sequential() {
        let provider = MockProvider;
        let sampler = PromptSampler::from_prompt("Test prompt".to_string());
        let scenario = Box::new(DeterministicScenario::new(0)); // No delay for testing
        let config = BenchmarkConfig {
            num_requests: 5,
            model: "test-model".to_string(),
            max_tokens: None,
            temperature: None,
            stream: false,
        };

        let runner = BenchmarkRunner::new(provider, sampler, scenario, config);
        let collector = runner.run_sequential().await.unwrap();

        assert_eq!(collector.len(), 5);
        let agg = collector.aggregate();
        assert_eq!(agg.successful_requests, 5);
    }
}
