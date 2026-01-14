//! Builder pattern for Orchestrator construction

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::channel::ChannelConfig;
use crate::config::ExperimentConfig;
use crate::error::{BenchError, BenchResult};
use crate::metrics::RequestRecord;
use crate::traits::{Sampler, StopCondition, VendorClient};

use super::executor::Orchestrator;

/// Builder for creating an Orchestrator with proper configuration
///
/// # Example
///
/// ```ignore
/// let (orchestrator, metrics_rx) = OrchestratorBuilder::new()
///     .concurrency(10)
///     .stop_condition(StopCondition::RequestCount(1000))
///     .rate_limit(Some(100.0))
///     .vendor(vendor)
///     .sampler(sampler)
///     .build()?;
/// ```
pub struct OrchestratorBuilder {
    config: ExperimentConfig,
    vendor: Option<Arc<dyn VendorClient>>,
    sampler: Option<Arc<dyn Sampler>>,
    channel_config: ChannelConfig,
}

impl OrchestratorBuilder {
    /// Create a new orchestrator builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ExperimentConfig::default(),
            vendor: None,
            sampler: None,
            channel_config: ChannelConfig::default(),
        }
    }

    /// Set the full experiment configuration
    pub fn config(mut self, config: ExperimentConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the concurrency level
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.config.concurrency = concurrency;
        self
    }

    /// Set the stop condition
    pub fn stop_condition(mut self, stop: StopCondition) -> Self {
        self.config.stop_condition = stop;
        self
    }

    /// Set the rate limit (requests per second)
    pub fn rate_limit(mut self, rps: Option<f64>) -> Self {
        self.config.rate_limit = rps;
        self
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

    /// Set the channel configuration
    pub fn channel_config(mut self, config: ChannelConfig) -> Self {
        self.channel_config = config;
        self
    }

    /// Build the orchestrator and return it along with the metrics receiver
    ///
    /// # Errors
    ///
    /// Returns an error if vendor or sampler are not set, or if configuration
    /// validation fails.
    pub fn build(self) -> BenchResult<(Orchestrator, mpsc::Receiver<RequestRecord>)> {
        let vendor = self
            .vendor
            .ok_or_else(|| BenchError::missing_config("vendor"))?;

        let sampler = self
            .sampler
            .ok_or_else(|| BenchError::missing_config("sampler"))?;

        self.config
            .validate()
            .map_err(|e| BenchError::config(e.to_string()))?;

        let (metrics_tx, metrics_rx) = mpsc::channel(self.channel_config.metrics_buffer);

        let orchestrator = Orchestrator::new(self.config, vendor, sampler, metrics_tx);

        Ok((orchestrator, metrics_rx))
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
