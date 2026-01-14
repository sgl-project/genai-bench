//! Experiment configuration types

use crate::traits::StopCondition;
use serde::{Deserialize, Serialize};

/// Experiment configuration
///
/// Defines how a benchmarking experiment should be run, including
/// concurrency level, stopping conditions, and rate limiting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Number of concurrent worker tasks
    pub concurrency: usize,

    /// Stop condition: request count, duration, or indefinite
    pub stop_condition: StopCondition,

    /// Optional rate limiting (requests per second)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<f64>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            stop_condition: StopCondition::default(),
            rate_limit: None,
        }
    }
}

impl ExperimentConfig {
    /// Create a new config with the given concurrency
    pub fn new(concurrency: usize) -> Self {
        Self {
            concurrency,
            ..Default::default()
        }
    }

    /// Set the stop condition
    pub fn with_stop_condition(mut self, stop: StopCondition) -> Self {
        self.stop_condition = stop;
        self
    }

    /// Set the rate limit
    pub fn with_rate_limit(mut self, rps: f64) -> Self {
        self.rate_limit = Some(rps);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.concurrency == 0 {
            return Err(ConfigError::InvalidConcurrency(
                "concurrency must be at least 1".into(),
            ));
        }

        if let Some(rps) = self.rate_limit {
            if rps <= 0.0 {
                return Err(ConfigError::InvalidRateLimit(
                    "rate limit must be positive".into(),
                ));
            }
        }

        if let StopCondition::RequestCount(n) = self.stop_condition {
            if n == 0 {
                return Err(ConfigError::InvalidStopCondition(
                    "request count must be at least 1".into(),
                ));
            }
        }

        Ok(())
    }
}

/// Configuration validation errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid concurrency value
    #[error("Invalid concurrency: {0}")]
    InvalidConcurrency(String),

    /// Invalid rate limit
    #[error("Invalid rate limit: {0}")]
    InvalidRateLimit(String),

    /// Invalid stop condition
    #[error("Invalid stop condition: {0}")]
    InvalidStopCondition(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = ExperimentConfig::default();
        assert_eq!(config.concurrency, 1);
        assert!(matches!(
            config.stop_condition,
            StopCondition::RequestCount(100)
        ));
        assert!(config.rate_limit.is_none());
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = ExperimentConfig::new(10)
            .with_stop_condition(StopCondition::Duration(Duration::from_secs(60)))
            .with_rate_limit(100.0);

        assert_eq!(config.concurrency, 10);
        assert!(matches!(config.stop_condition, StopCondition::Duration(_)));
        assert_eq!(config.rate_limit, Some(100.0));
    }

    #[test]
    fn test_config_validation_valid() {
        let config = ExperimentConfig::new(10)
            .with_stop_condition(StopCondition::RequestCount(100))
            .with_rate_limit(50.0);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_concurrency() {
        let config = ExperimentConfig {
            concurrency: 0,
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_negative_rate_limit() {
        let config = ExperimentConfig::new(1).with_rate_limit(-10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_request_count() {
        let config = ExperimentConfig::new(1).with_stop_condition(StopCondition::RequestCount(0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config =
            ExperimentConfig::new(5).with_stop_condition(StopCondition::RequestCount(1000));

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ExperimentConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.concurrency, 5);
    }
}
