//! Request scenarios for controlling timing and patterns
//!
//! Scenarios define how requests are distributed over time:
//! - **Deterministic**: Fixed delay between requests
//! - **Normal**: Delays sampled from normal distribution
//! - **Uniform**: Delays sampled from uniform distribution

use anyhow::Result;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Duration;

/// Trait for different request scenarios
pub trait Scenario: Send + Sync {
    /// Get the delay before the next request
    fn next_delay(&mut self) -> Duration;

    /// Get the scenario name
    fn name(&self) -> &str;
}

/// Deterministic scenario - fixed delay between requests
pub struct DeterministicScenario {
    delay_ms: u64,
}

impl DeterministicScenario {
    /// Create a new deterministic scenario
    pub fn new(delay_ms: u64) -> Self {
        Self { delay_ms }
    }
}

impl Scenario for DeterministicScenario {
    fn next_delay(&mut self) -> Duration {
        Duration::from_millis(self.delay_ms)
    }

    fn name(&self) -> &str {
        "deterministic"
    }
}

/// Normal distribution scenario
pub struct NormalScenario {
    distribution: Normal<f64>,
}

impl NormalScenario {
    /// Create a new normal scenario with mean and standard deviation in milliseconds
    pub fn new(mean_ms: f64, std_dev_ms: f64) -> Result<Self> {
        let distribution = Normal::new(mean_ms, std_dev_ms)
            .map_err(|e| anyhow::anyhow!("Invalid normal distribution parameters: {}", e))?;
        Ok(Self { distribution })
    }
}

impl Scenario for NormalScenario {
    fn next_delay(&mut self) -> Duration {
        let mut rng = rand::thread_rng();
        let delay_ms = self.distribution.sample(&mut rng).max(0.0) as u64;
        Duration::from_millis(delay_ms)
    }

    fn name(&self) -> &str {
        "normal"
    }
}

/// Uniform distribution scenario
pub struct UniformScenario {
    distribution: Uniform<u64>,
}

impl UniformScenario {
    /// Create a new uniform scenario with min and max delays in milliseconds
    pub fn new(min_ms: u64, max_ms: u64) -> Self {
        Self {
            distribution: Uniform::new(min_ms, max_ms),
        }
    }
}

impl Scenario for UniformScenario {
    fn next_delay(&mut self) -> Duration {
        let mut rng = rand::thread_rng();
        let delay_ms = self.distribution.sample(&mut rng);
        Duration::from_millis(delay_ms)
    }

    fn name(&self) -> &str {
        "uniform"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_scenario() {
        let mut scenario = DeterministicScenario::new(100);
        assert_eq!(scenario.name(), "deterministic");

        let delay = scenario.next_delay();
        assert_eq!(delay, Duration::from_millis(100));
    }

    #[test]
    fn test_normal_scenario() {
        let mut scenario = NormalScenario::new(100.0, 10.0).unwrap();
        assert_eq!(scenario.name(), "normal");

        // Just verify we can get delays without panicking
        for _ in 0..10 {
            let _delay = scenario.next_delay();
        }
    }

    #[test]
    fn test_uniform_scenario() {
        let mut scenario = UniformScenario::new(50, 150);
        assert_eq!(scenario.name(), "uniform");

        // Verify delays are within range
        for _ in 0..10 {
            let delay = scenario.next_delay();
            assert!(delay >= Duration::from_millis(50));
            assert!(delay < Duration::from_millis(150));
        }
    }
}
