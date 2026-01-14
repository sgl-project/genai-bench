//! Channel configuration for orchestrator communication

/// Channel buffer configuration for orchestrator communication
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Metrics channel buffer size (workers -> aggregator)
    pub metrics_buffer: usize,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            metrics_buffer: 10_000,
        }
    }
}

impl ChannelConfig {
    /// Create a new channel config with custom metrics buffer size
    pub fn with_metrics_buffer(mut self, size: usize) -> Self {
        self.metrics_buffer = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_config_default() {
        let config = ChannelConfig::default();
        assert_eq!(config.metrics_buffer, 10_000);
    }

    #[test]
    fn test_channel_config_builder() {
        let config = ChannelConfig::default().with_metrics_buffer(5000);
        assert_eq!(config.metrics_buffer, 5000);
    }
}
