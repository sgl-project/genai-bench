//! Rate limiting for request execution

use governor::{clock::DefaultClock, state::InMemoryState, state::NotKeyed, Quota, RateLimiter};
use std::num::NonZeroU32;

/// Rate limiter using token bucket algorithm via governor crate
///
/// This provides per-worker rate limiting. For global rate limiting
/// across all workers, share a single instance via Arc.
pub struct RequestRateLimiter {
    limiter: Option<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
    rate_limit: Option<f64>,
}

impl RequestRateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `rate_limit` - Optional requests per second limit. None disables rate limiting.
    ///
    /// # Examples
    /// ```
    /// use genai_bench_core::worker::RequestRateLimiter;
    ///
    /// // Create rate limiter for 100 requests per second
    /// let limiter = RequestRateLimiter::new(Some(100.0));
    ///
    /// // Create unlimited rate limiter
    /// let unlimited = RequestRateLimiter::new(None);
    /// ```
    pub fn new(rate_limit: Option<f64>) -> Self {
        let limiter = rate_limit.and_then(|rps| {
            if rps <= 0.0 {
                return None;
            }
            // Convert f64 RPS to integer quota (minimum 1).
            // Note: Sub-1 RPS values (e.g., 0.5) are rounded up to 1 RPS.
            // For very low rates, consider using longer intervals or external throttling.
            let rps_int = (rps.ceil() as u32).max(1);
            let quota = Quota::per_second(NonZeroU32::new(rps_int)?);
            Some(RateLimiter::direct(quota))
        });

        Self {
            limiter,
            rate_limit,
        }
    }

    /// Create an unlimited rate limiter (no rate limiting)
    pub fn unlimited() -> Self {
        Self::new(None)
    }

    /// Wait until a request is allowed
    ///
    /// Returns immediately if no rate limit is configured.
    /// Otherwise, waits until the rate limiter allows the request.
    pub async fn wait(&self) {
        if let Some(ref limiter) = self.limiter {
            limiter.until_ready().await;
        }
    }

    /// Try to acquire a permit without waiting
    ///
    /// Returns `true` if a request is allowed immediately, `false` otherwise.
    /// Always returns `true` if no rate limit is configured.
    pub fn try_acquire(&self) -> bool {
        match &self.limiter {
            Some(limiter) => limiter.check().is_ok(),
            None => true,
        }
    }

    /// Check if rate limiting is enabled
    pub fn is_enabled(&self) -> bool {
        self.limiter.is_some()
    }

    /// Get the configured rate limit (requests per second)
    pub fn rate_limit(&self) -> Option<f64> {
        self.rate_limit
    }
}

impl Default for RequestRateLimiter {
    fn default() -> Self {
        Self::new(None)
    }
}

impl std::fmt::Debug for RequestRateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestRateLimiter")
            .field("rate_limit", &self.rate_limit)
            .field("enabled", &self.is_enabled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_disabled() {
        let limiter = RequestRateLimiter::new(None);
        assert!(!limiter.is_enabled());
        assert!(limiter.rate_limit().is_none());
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_rate_limiter_zero_rps() {
        let limiter = RequestRateLimiter::new(Some(0.0));
        assert!(!limiter.is_enabled());
    }

    #[test]
    fn test_rate_limiter_negative_rps() {
        let limiter = RequestRateLimiter::new(Some(-10.0));
        assert!(!limiter.is_enabled());
    }

    #[test]
    fn test_rate_limiter_enabled() {
        let limiter = RequestRateLimiter::new(Some(100.0));
        assert!(limiter.is_enabled());
        assert_eq!(limiter.rate_limit(), Some(100.0));
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RequestRateLimiter::unlimited();
        assert!(!limiter.is_enabled());
    }

    #[test]
    fn test_rate_limiter_default() {
        let limiter = RequestRateLimiter::default();
        assert!(!limiter.is_enabled());
    }

    #[tokio::test]
    async fn test_rate_limiter_wait_disabled() {
        let limiter = RequestRateLimiter::new(None);
        // Should return immediately
        limiter.wait().await;
    }

    #[tokio::test]
    async fn test_rate_limiter_wait_enabled() {
        let limiter = RequestRateLimiter::new(Some(1000.0));
        // Should allow at least one request immediately
        limiter.wait().await;
    }

    #[test]
    fn test_rate_limiter_try_acquire() {
        let limiter = RequestRateLimiter::new(Some(1000.0));
        // Should allow at least one request immediately
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_rate_limiter_debug() {
        let limiter = RequestRateLimiter::new(Some(100.0));
        let debug = format!("{:?}", limiter);
        assert!(debug.contains("RequestRateLimiter"));
        assert!(debug.contains("100.0"));
        assert!(debug.contains("true"));
    }
}
