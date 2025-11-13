"""Tests for TokenBucketRateLimiter."""

import time

import gevent
import pytest

from genai_bench.rate_limiter import BUCKET_SIZE, TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Test suite for TokenBucketRateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initializes with correct parameters."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        assert rate_limiter.rate == 10.0
        assert rate_limiter.tokens == BUCKET_SIZE  # Starts full

    def test_initialization_bucket_size(self):
        """Test rate limiter uses fixed bucket size."""
        rate_limiter = TokenBucketRateLimiter(rate=5.0)

        assert rate_limiter.rate == 5.0
        assert rate_limiter.tokens == BUCKET_SIZE

    def test_initialization_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="Rate must be positive"):
            TokenBucketRateLimiter(rate=0)

        with pytest.raises(ValueError, match="Rate must be positive"):
            TokenBucketRateLimiter(rate=-5.0)

    def test_acquire_with_available_tokens(self):
        """Test acquiring tokens when available."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Should succeed immediately
        result = rate_limiter.acquire()
        assert result is True
        assert rate_limiter.tokens < BUCKET_SIZE  # Token consumed

    def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens in succession."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Acquire first token immediately
        result = rate_limiter.acquire()
        assert result is True

        # Subsequent tokens require waiting
        # Acquire 4 more tokens (should take ~0.4s at 10/s)
        start_time = time.monotonic()
        for _ in range(4):
            result = rate_limiter.acquire()
            assert result is True
        elapsed = time.monotonic() - start_time

        # Should take approximately 0.4s (4 tokens / 10 tokens per second)
        # Allow some tolerance
        assert 0.3 <= elapsed <= 0.6

    def test_token_refill(self):
        """Test that tokens refill over time."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Consume the only token
        rate_limiter.acquire()

        # Wait for token to refill (0.1s should give 1 token at 10/s)
        gevent.sleep(0.1)

        # Should be able to acquire again
        result = rate_limiter.acquire()
        assert result is True

    def test_rate_limiting_accuracy(self):
        """Test that rate limiting is accurate."""
        target_rate = 10.0
        rate_limiter = TokenBucketRateLimiter(rate=target_rate)

        # Consume the initial token
        rate_limiter.acquire()

        # Now measure sustained rate (should be limited to target_rate)
        num_requests = 20
        start_time = time.monotonic()

        for _ in range(num_requests):
            rate_limiter.acquire()

        elapsed_time = time.monotonic() - start_time
        actual_rate = num_requests / elapsed_time

        # Should take ~2 seconds for 20 requests at 10/s
        # Allow 20% tolerance for timing variations and gevent overhead
        assert abs(actual_rate - target_rate) < target_rate * 0.2

    def test_burst_handling(self):
        """Test that bucket allows single immediate request."""
        rate_limiter = TokenBucketRateLimiter(rate=5.0)

        # Should be able to acquire first request immediately
        start_time = time.monotonic()
        result = rate_limiter.acquire()
        burst_time = time.monotonic() - start_time

        assert result is True
        # First request should be very fast (< 0.1s)
        assert burst_time < 0.1

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        rate_limiter = TokenBucketRateLimiter(rate=20.0)
        acquired = []

        def acquire_token():
            result = rate_limiter.acquire()
            acquired.append(result)

        # Spawn multiple greenlets
        greenlets = [gevent.spawn(acquire_token) for _ in range(10)]
        gevent.joinall(greenlets)

        # All should succeed
        assert len(acquired) == 10
        assert all(acquired)

    def test_get_current_rate(self):
        """Test getting current rate."""
        rate_limiter = TokenBucketRateLimiter(rate=15.0)
        assert rate_limiter.get_current_rate() == 15.0

    def test_get_available_tokens(self):
        """Test getting available token count."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Should start with full bucket
        available = rate_limiter.get_available_tokens()
        assert available == BUCKET_SIZE

        # Consume the token
        rate_limiter.acquire()

        available = rate_limiter.get_available_tokens()
        assert available < BUCKET_SIZE

    def test_reset(self):
        """Test resetting the rate limiter."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Consume the token
        rate_limiter.acquire()

        # Reset
        rate_limiter.reset()

        # Should be full again
        available = rate_limiter.get_available_tokens()
        assert available == BUCKET_SIZE

    def test_acquire_timeout_false(self):
        """Test acquire returns False when timeout expires."""
        rate_limiter = TokenBucketRateLimiter(rate=1.0)

        # Consume the only token
        rate_limiter.acquire()

        # Try to acquire with short timeout (should fail)
        result = rate_limiter.acquire(timeout=0.1)
        assert result is False

    def test_fractional_rates(self):
        """Test rate limiter works with fractional rates."""
        rate_limiter = TokenBucketRateLimiter(rate=2.5)

        # Should be able to acquire tokens (will wait between acquisitions)
        for _ in range(5):
            result = rate_limiter.acquire()
            assert result is True

    def test_high_rate(self):
        """Test rate limiter works with high rates."""
        rate_limiter = TokenBucketRateLimiter(rate=100.0)

        # Consume initial token
        rate_limiter.acquire()

        # Should handle high sustained rate
        start_time = time.monotonic()
        for _ in range(50):
            rate_limiter.acquire()
        elapsed = time.monotonic() - start_time

        # Should complete in roughly 0.5s (50 requests at 100/s)
        # Allow some overhead
        assert elapsed < 0.7

    def test_low_rate(self):
        """Test rate limiter works with very low rates."""
        rate_limiter = TokenBucketRateLimiter(rate=2.0)  # 2 req/s

        # First request consumes the only token
        result = rate_limiter.acquire()
        assert result is True

        # Second request should take ~0.5 seconds (1 token / 2 tokens per second)
        start_time = time.monotonic()
        result = rate_limiter.acquire()
        elapsed = time.monotonic() - start_time

        assert result is True
        # Should take at least 0.4s, at most 0.7s (with overhead)
        assert 0.4 <= elapsed <= 0.7
