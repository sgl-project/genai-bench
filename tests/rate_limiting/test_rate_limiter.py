"""Tests for TokenBucketRateLimiter."""

import time

import gevent
import pytest

from genai_bench.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Test suite for TokenBucketRateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initializes with correct parameters."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        assert rate_limiter.rate == 10.0
        assert rate_limiter.tokens == 1  # Starts with 1 token to avoid burst

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
        assert rate_limiter.tokens < rate_limiter.bucket_size  # Token consumed

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
        assert 0.3 <= elapsed <= 0.5

    def test_token_refill(self):
        """Test that tokens refill over time."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Consume the only token
        rate_limiter.acquire()

        # Wait for token to refill (0.1s should give 1 token at 10/s)
        gevent.sleep(0.1)

        # Should be able to acquire again
        start_time = time.monotonic()
        result = rate_limiter.acquire()
        elapsed = time.monotonic() - start_time
        assert result is True
        assert elapsed < 0.05  # Should be immediate

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
        assert abs(actual_rate - target_rate) < target_rate * 0.1

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        rate_limiter = TokenBucketRateLimiter(rate=20.0)
        acquired = []

        def acquire_token():
            result = rate_limiter.acquire()
            acquired.append(result)

        # Spawn multiple greenlets
        start_time = time.monotonic()
        greenlets = [gevent.spawn(acquire_token) for _ in range(20)]
        gevent.joinall(greenlets)
        elapsed = time.monotonic() - start_time

        # All should succeed
        assert len(acquired) == 20
        assert all(acquired)

        # Should take approximately 1 second (20 tokens at 20/s)
        assert 0.9 <= elapsed <= 1.1

    def test_get_current_rate(self):
        """Test getting current rate."""
        rate_limiter = TokenBucketRateLimiter(rate=15.0)
        assert rate_limiter.get_current_rate() == 15.0

    def test_get_available_tokens(self):
        """Test getting available token count."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Should start with 1 token
        available = rate_limiter.get_available_tokens()
        assert available == 1

        # Consume the token
        rate_limiter.acquire()

        available = rate_limiter.get_available_tokens()
        assert available == 0

    def test_reset(self):
        """Test resetting the rate limiter."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Consume the token
        rate_limiter.acquire()

        # Reset
        rate_limiter.reset()

        # Should be full again
        available = rate_limiter.get_available_tokens()
        assert available == rate_limiter.bucket_size

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

    def test_stop_wakes_up_waiting_greenlets(self):
        """Test that stop() method wakes up waiting greenlets."""
        rate_limiter = TokenBucketRateLimiter(rate=1.0)

        # Consume the only token
        rate_limiter.acquire()

        # Spawn a greenlet that will wait for a token
        results = []

        def acquire_token():
            result = rate_limiter.acquire()
            results.append(result)

        greenlet = gevent.spawn(acquire_token)

        # Give it a moment to start waiting
        gevent.sleep(0.05)

        # Stop the rate limiter - should wake up the waiting greenlet
        rate_limiter.stop()

        # Wait for greenlet to finish
        greenlet.join()

        # Should have returned False
        assert len(results) == 1
        assert results[0] is False

    def test_acquire_returns_false_when_stopped(self):
        """Test that acquire() returns False immediately when stopped."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Stop the rate limiter
        rate_limiter.stop()

        # Try to acquire - should return False immediately
        result = rate_limiter.acquire()
        assert result is False

        # Even if tokens are available, should still return False
        rate_limiter.reset()  # This doesn't reset stopped state
        result = rate_limiter.acquire()
        assert result is False

    def test_concurrent_acquire_with_stop(self):
        """Test concurrent acquire operations with stop signal."""
        rate_limiter = TokenBucketRateLimiter(rate=1.0)

        # Consume the only token
        rate_limiter.acquire()

        results = []

        def acquire_token():
            result = rate_limiter.acquire()
            results.append(result)

        # Spawn multiple greenlets that will wait
        greenlets = [gevent.spawn(acquire_token) for _ in range(5)]

        # Give them a moment to start waiting
        gevent.sleep(0.05)

        # Stop the rate limiter
        rate_limiter.stop()

        # Wait for all greenlets to finish
        gevent.joinall(greenlets)

        # All should have returned False
        assert len(results) == 5
        assert all(r is False for r in results)

    def test_reset_doesnt_reset_stopped_state(self):
        """Test that reset() doesn't reset the stopped state."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Consume the initial token
        rate_limiter.acquire()
        assert rate_limiter.tokens == 0

        # Stop the rate limiter
        rate_limiter.stop()
        assert rate_limiter.stopped is True

        # Reset should refill tokens but not reset stopped state
        rate_limiter.reset()
        assert rate_limiter.tokens == rate_limiter.bucket_size
        assert rate_limiter.stopped is True  # Still stopped

        # Should still return False on acquire
        result = rate_limiter.acquire()
        assert result is False

    def test_stop_after_acquire(self):
        """Test that stop() works correctly after tokens have been acquired."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)

        # Acquire a token successfully
        result = rate_limiter.acquire()
        assert result is True

        # Stop the rate limiter
        rate_limiter.stop()

        # Future acquires should return False
        result = rate_limiter.acquire()
        assert result is False

    def test_acquire_with_timeout_after_stop(self):
        """Test acquire with timeout when rate limiter is stopped."""
        rate_limiter = TokenBucketRateLimiter(rate=1.0)

        # Consume the only token
        rate_limiter.acquire()

        # Stop the rate limiter
        rate_limiter.stop()

        # Try to acquire with timeout - should return False immediately
        start_time = time.monotonic()
        result = rate_limiter.acquire(timeout=5.0)
        elapsed = time.monotonic() - start_time

        assert result is False
        # Should return immediately, not wait for timeout
        assert elapsed < 0.1
