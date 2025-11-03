"""Token Bucket Rate Limiter for precise request rate control."""

import time
from typing import Optional

import gevent
from gevent.lock import Semaphore

from genai_bench.logging import init_logger

logger = init_logger(__name__)


class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter for precise request rate control.

    This implements a token bucket algorithm where:
    - Tokens are added to the bucket at a constant rate (target_rate)
    - Each request consumes one token
    - If no tokens available, the request waits until a token becomes available

    This provides exact rate control regardless of response latency or
    concurrency level.

    Attributes:
        rate: Target request rate in requests per second
        bucket_size: Maximum number of tokens that can accumulate
        tokens: Current number of tokens in the bucket
        last_update: Timestamp of last token refill
        lock: Lock for thread-safe token operations
    """

    def __init__(self, rate: float, bucket_size: Optional[float] = None):
        """
        Initialize the token bucket rate limiter.

        Args:
            rate: Target request rate in requests per second
            bucket_size: Maximum tokens in bucket. Defaults to rate * 2
                        (allows brief bursts up to 2 seconds worth)
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        self.rate = rate
        self.bucket_size = bucket_size if bucket_size is not None else rate * 2
        self.tokens = self.bucket_size  # Start with full bucket
        self.last_update = time.monotonic()
        self.lock = Semaphore(value=1)  # Gevent-compatible lock

        logger.info(
            f"🪣 Token Bucket Rate Limiter initialized: "
            f"rate={rate:.2f} req/s, bucket_size={self.bucket_size:.1f}"
        )

    def _refill_tokens(self) -> None:
        """
        Refill tokens based on time elapsed since last update.

        Called internally before acquiring a token.
        """
        now = time.monotonic()
        time_passed = now - self.last_update

        # Calculate tokens to add based on elapsed time
        tokens_to_add = time_passed * self.rate

        # Add tokens but don't exceed bucket size
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token to make a request.

        Blocks until a token is available or timeout expires.

        Args:
            timeout: Maximum time to wait for a token in seconds.
                    None means wait indefinitely.

        Returns:
            True if token acquired, False if timeout expired

        Raises:
            TimeoutError: If timeout expires (when timeout is not None)
        """
        start_time = time.monotonic()

        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= 1.0:
                    # Token available, consume it
                    self.tokens -= 1.0
                    return True

                # Calculate how long until next token is available
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / self.rate

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
                # Adjust wait time to not exceed timeout
                wait_time = min(wait_time, timeout - elapsed)

            # Sleep until next token should be available
            # Use gevent.sleep for cooperative multitasking
            gevent.sleep(wait_time)

    def get_current_rate(self) -> float:
        """
        Get the current token bucket fill rate.

        Returns:
            Current rate in tokens per second (same as configured rate)
        """
        return self.rate

    def get_available_tokens(self) -> float:
        """
        Get the number of tokens currently available.

        Returns:
            Number of tokens available for immediate use
        """
        with self.lock:
            self._refill_tokens()
            return self.tokens

    def reset(self) -> None:
        """
        Reset the rate limiter to initial state.

        Refills bucket to maximum size and resets timestamp.
        """
        with self.lock:
            self.tokens = self.bucket_size
            self.last_update = time.monotonic()
            logger.debug("Token bucket reset to full")
