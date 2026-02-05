"""Token Bucket Rate Limiter for precise request rate control."""

import time
from typing import Optional

from gevent.event import Event
from gevent.lock import Semaphore

from genai_bench.logging import init_logger

logger = init_logger(__name__)

MAX_BUCKET_SIZE = 100


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
        tokens: Current number of tokens in the bucket (always an integer)
        last_update: Timestamp of last token refill
        lock: Lock for thread-safe token operations
        stopped: Flag indicating if the rate limiter has been stopped
        _stop_event: Event to signal waiting greenlets to stop
    """

    def __init__(self, rate: float):
        """
        Initialize the token bucket rate limiter.

        Args:
            rate: Target request rate in requests per second
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        self.rate = rate
        self.bucket_size = max(1, min(int(rate), MAX_BUCKET_SIZE))
        self.tokens: int = 1
        self.last_update = time.monotonic()
        self.lock = Semaphore(value=1)  # Gevent-compatible lock
        self.stopped = False
        self._stop_event = Event()  # Event to signal stop

        logger.info(f"Token Bucket Rate Limiter initialized: " f"rate={rate:.2f} req/s")

    def _refill_tokens(self) -> None:
        """
        Refill tokens based on time elapsed since the last update.

        This method is called internally before acquiring a token.
        """
        now = time.monotonic()
        time_passed = now - self.last_update

        # Calculate how many tokens should have been generated
        tokens_to_add: float = time_passed * self.rate

        if tokens_to_add >= 1.0:
            tokens_added = int(tokens_to_add)
            self.tokens = min(self.bucket_size, self.tokens + tokens_added)

            time_used = tokens_added / self.rate
            self.last_update = self.last_update + time_used

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token to make a request.

        Blocks until a token is available or timeout expires.
        Returns False immediately if the rate limiter has been stopped.

        Args:
            timeout: Maximum time to wait for a token in seconds.
                    None means wait indefinitely.

        Returns:
            True if token acquired, False if timeout expired or rate limiter stopped
        """
        start_time = time.monotonic()

        while True:
            with self.lock:
                # Check if rate limiter has been stopped
                if self.stopped:
                    return False

                self._refill_tokens()

                if self.tokens > 0:
                    # Token available, consume it
                    self.tokens -= 1
                    return True

                wait_time = 1.0 / self.rate

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
                # Adjust wait time to not exceed timeout
                wait_time = min(wait_time, timeout - elapsed)

            # Wait for token or stop signal
            # Use _stop_event.wait() to allow interruption when stopped
            if self._stop_event.wait(timeout=wait_time):
                # Stop event was set, return False
                return False

            # Continue loop to check for token availability

    def get_current_rate(self) -> float:
        """
        Get the current token bucket fill rate.

        Returns:
            Current rate in tokens per second (same as configured rate)
        """
        return self.rate

    def get_available_tokens(self) -> int:
        """
        Get the number of tokens currently available.

        Returns:
            Number of tokens available for immediate use (always an integer)
        """
        with self.lock:
            self._refill_tokens()
            return int(self.tokens)  # Explicitly cast to int for clarity

    def stop(self) -> None:
        """
        Stop the rate limiter and wake up any waiting greenlets.

        This should be called before stopping a run to ensure all pending
        token acquisitions are cleaned up. After calling stop(), acquire()
        will return False immediately for any new or waiting calls.
        """
        with self.lock:
            self.stopped = True
        # Set the event to wake up any greenlets waiting in acquire()
        self._stop_event.set()
        logger.debug("Rate limiter stopped, waking up waiting greenlets")

    def reset(self) -> None:
        """
        Reset the rate limiter to initial state.

        Refills bucket to maximum size and resets timestamp.
        Note: This does not reset the stopped state. Use a new instance
        if you need to restart after stopping.
        """
        with self.lock:
            self.tokens = self.bucket_size
            self.last_update = time.monotonic()
            logger.debug("Token bucket reset to full")
