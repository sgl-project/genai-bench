"""Tests for ClosedLoopRunner (concurrency-based execution)."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_bench.protocol import UserChatRequest


@pytest.fixture
def mock_sampler():
    """Mock sampler for testing."""
    sampler = MagicMock()
    sampler.sample = MagicMock(
        return_value=UserChatRequest(
            model="gpt-4",
            prompt="Hello",
            num_prefill_tokens=10,
            max_tokens=20,
        )
    )
    sampler.get_token_length = MagicMock(return_value=5)
    return sampler


@pytest.fixture
def mock_auth_provider():
    """Mock auth provider for testing."""
    auth = MagicMock()
    auth.get_headers = MagicMock(
        return_value={
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
        }
    )
    return auth


@pytest.fixture
def mock_aggregated_metrics():
    """Mock aggregated metrics collector."""
    metrics = MagicMock()
    metrics.aggregated_metrics = MagicMock()
    metrics.aggregated_metrics.num_completed_requests = 0
    metrics.aggregated_metrics.num_error_requests = 0
    metrics.aggregated_metrics.total_arrivals = None
    metrics.add_single_request_metrics = MagicMock()
    metrics.get_live_metrics = MagicMock(return_value={})
    metrics.clear = MagicMock()
    return metrics


@pytest.fixture
def closed_loop_runner(mock_sampler, mock_auth_provider, mock_aggregated_metrics):
    """Create ClosedLoopRunner instance for testing."""
    from genai_bench.async_runner.closed_loop import ClosedLoopRunner

    return ClosedLoopRunner(
        sampler=mock_sampler,
        api_backend="openai",
        api_base="https://api.openai.com/v1",
        api_model_name="gpt-4",
        auth_provider=mock_auth_provider,
        aggregated_metrics_collector=mock_aggregated_metrics,
        dashboard=None,
    )


class TestClosedLoopRunner:
    """Test ClosedLoopRunner concurrency-based execution."""

    def test_run_requires_target_concurrency(self, closed_loop_runner):
        """Test that run requires target_concurrency for closed-loop mode."""
        with pytest.raises(ValueError, match="target_concurrency is required"):
            closed_loop_runner.run(
                target_concurrency=None,
                duration_s=1,
                distribution="exponential",  # Not used in closed-loop, but required
                random_seed=42,
                max_requests=None,
                max_time_s=None,
                scenario="D(100,100)",
            )

    def test_run_closed_loop_execution(
        self, closed_loop_runner, mock_aggregated_metrics
    ):
        """Test closed-loop execution maintaining target concurrency."""
        # This test verifies the basic structure works
        closed_loop_runner._send_one = AsyncMock(return_value=None)

        # Use a very small max_requests and max_time_s to ensure it completes quickly
        # Mock sleep to yield control but not block - this is critical for proper async behavior
        # We need to use the real asyncio.sleep(0) to yield, not the patched version
        real_sleep = asyncio.sleep

        async def yield_sleep(delay):
            # Yield control to let other tasks run, but don't actually sleep
            # Use real_sleep to avoid recursion
            await real_sleep(0)  # This yields to the event loop

        with patch(
            "genai_bench.async_runner.closed_loop.asyncio.sleep",
            side_effect=yield_sleep,
        ):
            # Use both max_requests and max_time_s as safety limits
            closed_loop_runner.run(
                target_concurrency=1,  # Single concurrency to simplify
                duration_s=1,
                distribution="exponential",
                random_seed=42,
                max_requests=2,  # Very small number
                max_time_s=1,  # 1 second timeout as safety
                scenario="D(100,100)",
            )

        # Verify total_arrivals was set (should be exactly 2 due to max_requests)
        assert mock_aggregated_metrics.aggregated_metrics.total_arrivals is not None
        assert mock_aggregated_metrics.aggregated_metrics.total_arrivals == 2

    def test_run_maintains_concurrency(self, closed_loop_runner):
        """Test that closed-loop maintains target concurrency."""
        # This test verifies the structure works, actual concurrency testing
        # would require more complex async orchestration
        closed_loop_runner._send_one = AsyncMock(return_value=None)

        # Mock sleep to yield control but not block
        real_sleep = asyncio.sleep

        async def yield_sleep(delay):
            await real_sleep(0)  # Yield to event loop

        # Test that run() method works with small max_requests
        with patch(
            "genai_bench.async_runner.closed_loop.asyncio.sleep",
            side_effect=yield_sleep,
        ):
            closed_loop_runner.run(
                target_concurrency=3,
                duration_s=10,
                distribution="exponential",
                random_seed=42,
                max_requests=3,  # Small number to complete quickly
                max_time_s=None,
                scenario="D(100,100)",
            )

    def test_run_respects_max_requests(
        self, closed_loop_runner, mock_aggregated_metrics
    ):
        """Test that closed-loop respects max_requests limit."""
        closed_loop_runner._send_one = AsyncMock(return_value=None)

        # Mock sleep to yield control but not block
        real_sleep = asyncio.sleep

        async def yield_sleep(delay):
            await real_sleep(0)  # Yield to event loop

        # Use a small max_requests to control test execution
        with patch(
            "genai_bench.async_runner.closed_loop.asyncio.sleep",
            side_effect=yield_sleep,
        ):
            closed_loop_runner.run(
                target_concurrency=2,
                duration_s=10,
                distribution="exponential",
                random_seed=42,
                max_requests=3,  # Limit to 3 requests
                max_time_s=None,
                scenario="D(100,100)",
            )

        # Should stop after max_requests
        assert mock_aggregated_metrics.aggregated_metrics.total_arrivals == 3

    def test_run_respects_max_time(self, closed_loop_runner):
        """Test that closed-loop respects max_time_s timeout."""
        closed_loop_runner._send_one = AsyncMock(return_value=None)

        # Mock sleep to yield control but not block
        import asyncio as asyncio_module

        real_sleep = asyncio_module.sleep

        async def yield_sleep(delay):
            await real_sleep(0)  # Yield to event loop

        # Mock asyncio.wait_for to raise TimeoutError
        with patch(
            "genai_bench.async_runner.closed_loop.asyncio.sleep",
            side_effect=yield_sleep,
        ):
            with patch(
                "genai_bench.async_runner.closed_loop.asyncio.wait_for"
            ) as mock_wait:
                mock_wait.side_effect = asyncio_module.TimeoutError()
                with contextlib.suppress(asyncio_module.TimeoutError):
                    closed_loop_runner.run(
                        target_concurrency=2,
                        duration_s=10,
                        distribution="exponential",
                        random_seed=42,
                        max_requests=None,
                        max_time_s=0.1,  # 100ms timeout
                        scenario="D(100,100)",
                    )
