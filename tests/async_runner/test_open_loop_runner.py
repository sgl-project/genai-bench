"""Tests for OpenLoopRunner (QPS-based execution in async runner)."""

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
def open_loop_runner(mock_sampler, mock_auth_provider, mock_aggregated_metrics):
    """Create OpenLoopRunner instance for testing."""
    from genai_bench.async_runner.open_loop import OpenLoopRunner

    return OpenLoopRunner(
        sampler=mock_sampler,
        api_backend="openai",
        api_base="https://api.openai.com/v1",
        api_model_name="gpt-4",
        auth_provider=mock_auth_provider,
        aggregated_metrics_collector=mock_aggregated_metrics,
        dashboard=None,
    )


class TestOpenLoopRunner:
    """Test OpenLoopRunner QPS-based execution."""

    def test_wait_intervals_exponential(self, open_loop_runner):
        """Test exponential distribution for inter-arrival intervals."""
        intervals = open_loop_runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="exponential",
        )

        assert len(intervals) == 10
        assert all(i > 0 for i in intervals)

    def test_wait_intervals_uniform(self, open_loop_runner):
        """Test uniform distribution for inter-arrival intervals."""
        intervals = open_loop_runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="uniform",
        )

        assert len(intervals) == 10
        assert all(i > 0 for i in intervals)

    def test_wait_intervals_constant(self, open_loop_runner):
        """Test constant distribution for inter-arrival intervals."""
        intervals = open_loop_runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="constant",
        )

        assert len(intervals) == 10
        # Constant distribution should have same value for all intervals
        mean = 1.0 / 10.0
        assert all(abs(i - mean) < 0.001 for i in intervals)

    def test_wait_intervals_invalid_distribution(self, open_loop_runner):
        """Test that invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="Invalid distribution"):
            open_loop_runner._wait_intervals(
                qps_level=10.0,
                duration_s=1,
                random_seed=42,
                distribution="invalid",
            )

    def test_run_requires_qps_level(self, open_loop_runner):
        """Test that run requires qps_level for open-loop mode."""
        with pytest.raises(ValueError, match="qps_level is required"):
            open_loop_runner.run(
                qps_level=None,
                duration_s=1,
                distribution="exponential",
                random_seed=42,
                max_requests=None,
                max_time_s=None,
                scenario="D(100,100)",
            )

    def test_run_open_loop_execution(self, open_loop_runner, mock_aggregated_metrics):
        """Test open-loop execution with QPS-based scheduling."""
        open_loop_runner._send_one = AsyncMock()

        # Mock asyncio.sleep to speed up test
        with patch(
            "genai_bench.async_runner.open_loop.asyncio.sleep", new_callable=AsyncMock
        ):
            open_loop_runner.run(
                qps_level=2.0,  # 2 requests per second
                duration_s=1,  # 1 second = 2 requests
                distribution="constant",
                random_seed=42,
                max_requests=None,
                max_time_s=None,
                scenario="D(100,100)",
            )

        # Verify total_arrivals was set
        assert mock_aggregated_metrics.aggregated_metrics.total_arrivals == 2

        # Verify arrival rate was calculated
        assert (
            mock_aggregated_metrics.aggregated_metrics.arrival_requests_per_second
            is not None
        )

    def test_run_respects_max_requests(self, open_loop_runner, mock_aggregated_metrics):
        """Test that run respects max_requests limit."""
        open_loop_runner._send_one = AsyncMock()

        with patch(
            "genai_bench.async_runner.open_loop.asyncio.sleep", new_callable=AsyncMock
        ):
            open_loop_runner.run(
                qps_level=10.0,  # Would generate 10 requests
                duration_s=1,
                distribution="constant",
                random_seed=42,
                max_requests=5,  # But limit to 5
                max_time_s=None,
                scenario="D(100,100)",
            )

        # Should only send max_requests
        assert mock_aggregated_metrics.aggregated_metrics.total_arrivals == 5

    def test_run_respects_max_time(self, open_loop_runner):
        """Test that run respects max_time_s timeout."""
        open_loop_runner._send_one = AsyncMock()

        # Mock asyncio.wait_for to raise TimeoutError
        with patch(
            "genai_bench.async_runner.open_loop.asyncio.sleep", new_callable=AsyncMock
        ):
            with patch(
                "genai_bench.async_runner.open_loop.asyncio.wait_for"
            ) as mock_wait:
                import asyncio

                mock_wait.side_effect = asyncio.TimeoutError()
                with contextlib.suppress(asyncio.TimeoutError):
                    open_loop_runner.run(
                        qps_level=10.0,
                        duration_s=10,  # Would run for 10 seconds
                        distribution="constant",
                        random_seed=42,
                        max_requests=None,
                        max_time_s=1,  # But limit to 1 second
                        scenario="D(100,100)",
                    )
