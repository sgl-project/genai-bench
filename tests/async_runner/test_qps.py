"""Tests for QPS traffic pattern generation in async runner."""

import pytest

from genai_bench.async_runner.open_loop import OpenLoopRunner


class TestQPS:
    """Test QPS traffic pattern generation."""

    def test_wait_intervals_exponential(self):
        """Test exponential distribution for inter-arrival intervals."""
        runner = OpenLoopRunner(
            sampler=None,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=None,
            aggregated_metrics_collector=None,
            dashboard=None,
        )

        intervals = runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="exponential",
        )

        assert len(intervals) == 10
        assert all(i > 0 for i in intervals)

    def test_wait_intervals_uniform(self):
        """Test uniform distribution for inter-arrival intervals."""
        runner = OpenLoopRunner(
            sampler=None,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=None,
            aggregated_metrics_collector=None,
            dashboard=None,
        )

        intervals = runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="uniform",
        )

        assert len(intervals) == 10
        assert all(i > 0 for i in intervals)

    def test_wait_intervals_constant(self):
        """Test constant distribution for inter-arrival intervals."""
        runner = OpenLoopRunner(
            sampler=None,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=None,
            aggregated_metrics_collector=None,
            dashboard=None,
        )

        intervals = runner._wait_intervals(
            qps_level=10.0,
            duration_s=1,
            random_seed=42,
            distribution="constant",
        )

        assert len(intervals) == 10
        # Constant distribution should have same value for all intervals
        mean = 1.0 / 10.0
        assert all(abs(i - mean) < 0.001 for i in intervals)

    def test_wait_intervals_invalid_distribution(self):
        """Test that invalid distribution raises ValueError."""
        runner = OpenLoopRunner(
            sampler=None,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=None,
            aggregated_metrics_collector=None,
            dashboard=None,
        )

        with pytest.raises(ValueError, match="Invalid distribution"):
            runner._wait_intervals(
                qps_level=10.0,
                duration_s=1,
                random_seed=42,
                distribution="invalid",
            )
