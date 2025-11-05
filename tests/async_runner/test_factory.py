"""Tests for runner factory function."""

from unittest.mock import MagicMock

import pytest

from genai_bench.async_runner.factory import create_runner


@pytest.fixture
def mock_sampler():
    """Mock sampler for testing."""
    return MagicMock()


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
    return MagicMock()


class TestFactory:
    """Test factory function for creating runners."""

    def test_create_open_loop_runner(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test creating OpenLoopRunner when qps_level is provided."""
        runner = create_runner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
            qps_level=10.0,
            target_concurrency=None,
        )

        from genai_bench.async_runner.open_loop import OpenLoopRunner

        assert isinstance(runner, OpenLoopRunner)

    def test_create_closed_loop_runner(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test creating ClosedLoopRunner when target_concurrency is provided."""
        runner = create_runner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
            qps_level=None,
            target_concurrency=5,
        )

        from genai_bench.async_runner.closed_loop import ClosedLoopRunner

        assert isinstance(runner, ClosedLoopRunner)

    def test_create_runner_requires_one_mode(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test that factory requires either qps_level or target_concurrency."""
        with pytest.raises(
            ValueError, match="Must specify either qps_level.*or target_concurrency"
        ):
            create_runner(
                sampler=mock_sampler,
                api_backend="openai",
                api_base="https://api.openai.com/v1",
                api_model_name="gpt-4",
                auth_provider=mock_auth_provider,
                aggregated_metrics_collector=mock_aggregated_metrics,
                dashboard=None,
                qps_level=None,
                target_concurrency=None,
            )

    def test_create_runner_rejects_both_modes(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test that factory rejects both qps_level and target_concurrency."""
        with pytest.raises(
            ValueError, match="Cannot specify both qps_level and target_concurrency"
        ):
            create_runner(
                sampler=mock_sampler,
                api_backend="openai",
                api_base="https://api.openai.com/v1",
                api_model_name="gpt-4",
                auth_provider=mock_auth_provider,
                aggregated_metrics_collector=mock_aggregated_metrics,
                dashboard=None,
                qps_level=10.0,
                target_concurrency=5,
            )

    def test_create_runner_passes_common_params(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test that factory passes common parameters to runner."""
        runner = create_runner(
            sampler=mock_sampler,
            api_backend="baseten",
            api_base="https://api.baseten.com/v1",
            api_model_name="model-123",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
            qps_level=5.0,
            target_concurrency=None,
        )

        assert runner.sampler == mock_sampler
        assert runner.api_backend == "baseten"
        assert runner.api_base == "https://api.baseten.com/v1"
        assert runner.api_model_name == "model-123"
        assert runner.auth_provider == mock_auth_provider
        assert runner.aggregated == mock_aggregated_metrics
