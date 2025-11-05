"""Tests for BaseAsyncRunner shared functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_bench.protocol import UserChatRequest, UserChatResponse, UserResponse
from genai_bench.scenarios.base import Scenario


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
    metrics.add_single_request_metrics = MagicMock()
    metrics.get_live_metrics = MagicMock(return_value={})
    metrics.clear = MagicMock()
    return metrics


class TestBaseAsyncRunner:
    """Test BaseAsyncRunner shared functionality."""

    def test_init(self, mock_sampler, mock_auth_provider, mock_aggregated_metrics):
        """Test BaseAsyncRunner initialization."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        assert runner.sampler == mock_sampler
        assert runner.api_backend == "openai"
        assert runner.api_base == "https://api.openai.com/v1"
        assert runner.api_model_name == "gpt-4"
        assert runner.auth_provider == mock_auth_provider
        assert runner.headers is not None
        assert "Authorization" in runner.headers

    def test_init_with_credentials(self, mock_sampler, mock_aggregated_metrics):
        """Test initialization with credentials instead of headers."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        auth = MagicMock()
        # Remove get_headers to ensure it uses get_credentials path
        del auth.get_headers
        auth.get_credentials = MagicMock(return_value="test-token")

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=auth,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        assert runner.headers is not None
        assert runner.headers["Authorization"] == "Bearer test-token"

    def test_prepare_request_from_string(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test preparing request from scenario string."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        req = runner._prepare_request("D(100,100)")

        assert isinstance(req, UserChatRequest)
        mock_sampler.sample.assert_called_once()
        # Verify scenario was converted to object
        call_args = mock_sampler.sample.call_args[0][0]
        assert isinstance(call_args, Scenario)

    def test_prepare_request_from_scenario(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test preparing request from Scenario object."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        scenario = Scenario.from_string("D(100,100)")
        req = runner._prepare_request(scenario)

        assert isinstance(req, UserChatRequest)
        mock_sampler.sample.assert_called_once_with(scenario)

    @pytest.mark.asyncio
    async def test_probe_latency(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test latency probing."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        # Mock successful responses
        mock_response = UserChatResponse(
            status_code=200,
            generated_text="test",
            tokens_received=10,
            time_at_first_token=0.5,
            num_prefill_tokens=5,
            start_time=0.0,
            end_time=1.0,
        )

        runner._send_request = AsyncMock(return_value=mock_response)

        avg_latency = await runner._probe_latency("D(100,100)", num_probe_requests=5)

        assert avg_latency == 1.0  # end_time - start_time
        assert runner._send_request.call_count == 5

    @pytest.mark.asyncio
    async def test_probe_latency_with_failures(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test latency probing with some failed requests."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        # Mock mix of successful and failed responses
        mock_success = UserChatResponse(
            status_code=200,
            generated_text="test",
            tokens_received=10,
            time_at_first_token=0.5,
            num_prefill_tokens=5,
            start_time=0.0,
            end_time=1.0,
        )
        mock_failure = UserResponse(status_code=500, error_message="Error")

        runner._send_request = AsyncMock(
            side_effect=[
                mock_success,
                mock_failure,
                mock_success,
                mock_success,
                mock_success,
            ]
        )

        avg_latency = await runner._probe_latency("D(100,100)", num_probe_requests=5)

        # Should calculate from successful requests only
        assert avg_latency == 1.0
        assert runner._send_request.call_count == 5

    @pytest.mark.asyncio
    async def test_probe_latency_all_failures(
        self, mock_sampler, mock_auth_provider, mock_aggregated_metrics
    ):
        """Test latency probing when all requests fail."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs) -> float:
                return 1.0

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com/v1",
            api_model_name="gpt-4",
            auth_provider=mock_auth_provider,
            aggregated_metrics_collector=mock_aggregated_metrics,
            dashboard=None,
        )

        mock_failure = UserResponse(status_code=500, error_message="Error")
        runner._send_request = AsyncMock(return_value=mock_failure)

        with pytest.raises(ValueError, match="Latency probe failed"):
            await runner._probe_latency("D(100,100)", num_probe_requests=5)
