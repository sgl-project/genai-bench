from unittest.mock import patch

from genai_bench.openloop.runner import OpenLoopRunner
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.protocol import UserChatRequest, UserResponse


class DummySampler:
    def __init__(self, model: str = "dummy-model") -> None:
        self.model = model

    def sample(self, scenario: str) -> UserChatRequest:
        return UserChatRequest(
            model=self.model,
            prompt="Hello",
            num_prefill_tokens=10,
            max_tokens=10,
            additional_request_params={},
        )


def _build_runner() -> tuple[OpenLoopRunner, AggregatedMetricsCollector]:
    aggregated = AggregatedMetricsCollector()
    runner = OpenLoopRunner(
        sampler=DummySampler(),
        api_backend="openai",
        api_base="https://example.com",
        api_model_name="dummy-model",
        auth_provider=None,
        aggregated_metrics_collector=aggregated,
        dashboard=None,
    )
    return runner, aggregated


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_arrival_metrics_recorded(mock_send):
    # Quick success stub
    async def _ok(self, req):
        return UserResponse(status_code=200, start_time=0.0, end_time=0.01, time_at_first_token=0.001, num_prefill_tokens=10)

    mock_send.side_effect = _ok
    runner, aggregated = _build_runner()

    # qps=5, duration=2 -> 10 planned arrivals
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=[0.0] * 10):
        _ = runner.run(
            qps_level=5,
            duration_s=2,
            distribution="constant",
            random_seed=0,
            max_requests=None,
            max_time_s=None,
            scenario="D(100,10)",
        )

    metrics = aggregated.aggregated_metrics
    assert metrics.total_arrivals == 10
    assert abs(metrics.arrival_requests_per_second - 5.0) < 1e-6


