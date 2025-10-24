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
def test_session_per_request_and_metrics(mock_send):
    async def _ok(self, req):
        return UserResponse(status_code=200, start_time=0.0, end_time=0.01, time_at_first_token=0.001, num_prefill_tokens=10)

    mock_send.side_effect = _ok

    runner, aggregated = _build_runner()
    # Two immediate arrivals
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=[0.0, 0.0]):
        _ = runner.run(
            qps_level=2,
            duration_s=1,
            distribution="constant",
            random_seed=0,
            max_requests=None,
            max_time_s=None,
            scenario="D(100,10)",
        )

    # Validate two completions were recorded
    assert aggregated.aggregated_metrics.num_completed_requests == 2


