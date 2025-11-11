import asyncio
import time
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
def test_max_time_timeout(mock_send):
    async def _slow(self, req):
        await asyncio.sleep(1.0)
        return UserResponse(status_code=200, start_time=0.0, end_time=1.0, time_at_first_token=0.5, num_prefill_tokens=10)

    mock_send.side_effect = _slow
    runner, aggregated = _build_runner()

    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=[0.0] * 10):
        t0 = time.perf_counter()
        _ = runner.run(
            qps_level=10,
            duration_s=10,
            distribution="constant",
            random_seed=0,
            max_requests=None,
            max_time_s=0.5,
            scenario="D(100,10)",
        )
        t1 = time.perf_counter()

    # Should exit near the timeout
    assert (t1 - t0) < 2.0
    assert len(aggregated.all_request_metrics) < 10


