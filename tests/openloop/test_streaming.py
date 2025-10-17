import asyncio
from unittest.mock import patch

from genai_bench.openloop.runner import OpenLoopRunner
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.protocol import UserChatRequest, UserChatResponse


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


def _ok_chat_resp() -> UserChatResponse:
    return UserChatResponse(
        status_code=200,
        generated_text="abcde",
        tokens_received=5,
        time_at_first_token=0.02,
        num_prefill_tokens=10,
        start_time=0.0,
        end_time=0.1,
    )


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_streaming_ttft_and_tokens(mock_send):
    async def _ok(self, req):
        # simulate small delay then a successful streaming completion
        await asyncio.sleep(0)
        return _ok_chat_resp()

    mock_send.side_effect = _ok
    runner, aggregated = _build_runner()

    # 2 arrivals immediately
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=[0.0, 0.0]):
        total_run_time = runner.run(
            qps_level=2,
            duration_s=1,
            distribution="constant",
            random_seed=0,
            max_requests=None,
            max_time_s=None,
            scenario="D(100,10)",
        )

    assert total_run_time >= 0
    assert aggregated.aggregated_metrics.num_completed_requests >= 1
    m = aggregated.all_request_metrics[0]
    assert m.ttft is not None and m.ttft > 0
    assert m.num_output_tokens is not None and m.num_output_tokens > 0

