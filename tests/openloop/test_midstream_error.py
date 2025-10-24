import asyncio
from typing import Tuple

from unittest.mock import patch

from genai_bench.openloop.runner import OpenLoopRunner
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.protocol import UserChatRequest, UserChatResponse, UserResponse


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


def _build_runner() -> Tuple[OpenLoopRunner, AggregatedMetricsCollector]:
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


def _err_resp() -> UserResponse:
    return UserResponse(status_code=500, error_message="mid-stream error")


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_midstream_error_recorded_without_blocking(mock_send):
    # First request fails, next two succeed
    async def _seq(self, req):
        if not hasattr(_seq, "i"):
            _seq.i = 0  # type: ignore[attr-defined]
        _seq.i += 1  # type: ignore[attr-defined]
        await asyncio.sleep(0)
        if _seq.i == 1:  # type: ignore[attr-defined]
            return _err_resp()
        return _ok_chat_resp()

    mock_send.side_effect = _seq
    runner, aggregated = _build_runner()

    # Three arrivals immediately
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=[0.0, 0.0, 0.0]):
        total_run_time = runner.run(
            qps_level=3,
            duration_s=1,
            distribution="constant",
            random_seed=0,
            max_requests=None,
            max_time_s=None,
            scenario="D(100,10)",
        )

    assert total_run_time >= 0
    # Two successes, one error
    assert aggregated.aggregated_metrics.num_completed_requests == 2
    # Error recorded in frequency map
    freq = aggregated.aggregated_metrics.error_codes_frequency
    assert 500 in freq and freq[500] == 1

