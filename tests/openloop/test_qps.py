import time
import asyncio
from typing import Any, List

import pytest
from unittest.mock import patch

from genai_bench.openloop.runner import OpenLoopRunner
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.protocol import UserChatRequest, UserResponse


class DummyAuth:
    def get_credentials(self) -> str:
        return "test-token"


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


class DummyResp:
    def __init__(self, status_code: int = 200, prompt_tokens: int = 10, completion_tokens: int = 1) -> None:
        self.status_code = status_code
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.text = "OK"

    def json(self) -> Any:
        return {
            "usage": {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            }
        }

    def close(self) -> None:
        return None


def _build_runner() -> tuple[OpenLoopRunner, AggregatedMetricsCollector]:
    aggregated = AggregatedMetricsCollector()
    runner = OpenLoopRunner(
        sampler=DummySampler(),
        api_backend="openai",
        api_base="https://example.com",
        api_model_name="dummy-model",
        auth_provider=DummyAuth(),
        aggregated_metrics_collector=aggregated,
        dashboard=None,
    )
    return runner, aggregated


def test_wait_intervals_reproducible_and_count():
    runner, _ = _build_runner()
    qps = 10
    duration = 2
    n = qps * duration
    intervals_a: List[float] = runner._wait_intervals(qps, duration, random_seed=42, distribution="uniform")
    intervals_b: List[float] = runner._wait_intervals(qps, duration, random_seed=42, distribution="uniform")
    intervals_c: List[float] = runner._wait_intervals(qps, duration, random_seed=43, distribution="uniform")
    assert len(intervals_a) == n
    assert intervals_a == intervals_b
    assert intervals_a != intervals_c


def test_wait_intervals_constant_distribution():
    runner, _ = _build_runner()
    qps = 5
    duration = 3
    intervals = runner._wait_intervals(qps, duration, random_seed=123, distribution="constant")
    assert all(abs(x - 1.0 / qps) < 1e-9 for x in intervals)


def test_wait_intervals_exponential_mean_close():
    runner, _ = _build_runner()
    qps = 10
    duration = 100  # enough samples for mean to concentrate
    intervals = runner._wait_intervals(qps, duration, random_seed=999, distribution="exponential")
    empirical_mean = sum(intervals) / len(intervals)
    assert abs(empirical_mean - (1.0 / qps)) < 0.05  # loose tolerance


def _ok_resp() -> UserResponse:
    return UserResponse(
        status_code=200,
        start_time=0.0,
        end_time=0.1,
        time_at_first_token=0.02,
        num_prefill_tokens=10,
    )


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_run_dispatches_exact_number_of_requests(mock_send):
    async def _ok(self, req):
        return _ok_resp()
    mock_send.side_effect = _ok
    runner, aggregated = _build_runner()
    qps = 7
    duration = 2
    expected = qps * duration

    # Force zero intervals for a quick run
    zero_intervals = [0.0] * expected
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=zero_intervals):
        total_run_time = runner.run(
            qps_level=qps,
            duration_s=duration,
            distribution="uniform",
            random_seed=42,
            max_requests=None,
            max_time_s=None,
            scenario="D(100,100)",
        )

    assert total_run_time >= 0
    assert len(aggregated.all_request_metrics) == expected


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_run_respects_max_requests(mock_send):
    async def _ok(self, req):
        return _ok_resp()
    mock_send.side_effect = _ok
    runner, aggregated = _build_runner()
    qps = 50
    duration = 2
    target = qps * duration
    max_requests = 30

    zero_intervals = [0.0] * target
    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=zero_intervals):
        runner.run(
            qps_level=qps,
            duration_s=duration,
            distribution="uniform",
            random_seed=42,
            max_requests=max_requests,
            max_time_s=None,
            scenario="D(100,100)",
        )

    assert len(aggregated.all_request_metrics) == max_requests


@patch.object(OpenLoopRunner, "_send_request", autospec=True)
def test_run_honors_timeout(mock_send):
    async def _slow(self, req):
        await asyncio.sleep(1.0)
        return _ok_resp()
    mock_send.side_effect = _slow
    runner, aggregated = _build_runner()
    qps = 5
    duration = 100  # many intervals
    zero_intervals = [0.1] * (qps * 10)

    with patch.object(OpenLoopRunner, "_wait_intervals", return_value=zero_intervals):
        start = time.monotonic()
        runner.run(
            qps_level=qps,
            duration_s=duration,
            distribution="uniform",
            random_seed=42,
            max_requests=None,
            max_time_s=0.5,  # time out early
            scenario="D(100,100)",
        )
        end = time.monotonic()

    # Should stop in around 0.5s, allow slack
    assert (end - start) < 2.0
    # And we should have fewer requests than intervals list length
    assert len(aggregated.all_request_metrics) < len(zero_intervals)


