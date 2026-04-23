"""Tests for audio transcription metrics calculation."""

from unittest.mock import MagicMock

import pytest

from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserAudioTranscriptionResponse


def _make_audio_response(
    transcribed_text: str,
    duration_s: float = 10.0,
    tokens_received: int = None,
):
    """Create a mock UserAudioTranscriptionResponse."""
    mock = MagicMock(spec=UserAudioTranscriptionResponse)
    mock.status_code = 200
    mock.transcribed_text = transcribed_text
    mock.audio_duration_s = duration_s
    mock.num_prefill_tokens = int(duration_s * 100)
    mock.time_at_first_token = 1000.5
    mock.start_time = 1000.0
    mock.end_time = 1000.5
    mock.tokens_received = tokens_received
    return mock


def test_audio_metrics_with_tokens():
    """TPOT and output_throughput computed from real token count."""
    # duration=5.0s, e2e_latency=0.5s, RTF=10.0, tokens=4
    response = _make_audio_response("Hello world", duration_s=5.0, tokens_received=4)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_input_tokens == 500  # 5.0 * 100
    assert collector.metrics.e2e_latency == pytest.approx(0.5)
    assert collector.metrics.num_output_tokens == 4
    assert collector.metrics.output_inference_speed == pytest.approx(10.0)  # RTF
    assert collector.metrics.tpot == pytest.approx(0.5 / 3)  # latency / (tokens-1)
    assert collector.metrics.output_throughput == pytest.approx(3 / 0.5)
    assert collector.metrics.num_reasoning_tokens == 0


def test_audio_metrics_zero_tokens():
    """Zero tokens: RTF still computed, TPOT and throughput are 0."""
    response = _make_audio_response("", duration_s=3.0, tokens_received=0)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_output_tokens == 0
    assert collector.metrics.tpot == 0
    assert collector.metrics.output_throughput == 0
    # RTF = 3.0 / 0.5 = 6.0
    assert collector.metrics.output_inference_speed == pytest.approx(6.0)


def test_audio_metrics_none_tokens_falls_to_zero():
    """None tokens_received treated as 0."""
    response = _make_audio_response("Hello", duration_s=4.0, tokens_received=None)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_output_tokens == 0
    # RTF = 4.0 / 0.5 = 8.0
    assert collector.metrics.output_inference_speed == pytest.approx(8.0)


def test_audio_metrics_zero_duration():
    """Zero audio duration gives RTF=0."""
    response = _make_audio_response("Hello", duration_s=0.0, tokens_received=2)
    response.audio_duration_s = 0.0
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.output_inference_speed == 0.0


def test_audio_metrics_total_tokens():
    """total_tokens = input tokens + output tokens."""
    response = _make_audio_response("Hello world", duration_s=5.0, tokens_received=3)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.total_tokens == 500 + 3
