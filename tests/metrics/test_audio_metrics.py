"""Tests for audio transcription metrics calculation."""

from unittest.mock import MagicMock

import pytest

from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserAudioTranscriptionResponse


def _make_audio_response(transcribed_text: str, duration_s: float = 10.0):
    """Create a mock UserAudioTranscriptionResponse."""
    mock = MagicMock(spec=UserAudioTranscriptionResponse)
    mock.status_code = 200
    mock.transcribed_text = transcribed_text
    mock.audio_duration_s = duration_s
    mock.num_prefill_tokens = int(duration_s * 100)
    mock.time_at_first_token = 1000.5
    mock.start_time = 1000.0
    mock.end_time = 1000.5
    return mock


def test_audio_metrics_with_transcription():
    """output_inference_speed stores RTF = audio_duration_s / e2e_latency."""
    # duration=5.0s, e2e_latency = end-start = 1000.5-1000.0 = 0.5s
    # RTF = 5.0 / 0.5 = 10.0
    response = _make_audio_response("Hello world", duration_s=5.0)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_input_tokens == 500  # 5.0 * 100
    assert collector.metrics.e2e_latency == pytest.approx(0.5)
    assert collector.metrics.num_output_tokens == len("Hello world")
    assert collector.metrics.output_latency == collector.metrics.e2e_latency
    assert collector.metrics.output_inference_speed == pytest.approx(10.0)  # RTF
    assert collector.metrics.tpot == 0
    assert collector.metrics.output_throughput == 0
    assert collector.metrics.num_reasoning_tokens == 0


def test_audio_metrics_empty_transcription():
    """Empty transcription: output_tokens=0, RTF still computed from duration."""
    response = _make_audio_response("", duration_s=3.0)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_output_tokens == 0
    assert collector.metrics.tpot == 0
    assert collector.metrics.output_throughput == 0
    # RTF = 3.0 / 0.5 = 6.0
    assert collector.metrics.output_inference_speed == pytest.approx(6.0)


def test_audio_metrics_none_transcription():
    """None transcribed_text treated as empty string; RTF still computed."""
    response = _make_audio_response("", duration_s=4.0)
    response.transcribed_text = None
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.num_output_tokens == 0
    # RTF = 4.0 / 0.5 = 8.0
    assert collector.metrics.output_inference_speed == pytest.approx(8.0)


def test_audio_metrics_zero_duration():
    """Zero audio duration gives RTF=0."""
    response = _make_audio_response("Hello", duration_s=0.0)
    response.audio_duration_s = 0.0
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.output_inference_speed == 0.0


def test_audio_metrics_total_tokens():
    """total_tokens = input tokens + output (char) tokens."""
    text = "abcdefghij"  # 10 chars
    response = _make_audio_response(text, duration_s=5.0)
    collector = RequestMetricsCollector()
    collector.calculate_metrics(response)

    assert collector.metrics.total_tokens == 500 + 10
