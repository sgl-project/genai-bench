"""Tests for AudioSampler and _truncate_wav."""

import io
import wave
from unittest.mock import MagicMock

import pytest

from genai_bench.protocol import UserAudioTranscriptionRequest
from genai_bench.sampling.audio import AudioSampler, _truncate_wav
from genai_bench.scenarios.multimodal import AudioScenario


def _make_wav_bytes(duration_s: float = 3.0, framerate: int = 16000) -> bytes:
    nframes = int(duration_s * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


@pytest.fixture
def sampler():
    wav = _make_wav_bytes(duration_s=5.0)
    data = [(wav, 5.0, "test.wav")]
    tokenizer = MagicMock()
    return AudioSampler(
        tokenizer=tokenizer,
        model="whisper",
        output_modality="text",
        data=data,
        additional_request_params={"response_format": "json"},
    )


def test_sample_returns_request(sampler):
    request = sampler.sample(scenario=None)
    assert isinstance(request, UserAudioTranscriptionRequest)
    assert request.model == "whisper"
    assert request.audio_filename == "test.wav"
    assert request.response_format == "json"


def test_sample_with_audio_scenario(sampler):
    scenario = AudioScenario(mean_s=3, std_s=1)
    request = sampler.sample(scenario=scenario)
    assert isinstance(request, UserAudioTranscriptionRequest)
    # Duration should be truncated to scenario sample, wav gets re-encoded
    assert request.audio_duration_s is not None
    assert request.audio_duration_s <= 3 + 3 * 1  # within 3 std


def test_sample_with_clip_duration_param():
    wav = _make_wav_bytes(duration_s=10.0)
    data = [(wav, 10.0, "long.wav")]
    tokenizer = MagicMock()
    s = AudioSampler(
        tokenizer=tokenizer,
        model="whisper",
        output_modality="text",
        data=data,
        additional_request_params={"clip_duration_s": 2.0, "response_format": "json"},
    )
    request = s.sample(scenario=None)
    assert request.audio_duration_s == pytest.approx(2.0, rel=1e-3)


def test_sample_non_wav_skips_truncation():
    mp3_data = b"fake mp3 content"
    data = [(mp3_data, None, "audio.mp3")]
    tokenizer = MagicMock()
    s = AudioSampler(
        tokenizer=tokenizer,
        model="whisper",
        output_modality="text",
        data=data,
        additional_request_params={"clip_duration_s": 2.0, "response_format": "json"},
    )
    request = s.sample(scenario=None)
    # mp3 is not truncated, original bytes preserved
    assert request.audio_content == mp3_data


def test_truncate_wav_shorter_than_max():
    wav = _make_wav_bytes(duration_s=2.0)
    result_bytes, result_dur = _truncate_wav(wav, max_duration_s=5.0)
    # File shorter than max — returned unchanged
    assert result_dur == pytest.approx(2.0, rel=1e-3)
    assert result_bytes == wav


def test_truncate_wav_longer_than_max():
    wav = _make_wav_bytes(duration_s=5.0)
    result_bytes, result_dur = _truncate_wav(wav, max_duration_s=2.0)
    assert result_dur == pytest.approx(2.0, rel=1e-3)
    assert len(result_bytes) < len(wav)


def test_truncate_wav_invalid_bytes():
    result_bytes, result_dur = _truncate_wav(b"not a wav", max_duration_s=3.0)
    assert result_bytes == b"not a wav"
    assert result_dur == 0.0
