"""Tests for AudioDatasetLoader."""

import io
import wave

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.audio import AudioDatasetLoader, _get_wav_duration


def _make_wav_bytes(duration_s: float = 1.0, framerate: int = 16000) -> bytes:
    """Create minimal in-memory WAV bytes for testing."""
    nframes = int(duration_s * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


@pytest.fixture
def dataset_config(tmp_path):
    list_file = tmp_path / "files.txt"
    list_file.write_text("")
    return DatasetConfig(
        source=DatasetSourceConfig(
            type="file", path=str(list_file), file_format="txt"
        ),
    )


def test_get_wav_duration_valid():
    wav = _make_wav_bytes(duration_s=3.0, framerate=16000)
    assert _get_wav_duration(wav) == pytest.approx(3.0, rel=1e-3)


def test_get_wav_duration_invalid():
    assert _get_wav_duration(b"not a wav file") == 0.0


def test_load_wav_files(dataset_config, tmp_path):
    wav_bytes = _make_wav_bytes(duration_s=2.0)
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(wav_bytes)

    loader = AudioDatasetLoader(dataset_config)
    results = loader._process_loaded_data([str(wav_path)])

    assert len(results) == 1
    audio_bytes, duration_s, filename = results[0]
    assert filename == "test.wav"
    assert duration_s == pytest.approx(2.0, rel=1e-3)
    assert len(audio_bytes) > 0


def test_skip_missing_file(dataset_config, tmp_path):
    wav_path = tmp_path / "real.wav"
    wav_path.write_bytes(_make_wav_bytes())
    missing_path = tmp_path / "missing.wav"

    loader = AudioDatasetLoader(dataset_config)
    results = loader._process_loaded_data([str(missing_path), str(wav_path)])

    assert len(results) == 1
    assert results[0][2] == "real.wav"


def test_skip_unsupported_extension(dataset_config, tmp_path):
    txt_path = tmp_path / "audio.txt"
    txt_path.write_text("not audio")
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(_make_wav_bytes())

    loader = AudioDatasetLoader(dataset_config)
    results = loader._process_loaded_data([str(txt_path), str(wav_path)])

    assert len(results) == 1
    assert results[0][2] == "audio.wav"


def test_raises_when_no_valid_files(dataset_config, tmp_path):
    loader = AudioDatasetLoader(dataset_config)
    with pytest.raises(ValueError, match="No valid audio files"):
        loader._process_loaded_data(["/nonexistent/file.wav"])


def test_non_wav_has_none_duration(dataset_config, tmp_path):
    mp3_path = tmp_path / "audio.mp3"
    mp3_path.write_bytes(b"fake mp3 data")

    loader = AudioDatasetLoader(dataset_config)
    results = loader._process_loaded_data([str(mp3_path)])

    assert len(results) == 1
    _, duration_s, filename = results[0]
    assert duration_s is None
    assert filename == "audio.mp3"
