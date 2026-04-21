"""Sampler for audio-to-text (speech transcription) benchmarking."""

import io
import random
import struct
import wave
from typing import Any, List, Optional, Tuple

import soundfile as sf

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import UserAudioTranscriptionRequest, UserRequest
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.multimodal import AudioScenario

logger = init_logger(__name__)


class AudioSampler(Sampler):
    """
    Sampler for audio-to-text tasks.

    Expects data to be a list of (audio_bytes, duration_s, filename) tuples
    as returned by AudioDatasetLoader.

    Scenario-driven mode: pass A(mean_s,std_s) as --traffic-scenario.
    Each request samples a duration from N(mean_s, std_s) and truncates the
    clip accordingly. This allows multiple scenarios to be benchmarked in one
    run and plotted on the same graph (same as text-to-text N(...) scenarios).

    Fixed-duration fallback: set clip_duration_s in --additional-request-params
    to truncate all requests to a fixed length (used when --traffic-scenario dataset).
    """

    input_modality = "audio"
    supported_tasks = {"audio-to-text"}

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: Any,
        dataset_config: Optional[DatasetConfig] = None,
        additional_request_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            model,
            output_modality,
            additional_request_params,
            dataset_config=dataset_config,
        )
        self.data: List[Tuple[bytes, Optional[float], str]] = data

    def sample(self, scenario: Optional[Scenario]) -> UserRequest:
        """Sample a random audio clip and return a UserAudioTranscriptionRequest."""
        audio_bytes, duration_s, filename = random.choice(self.data)

        # Scenario-driven: A(mean_s,std_s) samples duration from Gaussian distribution
        if isinstance(scenario, AudioScenario):
            clip_duration_s = scenario.sample()
        else:
            clip_duration_s = self.additional_request_params.get("clip_duration_s")

        if clip_duration_s is not None:
            suffix = filename.rsplit(".", 1)[-1].lower()
            if suffix == "wav":
                audio_bytes, duration_s = _truncate_wav(audio_bytes, float(clip_duration_s))
            elif suffix == "flac":
                audio_bytes, duration_s = _truncate_flac(audio_bytes, float(clip_duration_s))

        return UserAudioTranscriptionRequest(
            model=self.model,
            audio_content=audio_bytes,
            audio_filename=filename,
            audio_duration_s=duration_s,
            language=self.additional_request_params.get("language"),
            response_format=self.additional_request_params.get(
                "response_format", "json"
            ),
            additional_request_params={
                k: v
                for k, v in self.additional_request_params.items()
                if k not in ("language", "response_format", "clip_duration_s")
            },
        )


def _truncate_flac(audio_bytes: bytes, max_duration_s: float) -> Tuple[bytes, float]:
    """Truncate a FLAC file to at most max_duration_s seconds.

    Returns (truncated_bytes, actual_duration_s).
    If the file is already shorter than max_duration_s, returns it unchanged.
    Returns the original bytes unchanged on parse error.
    """
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            samplerate = f.samplerate
            total_frames = f.frames
            max_frames = int(max_duration_s * samplerate)
            if total_frames <= max_frames:
                return audio_bytes, total_frames / samplerate
            samples = f.read(max_frames, dtype="int16", always_2d=True)

        buf = io.BytesIO()
        sf.write(buf, samples, samplerate, format="flac", subtype="PCM_16")
        return buf.getvalue(), max_duration_s
    except Exception as e:
        logger.warning("Could not truncate FLAC, using original: %s", e)
        return audio_bytes, 0.0


def _truncate_wav(audio_bytes: bytes, max_duration_s: float) -> Tuple[bytes, float]:
    """Truncate a WAV file to at most max_duration_s seconds.

    Returns (truncated_bytes, actual_duration_s).
    If the file is already shorter than max_duration_s, returns it unchanged.
    Returns the original bytes unchanged on parse error.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()
            max_frames = int(max_duration_s * framerate)
            if total_frames <= max_frames:
                return audio_bytes, total_frames / framerate
            wf.rewind()
            frames = wf.readframes(max_frames)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nchannels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(frames)
        return buf.getvalue(), max_duration_s
    except (wave.Error, struct.error, EOFError) as e:
        logger.warning("Could not truncate WAV, using original: %s", e)
        return audio_bytes, 0.0
