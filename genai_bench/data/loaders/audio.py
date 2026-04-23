"""Dataset loader for audio files used in audio-to-text benchmarking."""

import io
import struct
import wave
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class AudioDatasetLoader(DatasetLoader):
    """
    Loads audio files from a local file path for audio-to-text benchmarking.

    Supported source: local file only (wav, mp3, flac, m4a).

    Returns a list of (audio_bytes, duration_s, filename) tuples.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,  # reuse TEXT format enum for local file paths
    }
    media_type = "Audio"

    # Supported audio extensions
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

    def _process_loaded_data(self, data: Any) -> List[Tuple[bytes, float, str]]:
        """Process loaded data into (audio_bytes, duration_s, filename) tuples."""
        # data from file source is a list of strings (file paths)
        if not isinstance(data, list):
            raise ValueError(
                f"AudioDatasetLoader expected a list of file paths, got {type(data)}"
            )

        results = []
        for item in data:
            path = Path(item.strip())
            if not path.exists():
                logger.warning("Audio file not found, skipping: %s", path)
                continue
            if path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                logger.warning(
                    "Unsupported audio extension '%s', skipping: %s",
                    path.suffix,
                    path,
                )
                continue
            audio_bytes = path.read_bytes()
            duration_s = _get_audio_duration(audio_bytes, path.suffix.lower())
            results.append((audio_bytes, duration_s, path.name))

        if not results:
            raise ValueError(
                "No valid audio files found. "
                "Provide a text file listing one audio file path per line."
            )
        return results


def _get_audio_duration(audio_bytes: bytes, suffix: str) -> Optional[float]:
    """Return duration in seconds for an audio file, or None on parse error."""
    try:
        if suffix == ".wav":
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                return wf.getnframes() / wf.getframerate()
        else:
            import soundfile as sf

            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                return f.frames / f.samplerate
    except Exception:
        return None
