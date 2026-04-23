"""Remote speaker embedding extraction and local speaker identification."""

import io
import logging
import math
import shutil
import wave
from datetime import datetime
from pathlib import Path

import httpx

from meetscribe.errors import SpeachesAPIError, speaches_retry

from .models import SpeechSegment

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Speaker embedding extraction via speaches POST /v1/audio/speech/embedding."""

    def __init__(self, server_url: str, timeout: float, min_duration_ms: int, model: str):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.min_duration_ms = min_duration_ms
        self.model = model

    @speaches_retry
    def extract(self, audio_bytes: bytes, filename: str = "audio.wav") -> list[float]:
        """Extract speaker embedding from audio bytes.

        Args:
            audio_bytes: WAV audio data.
            filename: Filename for the multipart upload.

        Returns:
            Embedding vector as list of floats.
        """
        endpoint = f"{self.server_url}/v1/audio/speech/embedding"
        try:
            response = httpx.post(
                endpoint,
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={"model": self.model},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SpeachesAPIError(
                f"Embedding failed: {e.response.status_code}",
                status_code=e.response.status_code,
                endpoint=endpoint,
                detail=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise SpeachesAPIError(
                f"Embedding connection error: {e}",
                endpoint=endpoint,
            ) from e

        result = response.json()
        try:
            embedding = result["data"][0]["embedding"]
            return [float(x) for x in embedding]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise SpeachesAPIError(
                f"Unexpected embedding response format: {e}",
                endpoint=endpoint,
                detail=str(result)[:200],
            ) from e

    def extract_from_file(self, audio_path: Path) -> list[float]:
        """Extract speaker embedding from an audio file."""
        return self.extract(audio_path.read_bytes(), audio_path.name)


def slice_wav(raw_frames: bytes, sample_rate: int, sample_width: int, seg: SpeechSegment) -> bytes:
    """Slice raw PCM frames and wrap into WAV bytes in memory."""
    start_sample = seg.start_ms * sample_rate // 1000
    end_sample = seg.end_ms * sample_rate // 1000
    frame_size = sample_width  # mono
    chunk = raw_frames[start_sample * frame_size : end_sample * frame_size]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(chunk)
    return buf.getvalue()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_voiceprint(extractor: EmbeddingExtractor, wav_files: list[Path]) -> list[float]:
    """Extract embeddings from wav files and return their average."""
    embeddings = [extractor.extract_from_file(path) for path in wav_files]
    return [sum(col) / len(col) for col in zip(*embeddings)]


def enroll_samples(
    extractor: EmbeddingExtractor,
    sample_paths: list[Path],
    enrolled_dir: Path,
) -> tuple[list[float], int, int]:
    """Copy samples to enrolled dir and compute voiceprint from all samples.

    Returns (avg_embedding, total_count, new_count).
    """
    enrolled_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = enrolled_dir.resolve()
    date_prefix = datetime.now().strftime("%Y%m%d")
    new_count = 0
    for path in sample_paths:
        # Skip files already in enrolled_dir
        if path.resolve().parent == resolved_dir:
            continue
        dest = enrolled_dir / f"{date_prefix}_{path.name}"
        if path.exists():
            shutil.copy2(path, dest)
            new_count += 1

    all_wav_files = sorted(enrolled_dir.glob("*.wav"))
    avg_embedding = compute_voiceprint(extractor, all_wav_files)
    return avg_embedding, len(all_wav_files), new_count


class SpeakerIdentifier:
    """Identify speakers by comparing embeddings against stored voiceprints."""

    def __init__(
        self,
        voiceprints: dict[str, list[float]],
        threshold: float,
        confident_gap: float,
        min_threshold: float,
    ):
        self.voiceprints = voiceprints
        self.threshold = threshold
        self.confident_gap = confident_gap
        self.min_threshold = min_threshold
        logger.info(
            "Speaker identifier initialized",
            extra={
                "voiceprints": len(self.voiceprints),
                "threshold": threshold,
            },
        )

    def identify(self, embedding: list[float]) -> tuple[str | None, float]:
        """Find the best matching known speaker.

        Uses two criteria:
        1. Direct match: best_sim >= threshold
        2. Confident gap: best_sim >= min_threshold and gap to 2nd >= confident_gap

        Returns:
            (speaker_name, similarity) if matched, else (None, best_sim).
        """
        if not self.voiceprints:
            return None, -1.0

        scores = [(cosine_similarity(embedding, vp), name) for name, vp in self.voiceprints.items()]
        scores.sort(reverse=True)

        best_sim, best_name = scores[0]
        second_sim = scores[1][0] if len(scores) > 1 else -1.0

        if best_sim >= self.threshold:
            return best_name, best_sim

        # Confident gap: large margin over 2nd candidate
        gap = best_sim - second_sim
        if best_sim >= self.min_threshold and gap >= self.confident_gap:
            logger.debug(
                "Accepted via confident gap",
                extra={
                    "speaker": best_name,
                    "similarity": round(best_sim, 3),
                    "gap": round(gap, 3),
                },
            )
            return best_name, best_sim

        return None, best_sim
