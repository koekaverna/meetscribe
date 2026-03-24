"""Remote Voice Activity Detection via speaches API."""

import logging
import time
from pathlib import Path

import httpx

from meetscribe.errors import SpeachesAPIError, speaches_retry

from .models import SpeechSegment

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """VAD using speaches POST /v1/audio/speech/timestamps endpoint."""

    def __init__(
        self,
        server_url: str,
        timeout: float,
        min_silence_duration_ms: int = 1200,
        speech_pad_ms: int = 30,
        threshold: float = 0.5,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.threshold = threshold

    @speaches_retry
    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        """Detect speech segments in an audio file.

        Args:
            audio_path: Path to audio file (WAV).

        Returns:
            List of SpeechSegment with start/end times (speaker=None).
        """
        logger.info("VAD started", extra={"file": audio_path.name})
        t0 = time.perf_counter()
        endpoint = f"{self.server_url}/v1/audio/speech/timestamps"

        try:
            with open(audio_path, "rb") as f:
                response = httpx.post(
                    endpoint,
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={
                        "min_silence_duration_ms": self.min_silence_duration_ms,
                        "speech_pad_ms": self.speech_pad_ms,
                        "threshold": self.threshold,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SpeachesAPIError(
                f"VAD failed: {e.response.status_code}",
                status_code=e.response.status_code,
                endpoint=endpoint,
                detail=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise SpeachesAPIError(
                f"VAD connection error: {e}",
                endpoint=endpoint,
            ) from e

        result = response.json()
        segments = [SpeechSegment(start_ms=seg["start"], end_ms=seg["end"]) for seg in result]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        audio_duration_ms = segments[-1].end_ms if segments else 0
        logger.info(
            "VAD completed",
            extra={
                "file": audio_path.name,
                "segments": len(segments),
                "audio_duration_ms": audio_duration_ms,
                "elapsed_ms": round(elapsed_ms),
            },
        )
        return segments
