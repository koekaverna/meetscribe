"""Remote Voice Activity Detection via speaches API."""

import logging
from pathlib import Path

import httpx

from .models import SpeechSegment

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """VAD using speaches POST /v1/audio/speech/timestamps endpoint."""

    def __init__(self, server_url: str, timeout: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        """Detect speech segments in an audio file.

        Args:
            audio_path: Path to audio file (WAV).

        Returns:
            List of SpeechSegment with start/end times (speaker=None).
        """
        logger.info("VAD: processing %s", audio_path.name)

        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{self.server_url}/v1/audio/speech/timestamps",
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.error("VAD failed: %s %s", response.status_code, response.text)
            response.raise_for_status()

        result = response.json()
        segments = [SpeechSegment(start_ms=seg["start"], end_ms=seg["end"]) for seg in result]

        logger.info("VAD returned %d segments", len(segments))
        return segments
