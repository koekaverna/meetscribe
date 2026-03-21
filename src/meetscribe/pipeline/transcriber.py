"""Remote transcription via speaches API (OpenAI-compatible)."""

import logging
import tempfile
from pathlib import Path

import httpx
from tqdm import tqdm

from .. import config
from .audio import extract_segment
from .models import SpeechSegment, TranscriptSegment, merge_close_segments

logger = logging.getLogger(__name__)


class RemoteTranscriber:
    """Single-server transcription client using OpenAI-compatible /v1/audio/transcriptions."""

    def __init__(self, server_url: str, timeout: float, model: str):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.model = self._detect_model() or model

    def _detect_model(self) -> str | None:
        """Try to detect available ASR model from server."""
        try:
            resp = httpx.get(f"{self.server_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    if m.get("task") == "automatic-speech-recognition":
                        return m.get("id")
        except Exception:
            pass
        return None

    def transcribe(
        self,
        audio_path: Path,
        language: str,
    ) -> list[TranscriptSegment]:
        """Transcribe an audio file via the remote API.

        Returns segments with timestamps relative to the audio file start.
        """
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{self.server_url}/v1/audio/transcriptions",
                files={"file": (audio_path.name, f, "audio/wav")},
                data={
                    "model": self.model,
                    "language": language,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

        result = response.json()
        segments = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start_ms=int(seg["start"] * 1000),
                        end_ms=int(seg["end"] * 1000),
                        text=text,
                    )
                )
        return segments


class Transcriber:
    """Distribute transcription of diarized segments across remote servers.

    Takes diarized SpeechSegments (with speaker labels) and transcribes each chunk
    via remote speaches API servers.
    """

    def __init__(
        self,
        server_urls: list[str],
        language: str,
        timeout: float,
        model: str,
        max_gap_ms: int,
        max_chunk_ms: int,
    ):
        if not server_urls:
            raise ValueError("At least one transcription server URL is required")
        self.clients = [RemoteTranscriber(url, timeout, model) for url in server_urls]
        self.language = language
        self.max_gap_ms = max_gap_ms
        self.max_chunk_ms = max_chunk_ms

    def transcribe_file(
        self,
        audio_path: Path,
        speaker: str | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe an entire audio file without segmentation.

        Useful for named tracks where the speaker is already known.
        Uses a longer timeout since the file may be large.
        """
        client = self.clients[0]
        # Use longer timeout for whole-file transcription
        orig_timeout = client.timeout
        client.timeout = max(orig_timeout, 600.0)
        try:
            segments = client.transcribe(audio_path, self.language)
        finally:
            client.timeout = orig_timeout
        for seg in segments:
            seg.speaker = speaker
        return segments

    def transcribe_segments(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
    ) -> list[TranscriptSegment]:
        """Transcribe speech segments from an audio file.

        Merges close segments into larger chunks, extracts audio via FFmpeg,
        transcribes each chunk, and maps speakers back.

        Args:
            audio_path: Path to the full audio track.
            segments: Diarized speech segments with speaker labels.

        Returns:
            List of TranscriptSegment with text, timestamps, and speakers.
        """
        if not segments:
            return []

        merged = merge_close_segments(segments, self.max_gap_ms, self.max_chunk_ms)
        results: list[TranscriptSegment] = []

        total_ms = sum(s.duration_ms for s in merged)
        pbar = tqdm(
            total=total_ms, unit="ms", unit_scale=True, desc="  Transcribing", leave=False
        )

        for i, chunk in enumerate(merged):
            client = self.clients[i % len(self.clients)]

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=config.TMP_DIR
            ) as tmp:
                chunk_path = Path(tmp.name)

            try:
                extract_segment(audio_path, chunk_path, chunk.start_ms, chunk.end_ms)
                transcript_segs = client.transcribe(chunk_path, self.language)

                for seg in transcript_segs:
                    # Offset timestamps to absolute position
                    seg.start_ms += chunk.start_ms
                    seg.end_ms += chunk.start_ms
                    # Assign speaker from the diarization chunk
                    seg.speaker = self._find_speaker(seg.start_ms, seg.end_ms, segments)
                    results.append(seg)
            finally:
                chunk_path.unlink(missing_ok=True)

            pbar.update(chunk.duration_ms)

        pbar.close()
        return results

    @staticmethod
    def _find_speaker(start_ms: int, end_ms: int, segments: list[SpeechSegment]) -> str:
        """Find speaker with maximum time overlap."""
        best_speaker = "Unknown"
        best_overlap = 0

        for seg in segments:
            overlap = max(0, min(end_ms, seg.end_ms) - max(start_ms, seg.start_ms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker or "Unknown"

        return best_speaker
