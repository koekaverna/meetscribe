"""Remote transcription via speaches API (OpenAI-compatible)."""

import io
import logging
import wave
from pathlib import Path

import httpx
from tqdm import tqdm

from .models import SpeechSegment, TranscriptSegment, merge_close_segments

logger = logging.getLogger(__name__)


class RemoteTranscriber:
    """Single-server transcription client using OpenAI-compatible /v1/audio/transcriptions."""

    def __init__(self, server_url: str, timeout: float, model: str):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.model = model

    def transcribe(
        self,
        audio_path: Path,
        language: str,
    ) -> list[TranscriptSegment]:
        """Transcribe an audio file via the remote API."""
        with open(audio_path, "rb") as f:
            return self._transcribe_request(audio_path.name, f.read(), language)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str,
        filename: str = "chunk.wav",
    ) -> list[TranscriptSegment]:
        """Transcribe WAV bytes via the remote API."""
        return self._transcribe_request(filename, audio_bytes, language)

    def _transcribe_request(
        self,
        filename: str,
        audio_bytes: bytes,
        language: str,
    ) -> list[TranscriptSegment]:
        """Send transcription request and parse response."""
        response = httpx.post(
            f"{self.server_url}/v1/audio/transcriptions",
            files={"file": (filename, audio_bytes, "audio/wav")},
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

        Loads audio into memory, merges close segments into larger chunks,
        slices each chunk as WAV bytes, and transcribes via remote API.

        Args:
            audio_path: Path to the full audio track (16kHz mono WAV).
            segments: Diarized speech segments with speaker labels.

        Returns:
            List of TranscriptSegment with text, timestamps, and speakers.
        """
        if not segments:
            return []

        # Load full audio into memory once
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            raw_frames = wf.readframes(wf.getnframes())

        merged = merge_close_segments(segments, self.max_gap_ms, self.max_chunk_ms)
        results: list[TranscriptSegment] = []

        total_ms = sum(s.duration_ms for s in merged)
        pbar = tqdm(total=total_ms, unit="ms", unit_scale=True, desc="  Transcribing", leave=False)

        frame_size = sample_width * n_channels

        for i, chunk in enumerate(merged):
            client = self.clients[i % len(self.clients)]

            # Slice audio in memory
            start_sample = chunk.start_ms * sample_rate // 1000
            end_sample = chunk.end_ms * sample_rate // 1000
            chunk_frames = raw_frames[start_sample * frame_size : end_sample * frame_size]

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(chunk_frames)

            transcript_segs = client.transcribe_bytes(buf.getvalue(), self.language)

            for seg in transcript_segs:
                # Offset timestamps to absolute position
                seg.start_ms += chunk.start_ms
                seg.end_ms += chunk.start_ms
                # Assign speaker from the diarization chunk
                seg.speaker = self._find_speaker(seg.start_ms, seg.end_ms, segments)
                results.append(seg)

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
