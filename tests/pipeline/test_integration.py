"""Integration tests for the diarization and transcription pipeline with mocked HTTP."""

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from meetscribe.pipeline.diarization import DiarizationPipeline
from meetscribe.pipeline.models import SpeechSegment, TranscriptSegment
from meetscribe.pipeline.transcriber import Transcriber


def _make_wav(path: Path, duration_s: float = 3.0) -> Path:
    """Create a WAV file for testing."""
    sample_rate = 16000
    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    path.write_bytes(buf.getvalue())
    return path


class TestTranscriberFindSpeaker:
    def test_best_overlap_wins(self):
        segments = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(800, 2000, "Bob"),
        ]
        # Query 900-1100: overlaps Alice by 100ms, Bob by 200ms
        speaker = Transcriber._find_speaker(900, 1100, segments)
        assert speaker == "Bob"

    def test_no_overlap_returns_unknown(self):
        segments = [SpeechSegment(0, 500, "Alice")]
        speaker = Transcriber._find_speaker(1000, 2000, segments)
        assert speaker == "Unknown"

    def test_exact_match(self):
        segments = [SpeechSegment(1000, 2000, "Alice")]
        speaker = Transcriber._find_speaker(1000, 2000, segments)
        assert speaker == "Alice"


class TestTranscriberTranscribeSegments:
    def test_offsets_timestamps_and_assigns_speakers(self, tmp_path: Path):
        audio = _make_wav(tmp_path / "test.wav", duration_s=5.0)

        diarized_segments = [
            SpeechSegment(1000, 3000, "Alice"),
            SpeechSegment(3500, 4500, "Bob"),
        ]

        # Mock remote transcriber to return segments with relative timestamps
        def mock_transcribe_bytes(audio_bytes, language, filename="chunk.wav"):
            return [TranscriptSegment(start_ms=0, end_ms=500, text="Hello")]

        mock_client = MagicMock()
        mock_client.transcribe_bytes = mock_transcribe_bytes

        transcriber = Transcriber(
            server_urls=["http://fake:8000"],
            language="en",
            timeout=10.0,
            model="test",
            max_gap_ms=500,
            max_chunk_ms=30000,
        )
        transcriber.clients = [mock_client]

        results = transcriber.transcribe_segments(audio, diarized_segments)

        assert len(results) >= 1
        # First result should have offset applied (chunk starts at 1000ms)
        first = results[0]
        assert first.start_ms == 1000  # 0 + 1000 offset
        assert first.end_ms == 1500  # 500 + 1000 offset
        assert first.speaker == "Alice"


class TestDiarizationPipelineEndToEnd:
    def test_full_flow(self, tmp_path: Path):
        audio = _make_wav(tmp_path / "test.wav", duration_s=3.0)

        alice_emb = [1.0] + [0.0] * 255
        bob_emb = [0.0] + [1.0] + [0.0] * 254

        # Mock embedding extraction — first call returns alice, second returns bob
        emb_calls = iter([alice_emb, bob_emb])

        def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "diarization" in url:
                resp.json.return_value = {
                    "duration": 3.0,
                    "segments": [
                        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
                        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
                    ],
                }
            else:
                # embedding endpoint
                resp.json.return_value = {"data": [{"embedding": next(emb_calls)}]}
            return resp

        voiceprints = {"Alice": alice_emb, "Bob": bob_emb}

        pipeline = DiarizationPipeline(
            diarization_url="http://fake:8000",
            embedding_url="http://fake:8000",
            voiceprints=voiceprints,
            threshold=0.8,
            confident_gap=0.2,
            min_threshold=0.4,
            diarization_timeout=10.0,
            embedding_timeout=10.0,
            min_duration_ms=500,
            embedding_model="test-model",
        )

        with (
            patch("meetscribe.pipeline.diarization.httpx.post", side_effect=mock_post),
            patch("meetscribe.pipeline.embeddings.httpx.post", side_effect=mock_post),
        ):
            result = pipeline.diarize(audio)

        assert len(result) == 2
        assert result[0].speaker == "Alice"
        assert result[1].speaker == "Bob"
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1500
        assert result[1].start_ms == 1500
        assert result[1].end_ms == 3000

    def test_no_segments_returns_empty(self, tmp_path: Path):
        audio = _make_wav(tmp_path / "test.wav", duration_s=1.0)

        def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"duration": 1.0, "segments": []}
            return resp

        pipeline = DiarizationPipeline(
            diarization_url="http://fake:8000",
            embedding_url="http://fake:8000",
            voiceprints={},
            threshold=0.8,
            confident_gap=0.2,
            min_threshold=0.4,
            diarization_timeout=10.0,
            embedding_timeout=10.0,
            min_duration_ms=500,
            embedding_model="test-model",
        )

        with patch("meetscribe.pipeline.diarization.httpx.post", side_effect=mock_post):
            result = pipeline.diarize(audio)

        assert result == []

    def test_unknown_speakers_when_no_voiceprints(self, tmp_path: Path):
        audio = _make_wav(tmp_path / "test.wav", duration_s=3.0)

        def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "diarization" in url:
                resp.json.return_value = {
                    "duration": 3.0,
                    "segments": [
                        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
                        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
                    ],
                }
            else:
                resp.json.return_value = {"data": [{"embedding": [0.1] * 256}]}
            return resp

        pipeline = DiarizationPipeline(
            diarization_url="http://fake:8000",
            embedding_url="http://fake:8000",
            voiceprints={},
            threshold=0.8,
            confident_gap=0.2,
            min_threshold=0.4,
            diarization_timeout=10.0,
            embedding_timeout=10.0,
            min_duration_ms=500,
            embedding_model="test-model",
        )

        with (
            patch("meetscribe.pipeline.diarization.httpx.post", side_effect=mock_post),
            patch("meetscribe.pipeline.embeddings.httpx.post", side_effect=mock_post),
        ):
            result = pipeline.diarize(audio)

        assert len(result) == 2
        assert result[0].speaker.startswith("Unknown-")
        assert result[1].speaker.startswith("Unknown-")
        assert result[0].speaker != result[1].speaker
