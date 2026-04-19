"""Tests for pipeline/embeddings.py — EmbeddingExtractor, cosine_similarity, compute_voiceprint."""

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetscribe.pipeline.embeddings import (
    EmbeddingExtractor,
    compute_voiceprint,
    cosine_similarity,
    slice_wav,
)
from meetscribe.pipeline.models import SpeechSegment


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_both_zero(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_similar_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.95, 0.05, 0.0, 0.0]
        sim = cosine_similarity(a, b)
        assert sim > 0.99
        assert sim < 1.0


class TestComputeVoiceprint:
    def test_single_embedding(self):
        ext = MagicMock()
        emb = [1.0, 2.0, 3.0]
        ext.extract_from_file.return_value = emb
        result = compute_voiceprint(ext, [Path("a.wav")])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_averages_multiple(self):
        ext = MagicMock()
        ext.extract_from_file.side_effect = [
            [1.0, 0.0, 4.0],
            [3.0, 2.0, 0.0],
        ]
        result = compute_voiceprint(ext, [Path("a.wav"), Path("b.wav")])
        assert result == pytest.approx([2.0, 1.0, 2.0])
        assert ext.extract_from_file.call_count == 2


class TestEmbeddingExtractorInit:
    def test_strips_trailing_slash(self):
        ext = EmbeddingExtractor("http://host:8000/", timeout=10.0, min_duration_ms=1500, model="m")
        assert ext.server_url == "http://host:8000"

    def test_no_trailing_slash(self):
        ext = EmbeddingExtractor("http://host:8000", timeout=10.0, min_duration_ms=1500, model="m")
        assert ext.server_url == "http://host:8000"

    def test_stores_params(self):
        ext = EmbeddingExtractor("http://h", timeout=30.0, min_duration_ms=2000, model="test-model")
        assert ext.timeout == 30.0
        assert ext.min_duration_ms == 2000
        assert ext.model == "test-model"


class TestEmbeddingExtractorExtract:
    def test_request_params(self):
        ext = EmbeddingExtractor("http://host:8000", timeout=10.0, min_duration_ms=1500, model="m1")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.embeddings.httpx.post", return_value=mock_resp) as m:
            result = ext.extract(b"wav_data", "test.wav")

        assert result == [0.1, 0.2]
        call_kwargs = m.call_args
        assert call_kwargs[0][0] == "http://host:8000/v1/audio/speech/embedding"
        files = call_kwargs[1]["files"]
        assert files["file"][0] == "test.wav"
        assert files["file"][1] == b"wav_data"
        assert files["file"][2] == "audio/wav"
        data = call_kwargs[1]["data"]
        assert data["model"] == "m1"
        assert call_kwargs[1]["timeout"] == 10.0


class TestSliceWav:
    def _make_raw_frames(self, sample_rate: int, duration_s: float) -> bytes:
        """Create raw PCM frames (16-bit mono)."""
        n_frames = int(sample_rate * duration_s)
        # Use sequential values so we can verify slicing
        frames = bytearray()
        for i in range(n_frames):
            # Low byte = i % 256, high byte = (i // 256) % 256
            frames.extend((i % 65536).to_bytes(2, "little"))
        return bytes(frames)

    def test_correct_frame_count(self):
        sample_rate = 16000
        raw = self._make_raw_frames(sample_rate, 3.0)
        seg = SpeechSegment(start_ms=1000, end_ms=2000)

        wav_bytes = slice_wav(raw, sample_rate, 2, seg)

        # Parse the WAV to check frame count
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 16000  # 1 second at 16kHz
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == sample_rate

    def test_correct_offset(self):
        sample_rate = 16000
        raw = self._make_raw_frames(sample_rate, 3.0)
        seg = SpeechSegment(start_ms=500, end_ms=1500)

        wav_bytes = slice_wav(raw, sample_rate, 2, seg)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        # First frame should be at sample 500ms * 16000 / 1000 = 8000
        # The raw PCM at that position: sample 8000 → bytes at 8000*2 = 16000
        expected_start = raw[8000 * 2 : 8000 * 2 + 2]
        assert frames[:2] == expected_start

    def test_header_format(self):
        sample_rate = 16000
        raw = self._make_raw_frames(sample_rate, 1.0)
        seg = SpeechSegment(start_ms=0, end_ms=500)

        wav_bytes = slice_wav(raw, sample_rate, 2, seg)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
