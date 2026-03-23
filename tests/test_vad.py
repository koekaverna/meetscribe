"""Tests for pipeline/vad.py — VoiceActivityDetector construction and request params."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from meetscribe.errors import SpeachesAPIError
from meetscribe.pipeline.vad import VoiceActivityDetector

from .conftest import make_wav_file


class TestVadConstructor:
    def test_stores_all_params(self):
        vad = VoiceActivityDetector(
            server_url="http://host:8000",
            timeout=30.0,
            min_silence_duration_ms=1000,
            speech_pad_ms=50,
            threshold=0.6,
        )
        assert vad.server_url == "http://host:8000"
        assert vad.timeout == 30.0
        assert vad.min_silence_duration_ms == 1000
        assert vad.speech_pad_ms == 50
        assert vad.threshold == 0.6

    def test_strips_trailing_slash(self):
        vad = VoiceActivityDetector("http://host:8000/", timeout=10.0)
        assert vad.server_url == "http://host:8000"

    def test_default_params(self):
        vad = VoiceActivityDetector("http://host:8000", timeout=10.0)
        assert vad.min_silence_duration_ms == 1200
        assert vad.speech_pad_ms == 30
        assert vad.threshold == 0.5


class TestVadDetect:
    def test_request_params(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=1.0)
        vad = VoiceActivityDetector(
            "http://host:8000",
            timeout=15.0,
            min_silence_duration_ms=900,
            speech_pad_ms=40,
            threshold=0.7,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"start": 0, "end": 500}]
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.vad.httpx.post", return_value=mock_resp) as m:
            vad.detect(audio)

        call_kwargs = m.call_args
        assert call_kwargs[0][0] == "http://host:8000/v1/audio/speech/timestamps"
        data = call_kwargs[1]["data"]
        assert data["min_silence_duration_ms"] == 900
        assert data["speech_pad_ms"] == 40
        assert data["threshold"] == 0.7
        assert call_kwargs[1]["timeout"] == 15.0

        files = call_kwargs[1]["files"]
        assert files["file"][0] == "test.wav"
        assert files["file"][2] == "audio/wav"

    def test_parses_response(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=1.0)
        vad = VoiceActivityDetector("http://host:8000", timeout=10.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"start": 100, "end": 500},
            {"start": 700, "end": 900},
        ]
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.vad.httpx.post", return_value=mock_resp):
            segments = vad.detect(audio)

        assert len(segments) == 2
        assert segments[0].start_ms == 100
        assert segments[0].end_ms == 500
        assert segments[1].start_ms == 700
        assert segments[1].end_ms == 900

    def test_error_raises(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=1.0)
        vad = VoiceActivityDetector("http://host:8000", timeout=10.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )

        with patch("meetscribe.pipeline.vad.httpx.post", return_value=mock_resp):
            with pytest.raises(SpeachesAPIError, match="VAD failed: 500") as exc_info:
                vad.detect(audio)
            assert exc_info.value.status_code == 500
            assert exc_info.value.is_transient is True
