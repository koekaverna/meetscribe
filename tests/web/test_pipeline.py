"""Tests for PipelineRunner.transcribe() track_num assignment."""

import io
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from meetscribe.pipeline.models import SpeechSegment, TranscriptSegment
from meetscribe.web.services.pipeline import PipelineRunner


def _make_wav(path: Path, duration_s: float = 3.0) -> Path:
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


class TestTranscribeTrackNum:
    """Segments must carry the correct track_num after transcription."""

    def _run_transcribe(self, tmp_path: Path, named: bool) -> list[dict[str, Any]]:
        """Run PipelineRunner.transcribe() with 2 tracks and return output segments."""
        track1 = _make_wav(tmp_path / "track1.wav")
        track2 = _make_wav(tmp_path / "track2.wav")

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_file.side_effect = [
            [TranscriptSegment(0, 1000, "Hello", "Alice")],
            [TranscriptSegment(500, 2000, "Hi", "Bob")],
        ]

        mock_diarization = MagicMock()
        mock_diarization.diarize.side_effect = [
            [SpeechSegment(0, 1000, "Alice")],
            [SpeechSegment(500, 2000, "Bob")],
        ]
        mock_transcriber.transcribe_segments.side_effect = [
            [TranscriptSegment(0, 1000, "Hello", "Alice")],
            [TranscriptSegment(500, 2000, "Hi", "Bob")],
        ]

        mock_team_ctx = MagicMock()

        runner = PipelineRunner()
        runner._cfg = MagicMock()
        runner._cfg.transcription.language = "en"

        track_speakers: dict[int, str | None]
        if named:
            track_speakers = {1: "Alice", 2: "Bob"}
        else:
            track_speakers = {1: None, 2: None}

        with (
            patch.object(runner, "_resolve", return_value=mock_team_ctx),
            patch.object(runner, "_create_diarization", return_value=mock_diarization),
            patch.object(runner, "_create_transcriber", return_value=mock_transcriber),
        ):
            results = list(runner.transcribe([track1, track2], track_speakers))

        final = results[-1]
        assert "segments" in final
        segments: list[dict[str, Any]] = final["segments"]
        return segments

    def test_named_tracks_get_correct_track_num(self, tmp_path: Path) -> None:
        segments = self._run_transcribe(tmp_path, named=True)
        assert len(segments) == 2
        # Sorted by start_ms: track1 seg (0ms) first, track2 seg (500ms) second
        assert segments[0]["track_num"] == 1
        assert segments[1]["track_num"] == 2

    def test_diarized_tracks_get_correct_track_num(self, tmp_path: Path) -> None:
        segments = self._run_transcribe(tmp_path, named=False)
        assert len(segments) == 2
        assert segments[0]["track_num"] == 1
        assert segments[1]["track_num"] == 2


class TestOpenSpaceFilter:
    """Open-space tracks must fail loudly when the assigned name matches no voice."""

    def _run(self, tmp_path: Path, assigned: str, diarized: list[SpeechSegment]) -> list[dict]:
        track = _make_wav(tmp_path / "track1.wav")

        mock_diarization = MagicMock()
        mock_diarization.diarize.return_value = diarized
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_segments.return_value = [
            TranscriptSegment(0, 1000, "Hello", assigned)
        ]

        runner = PipelineRunner()
        runner._cfg = MagicMock()
        runner._cfg.transcription.language = "en"

        with (
            patch.object(runner, "_resolve", return_value=MagicMock()),
            patch.object(runner, "_create_diarization", return_value=mock_diarization),
            patch.object(runner, "_create_transcriber", return_value=mock_transcriber),
        ):
            return list(runner.transcribe([track], {1: assigned}, track_open_space={1: True}))

    def test_mismatched_name_raises_with_found_speakers(self, tmp_path: Path) -> None:
        diarized = [
            SpeechSegment(0, 1000, "Стародубцев Евгений"),
            SpeechSegment(1000, 2000, "Unknown-1"),
        ]
        with pytest.raises(ValueError) as exc:
            self._run(tmp_path, "Евгений Стародубцев", diarized)
        msg = str(exc.value)
        assert "'Евгений Стародубцев' does not match any voice" in msg
        assert "Стародубцев Евгений" in msg
        assert "Unknown-1" in msg

    def test_matching_name_transcribes_track(self, tmp_path: Path) -> None:
        diarized = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1000, 2000, "Unknown-1"),
        ]
        results = self._run(tmp_path, "Alice", diarized)
        segments = results[-1]["segments"]
        assert len(segments) == 1
        assert segments[0]["speaker"] == "Alice"

    def test_track_without_speech_is_skipped_quietly(self, tmp_path: Path) -> None:
        results = self._run(tmp_path, "Alice", [])
        messages = [r.get("message", "") for r in results]
        assert any("No speech found" in m for m in messages)
