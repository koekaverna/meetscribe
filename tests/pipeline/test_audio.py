"""Tests for pipeline/audio.py — FFmpeg command construction and error handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meetscribe.errors import PipelineError
from meetscribe.pipeline.audio import (
    FFmpegNotFoundError,
    check_ffmpeg,
    convert_to_wav,
    convert_to_wav_async,
    extract_audio,
    extract_audio_async,
    extract_segment,
    probe_audio_tracks,
    probe_audio_tracks_async,
)


class TestCheckFfmpeg:
    def test_available(self):
        with patch("meetscribe.pipeline.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            check_ffmpeg()

    def test_missing_raises(self):
        with patch("meetscribe.pipeline.audio.shutil.which", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
                check_ffmpeg()

    def test_error_message_contains_install_help(self):
        with patch("meetscribe.pipeline.audio.shutil.which", return_value=None):
            with pytest.raises(FFmpegNotFoundError, match="FFmpeg is required"):
                check_ffmpeg()


class TestProbeAudioTracks:
    def test_parses_stdout(self):
        mock_result = MagicMock()
        mock_result.stdout = "0\n1\n"
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            tracks = probe_audio_tracks(Path("/fake/video.mkv"))
        assert tracks == [0, 1]

    def test_single_track(self):
        mock_result = MagicMock()
        mock_result.stdout = "1\n"
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            tracks = probe_audio_tracks(Path("/fake/video.mkv"))
        assert tracks == [1]

    def test_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            tracks = probe_audio_tracks(Path("/fake/video.mkv"))
        assert tracks == []

    def test_exact_command_and_kwargs(self):
        mock_result = MagicMock()
        mock_result.stdout = "0\n"
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            probe_audio_tracks(Path("/fake/video.mkv"))

        cmd = m.call_args[0][0]
        kwargs = m.call_args[1]
        assert cmd == [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            "/fake/video.mkv",
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    def test_failure_raises_pipeline_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "probe error"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to probe"):
                probe_audio_tracks(Path("/fake/video.mkv"))


class TestExtractAudio:
    def test_exact_command_and_kwargs(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "out.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            result = extract_audio(Path("/fake/video.mkv"), out, track_index=2)

        cmd = m.call_args[0][0]
        kwargs = m.call_args[1]
        assert cmd == [
            "ffmpeg",
            "-y",
            "-i",
            "/fake/video.mkv",
            "-map",
            "0:2",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert result == out

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to extract track"):
                extract_audio(Path("/fake/video.mkv"), tmp_path / "out.wav", track_index=0)


class TestConvertToWav:
    def test_exact_command_and_kwargs(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "out.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            result = convert_to_wav(Path("/fake/input.mp3"), out)

        cmd = m.call_args[0][0]
        kwargs = m.call_args[1]
        assert cmd == [
            "ffmpeg",
            "-y",
            "-i",
            "/fake/input.mp3",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert result == out

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to convert"):
                convert_to_wav(Path("/fake/input.mp3"), tmp_path / "out.wav")


class TestExtractSegment:
    def test_exact_command_with_time_args(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "seg.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            result = extract_segment(Path("/fake/audio.wav"), out, start_ms=5000, end_ms=8000)

        cmd = m.call_args[0][0]
        kwargs = m.call_args[1]
        assert cmd == [
            "ffmpeg",
            "-y",
            "-ss",
            "5.0",
            "-t",
            "3.0",
            "-i",
            "/fake/audio.wav",
            "-c",
            "copy",
            str(out),
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert result == out

    def test_fractional_time_args(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            extract_segment(Path("/f.wav"), tmp_path / "o.wav", start_ms=1500, end_ms=3500)
        cmd = m.call_args[0][0]
        assert cmd[3] == "1.5"  # -ss value
        assert cmd[5] == "2.0"  # -t value

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "err"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to extract segment"):
                extract_segment(Path("/f.wav"), tmp_path / "o.wav", 0, 1000)


class TestProbeAudioTracksAsync:
    def test_parses_stdout_and_uses_correct_args(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"0\n1\n", b"")
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            tracks = asyncio.run(probe_audio_tracks_async(Path("/fake/video.mkv")))

        assert tracks == [0, 1]
        call_args = m.call_args[0]
        assert call_args == (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            "/fake/video.mkv",
        )
        call_kwargs = m.call_args[1]
        assert call_kwargs["stdout"] == asyncio.subprocess.PIPE
        assert call_kwargs["stderr"] == asyncio.subprocess.PIPE

    def test_empty_output(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            tracks = asyncio.run(probe_audio_tracks_async(Path("/fake/video.mkv")))
        assert tracks == []

    def test_failure_raises(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(PipelineError, match="Failed to probe"):
                asyncio.run(probe_audio_tracks_async(Path("/fake/video.mkv")))


class TestExtractAudioAsync:
    def test_exact_command_and_pipe_kwargs(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        out = tmp_path / "out.wav"
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            result = asyncio.run(extract_audio_async(Path("/fake/video.mkv"), out, 2))

        call_args = m.call_args[0]
        assert call_args == (
            "ffmpeg",
            "-y",
            "-i",
            "/fake/video.mkv",
            "-map",
            "0:2",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        )
        assert m.call_args[1]["stdout"] == asyncio.subprocess.PIPE
        assert m.call_args[1]["stderr"] == asyncio.subprocess.PIPE
        assert result == out

    def test_failure_raises(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"some error")
        mock_proc.returncode = 1
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(PipelineError, match="Failed to extract track"):
                asyncio.run(extract_audio_async(Path("/fake/v.mkv"), tmp_path / "o.wav", 0))


class TestConvertToWavAsync:
    def test_exact_command_and_pipe_kwargs(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        out = tmp_path / "out.wav"
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            result = asyncio.run(convert_to_wav_async(Path("/fake/input.mp3"), out))

        call_args = m.call_args[0]
        assert call_args == (
            "ffmpeg",
            "-y",
            "-i",
            "/fake/input.mp3",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        )
        assert m.call_args[1]["stdout"] == asyncio.subprocess.PIPE
        assert m.call_args[1]["stderr"] == asyncio.subprocess.PIPE
        assert result == out

    def test_failure_raises(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"convert error")
        mock_proc.returncode = 1
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(PipelineError, match="Failed to convert"):
                asyncio.run(convert_to_wav_async(Path("/fake/in.mp3"), tmp_path / "o.wav"))
