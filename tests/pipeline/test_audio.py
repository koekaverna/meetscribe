"""Tests for pipeline/audio.py — FFmpeg command construction and error handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meetscribe.errors import PipelineError
from meetscribe.pipeline.audio import (
    FFMPEG_BIN,
    FFMPEG_INSTALL_HELP,
    FFPROBE_BIN,
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


class TestConstants:
    def test_ffmpeg_bin(self):
        assert FFMPEG_BIN == "ffmpeg"

    def test_ffprobe_bin(self):
        assert FFPROBE_BIN == "ffprobe"

    def test_install_help_is_string(self):
        assert isinstance(FFMPEG_INSTALL_HELP, str)
        assert "FFmpeg" in FFMPEG_INSTALL_HELP


class TestCheckFfmpeg:
    def test_available(self):
        with patch("meetscribe.pipeline.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            check_ffmpeg()  # should not raise

    def test_missing_raises(self):
        with patch("meetscribe.pipeline.audio.shutil.which", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
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

    def test_command_args(self):
        mock_result = MagicMock()
        mock_result.stdout = "0\n"
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            probe_audio_tracks(Path("/fake/video.mkv"))
        args = m.call_args[0][0]
        assert args[0] == "ffprobe"
        assert "-v" in args
        assert "error" in args
        assert "-select_streams" in args
        assert "a" in args
        assert "-show_entries" in args
        assert "stream=index" in args
        assert "-of" in args
        assert "csv=p=0" in args
        assert str(Path("/fake/video.mkv")) in args


class TestExtractAudio:
    def test_command_args(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            result = extract_audio(Path("/fake/video.mkv"), tmp_path / "out.wav", track_index=2)
        args = m.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-y" in args
        assert "-i" in args
        assert str(Path("/fake/video.mkv")) in args
        assert "-map" in args
        assert "0:2" in args
        assert "-ac" in args
        assert "1" in args
        assert "-ar" in args
        assert "16000" in args
        assert str(tmp_path / "out.wav") in args
        assert result == tmp_path / "out.wav"

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to extract track"):
                extract_audio(Path("/fake/video.mkv"), tmp_path / "out.wav", track_index=0)

    def test_returns_output_path(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "out.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            assert extract_audio(Path("/fake/v.mkv"), out, 0) == out


class TestConvertToWav:
    def test_command_args(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            convert_to_wav(Path("/fake/input.mp3"), tmp_path / "out.wav")
        args = m.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-y" in args
        assert "-i" in args
        assert str(Path("/fake/input.mp3")) in args
        assert "-ac" in args
        assert "1" in args
        assert "-ar" in args
        assert "16000" in args
        assert str(tmp_path / "out.wav") in args

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to convert"):
                convert_to_wav(Path("/fake/input.mp3"), tmp_path / "out.wav")

    def test_returns_output_path(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "out.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            assert convert_to_wav(Path("/fake/input.mp3"), out) == out


class TestExtractSegment:
    def test_command_args(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            extract_segment(
                Path("/fake/audio.wav"),
                tmp_path / "seg.wav",
                start_ms=5000,
                end_ms=8000,
            )
        args = m.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-y" in args
        assert "-ss" in args
        # start_sec = 5000 / 1000 = 5.0
        ss_idx = args.index("-ss")
        assert args[ss_idx + 1] == "5.0"
        # duration = (8000 - 5000) / 1000 = 3.0
        t_idx = args.index("-t")
        assert args[t_idx + 1] == "3.0"
        assert "-i" in args
        assert "-c" in args
        assert "copy" in args

    def test_arithmetic_start_sec(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result) as m:
            extract_segment(Path("/f.wav"), tmp_path / "o.wav", start_ms=1500, end_ms=3500)
        args = m.call_args[0][0]
        ss_idx = args.index("-ss")
        assert args[ss_idx + 1] == "1.5"
        t_idx = args.index("-t")
        assert args[t_idx + 1] == "2.0"

    def test_failure_raises(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "err"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(PipelineError, match="Failed to extract segment"):
                extract_segment(Path("/f.wav"), tmp_path / "o.wav", 0, 1000)

    def test_returns_output_path(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        out = tmp_path / "o.wav"
        with patch("meetscribe.pipeline.audio.subprocess.run", return_value=mock_result):
            assert extract_segment(Path("/f.wav"), out, 0, 1000) == out


class TestProbeAudioTracksAsync:
    def test_parses_stdout(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"0\n1\n", b"")
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            tracks = asyncio.run(probe_audio_tracks_async(Path("/fake/video.mkv")))
        assert tracks == [0, 1]
        # Verify ffprobe is first arg
        call_args = m.call_args[0]
        assert call_args[0] == "ffprobe"
        assert "-v" in call_args
        assert "error" in call_args
        assert "-select_streams" in call_args
        assert "a" in call_args
        assert "-show_entries" in call_args
        assert "stream=index" in call_args
        assert "-of" in call_args
        assert "csv=p=0" in call_args

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


class TestExtractAudioAsync:
    def test_command_args(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            result = asyncio.run(
                extract_audio_async(Path("/fake/video.mkv"), tmp_path / "out.wav", 2)
            )
        call_args = m.call_args[0]
        assert call_args[0] == "ffmpeg"
        assert "-y" in call_args
        assert "-i" in call_args
        assert "-map" in call_args
        assert "0:2" in call_args
        assert "-ac" in call_args
        assert "1" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert result == tmp_path / "out.wav"

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
    def test_command_args(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as m:
            result = asyncio.run(
                convert_to_wav_async(Path("/fake/input.mp3"), tmp_path / "out.wav")
            )
        call_args = m.call_args[0]
        assert call_args[0] == "ffmpeg"
        assert "-y" in call_args
        assert "-i" in call_args
        assert "-ac" in call_args
        assert "1" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert result == tmp_path / "out.wav"

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

    def test_returns_output_path(self, tmp_path: Path):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        out = tmp_path / "out.wav"
        with patch(
            "meetscribe.pipeline.audio.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = asyncio.run(convert_to_wav_async(Path("/fake/in.mp3"), out))
        assert result == out
