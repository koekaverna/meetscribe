"""FFmpeg utilities for audio extraction and conversion.

All functions use system FFmpeg binary.
"""

import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path

from meetscribe.errors import PipelineError

logger = logging.getLogger(__name__)

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

FFMPEG_INSTALL_HELP = """
FFmpeg is required but not found. Install it:

  Windows:   winget install "FFmpeg (Shared)"
  macOS:     brew install ffmpeg
  Linux:     sudo apt install ffmpeg

After installation, restart your terminal.
""".strip()


class FFmpegNotFoundError(RuntimeError):
    """Raised when FFmpeg is not available."""

    def __init__(self) -> None:
        super().__init__(FFMPEG_INSTALL_HELP)


def check_ffmpeg() -> None:
    """Check if FFmpeg is available in PATH.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not found.
    """
    if shutil.which(FFMPEG_BIN) is None:
        raise FFmpegNotFoundError()


def probe_audio_tracks(file_path: Path) -> list[int]:
    """Return list of audio stream indices in the file."""
    result = subprocess.run(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )
    tracks = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    logger.info(
        "Audio tracks probed",
        extra={"file": file_path.name, "tracks": len(tracks)},
    )
    return tracks


def extract_audio(video_path: Path, output_path: Path, track_index: int) -> Path:
    """Extract audio track from video file as 16kHz mono WAV."""
    t0 = time.perf_counter()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{track_index}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Failed to extract track {track_index}: {result.stderr}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Audio track extracted",
        extra={
            "file": video_path.name,
            "track_index": track_index,
            "output": output_path.name,
            "elapsed_ms": round(elapsed_ms),
        },
    )
    return output_path


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert audio file to 16kHz mono WAV."""
    t0 = time.perf_counter()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Failed to convert {input_path.name}: {result.stderr}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Audio converted to WAV",
        extra={
            "file": input_path.name,
            "output": output_path.name,
            "elapsed_ms": round(elapsed_ms),
        },
    )
    return output_path


async def probe_audio_tracks_async(file_path: Path) -> list[int]:
    """Return list of audio stream indices in the file (async)."""
    proc = await asyncio.create_subprocess_exec(
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    tracks = [int(line.strip()) for line in stdout.decode().strip().split("\n") if line.strip()]
    logger.info(
        "Audio tracks probed",
        extra={"file": file_path.name, "tracks": len(tracks)},
    )
    return tracks


async def extract_audio_async(video_path: Path, output_path: Path, track_index: int) -> Path:
    """Extract audio track from video file as 16kHz mono WAV (async)."""
    t0 = time.perf_counter()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{track_index}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace")
        raise PipelineError(f"Failed to extract track {track_index}: {err}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Audio track extracted",
        extra={
            "file": video_path.name,
            "track_index": track_index,
            "output": output_path.name,
            "elapsed_ms": round(elapsed_ms),
        },
    )
    return output_path


async def convert_to_wav_async(input_path: Path, output_path: Path) -> Path:
    """Convert audio file to 16kHz mono WAV (async)."""
    t0 = time.perf_counter()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace")
        raise PipelineError(f"Failed to convert {input_path.name}: {err}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Audio converted to WAV",
        extra={
            "file": input_path.name,
            "output": output_path.name,
            "elapsed_ms": round(elapsed_ms),
        },
    )
    return output_path


def extract_segment(audio_path: Path, output_path: Path, start_ms: int, end_ms: int) -> Path:
    """Extract a time segment from an audio file as 16kHz mono WAV.

    Args:
        audio_path: Source audio file.
        output_path: Output WAV file.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.

    Returns:
        Path to the extracted segment.
    """
    start_sec = start_ms / 1000
    duration_sec = (end_ms - start_ms) / 1000

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(audio_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Failed to extract segment {start_ms}-{end_ms}ms: {result.stderr}")
    return output_path
