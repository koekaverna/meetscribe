"""FFmpeg utilities for audio extraction and conversion.

All functions use system FFmpeg binary.
"""

import re
import shutil
import subprocess
from pathlib import Path

FFMPEG_BIN = "ffmpeg"

FFMPEG_INSTALL_HELP = """
FFmpeg is required but not found. Install it:

  Windows:   winget install "FFmpeg (Shared)"
  macOS:     brew install ffmpeg
  Linux:     sudo apt install ffmpeg

After installation, restart your terminal.
""".strip()


class FFmpegNotFoundError(RuntimeError):
    """Raised when FFmpeg is not available."""

    def __init__(self):
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
        [FFMPEG_BIN, "-i", str(file_path), "-hide_banner"],
        capture_output=True,
        text=True,
    )
    indices = []
    for m in re.finditer(r"Stream #0:(\d+).*?: Audio:", result.stderr):
        indices.append(int(m.group(1)))
    return indices


def extract_audio(video_path: Path, output_path: Path, track_index: int) -> Path:
    """Extract audio track from video file as 16kHz mono WAV."""
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
        raise RuntimeError(f"Failed to extract track {track_index}: {result.stderr}")
    return output_path


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert audio file to 16kHz mono WAV."""
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
        raise RuntimeError(f"Failed to convert {input_path.name}: {result.stderr}")
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
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract segment {start_ms}-{end_ms}ms: {result.stderr}")
    return output_path
