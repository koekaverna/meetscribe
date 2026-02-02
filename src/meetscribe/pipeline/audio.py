"""FFmpeg utilities for audio extraction and conversion.

All functions use system FFmpeg binary.
"""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

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


def load_audio(file: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and convert to float32 array at specified sample rate.

    Uses FFmpeg to decode any audio format to raw PCM.

    Args:
        file: Path to audio file.
        sr: Target sample rate (default 16000 Hz).

    Returns:
        Audio samples as float32 numpy array, normalized to [-1, 1].

    Raises:
        RuntimeError: If FFmpeg fails to decode the file.
    """
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def probe_audio_tracks(file_path: Path) -> list[int]:
    """Return list of audio stream indices in the file.

    Args:
        file_path: Path to media file.

    Returns:
        List of audio stream indices (e.g., [1, 2] for streams 0:1 and 0:2).
    """
    result = subprocess.run(
        [FFMPEG_BIN, "-i", str(file_path), "-hide_banner"],
        capture_output=True,
        text=True,
    )
    # ffmpeg -i exits with 1 when no output specified, but prints stream info to stderr
    indices = []
    for m in re.finditer(r"Stream #0:(\d+).*?: Audio:", result.stderr):
        indices.append(int(m.group(1)))
    return indices


def extract_audio(video_path: Path, output_path: Path, track_index: int) -> Path:
    """Extract audio track from video file.

    Args:
        video_path: Path to video file.
        output_path: Path for output audio file.
        track_index: Stream index to extract.

    Returns:
        Path to extracted audio file.

    Raises:
        RuntimeError: If extraction fails.
    """
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(video_path),
        "-map", f"0:{track_index}",
        "-ac", "1",
        "-ar", "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract track {track_index}: {result.stderr}")
    return output_path


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert audio file to 16kHz mono WAV.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output WAV file.

    Returns:
        Path to converted file.

    Raises:
        RuntimeError: If conversion fails.
    """
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert {input_path.name}: {result.stderr}")
    return output_path
