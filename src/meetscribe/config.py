"""Configuration and paths management."""

import os
import sys
from pathlib import Path


def get_data_dir() -> Path:
    """Get platform-specific data directory for MeetScribe.

    Windows: %LOCALAPPDATA%/meetscribe
    macOS: ~/Library/Application Support/meetscribe
    Linux: ~/.local/share/meetscribe (XDG_DATA_HOME)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / "meetscribe"


def get_cache_dir() -> Path:
    """Get platform-specific cache directory for MeetScribe.

    Windows: %LOCALAPPDATA%/meetscribe/cache
    macOS: ~/Library/Caches/meetscribe
    Linux: ~/.cache/meetscribe (XDG_CACHE_HOME)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "meetscribe" / "cache"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "meetscribe"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "meetscribe"


# Directory paths
DATA_DIR = get_data_dir()
CACHE_DIR = get_cache_dir()
MODELS_DIR = CACHE_DIR / "models"
VOICEPRINTS_DIR = DATA_DIR / "voiceprints"
SAMPLES_DIR = DATA_DIR / "samples"
ENROLLED_SAMPLES_DIR = SAMPLES_DIR / "enrolled"
LOGS_DIR = DATA_DIR / "logs"
SESSIONS_DIR = CACHE_DIR / "sessions"



def ensure_dirs() -> None:
    """Create all necessary directories."""
    for d in [
        DATA_DIR,
        CACHE_DIR,
        MODELS_DIR,
        VOICEPRINTS_DIR,
        SAMPLES_DIR,
        ENROLLED_SAMPLES_DIR,
        LOGS_DIR,
        SESSIONS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
