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
LOGS_DIR = DATA_DIR / "logs"
TEAMS_DIR = DATA_DIR / "teams"

# Legacy paths (used for migration detection)
VOICEPRINTS_DIR = DATA_DIR / "voiceprints"
SAMPLES_DIR = DATA_DIR / "samples"
ENROLLED_SAMPLES_DIR = SAMPLES_DIR / "enrolled"

# Database
DB_PATH = DATA_DIR / "meetscribe.db"

# Server config file (global, shared across teams)
SERVERS_CONFIG = DATA_DIR / "servers.yaml"


def get_team_samples_dir(team_name: str) -> Path:
    """Get the samples root directory for a team."""
    return TEAMS_DIR / team_name / "samples"


def get_team_enrolled_dir(team_name: str) -> Path:
    """Get the enrolled samples directory for a team."""
    return TEAMS_DIR / team_name / "samples" / "enrolled"


def get_team_unknown_dir(team_name: str) -> Path:
    """Get the unknown samples directory for a team."""
    return TEAMS_DIR / team_name / "samples" / "unknown"


def ensure_dirs() -> None:
    """Create global directories (not team-specific)."""
    for d in [DATA_DIR, CACHE_DIR, LOGS_DIR, TEAMS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_team_dirs(team_name: str) -> None:
    """Create directories for a specific team."""
    for d in [
        get_team_samples_dir(team_name),
        get_team_enrolled_dir(team_name),
        get_team_unknown_dir(team_name),
    ]:
        d.mkdir(parents=True, exist_ok=True)
