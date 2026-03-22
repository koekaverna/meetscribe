"""Configuration and paths management."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_data_dir() -> Path:
    """Get data directory for MeetScribe.

    Override with MEETSCRIBE_DATA_DIR environment variable.
    Default: platform-specific location.
    """
    override = os.environ.get("MEETSCRIBE_DATA_DIR")
    if override:
        return Path(override).resolve()

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / "meetscribe"


def get_tmp_dir() -> Path:
    """Get temp directory for MeetScribe.

    Override with MEETSCRIBE_TMP_DIR environment variable.
    Default: DATA_DIR/tmp (ensures temp files are on the same disk as data).
    """
    override = os.environ.get("MEETSCRIBE_TMP_DIR")
    if override:
        return Path(override).resolve()
    return get_data_dir() / "tmp"


# Directory paths
DATA_DIR = get_data_dir()
TMP_DIR = get_tmp_dir()
LOGS_DIR = DATA_DIR / "logs"
TEAMS_DIR = DATA_DIR / "teams"

# Upload limits
MAX_UPLOAD_SIZE = int(os.environ.get("MEETSCRIBE_MAX_UPLOAD_SIZE", 4 * 1024 * 1024 * 1024))  # 4 GB

# Database
DB_PATH = DATA_DIR / "meetscribe.db"

# Application config file
CONFIG_FILE = DATA_DIR / "config.yaml"


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
    for d in [DATA_DIR, TMP_DIR, LOGS_DIR, TEAMS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_team_dirs(team_name: str) -> None:
    """Create directories for a specific team."""
    for d in [
        get_team_samples_dir(team_name),
        get_team_enrolled_dir(team_name),
        get_team_unknown_dir(team_name),
    ]:
        d.mkdir(parents=True, exist_ok=True)
