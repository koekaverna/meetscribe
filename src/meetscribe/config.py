"""Configuration: paths, directories, and YAML application settings."""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .errors import ConfigurationError

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


# ---------------------------------------------------------------------------
# YAML application settings (loaded from CONFIG_FILE)
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """A remote processing server."""

    url: str
    name: str


@dataclass
class VadConfig:
    """VAD endpoint configuration."""

    server: str = ""
    timeout: float = 120.0
    min_silence_duration_ms: int = 1200
    speech_pad_ms: int = 30
    threshold: float = 0.5


@dataclass
class EmbeddingsConfig:
    """Speaker embeddings configuration."""

    server: str = ""
    model: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    timeout: float = 60.0
    threshold: float = 0.6
    min_duration_ms: int = 1500
    unknown_cluster_threshold: float = 0.25
    confident_gap: float = 0.2
    min_threshold: float = 0.45
    max_workers: int = 4


@dataclass
class TranscriptionConfig:
    """Transcription configuration."""

    servers: list[str] = field(default_factory=list)
    model: str = "Systran/faster-whisper-medium"
    language: str = "ru"
    timeout: float = 120.0
    max_gap_ms: int = 500
    max_chunk_ms: int = 30000


@dataclass
class WebConfig:
    """Web UI configuration."""

    host: str = "127.0.0.1"
    port: int = 8080
    session_ttl_days: int = 7
    secure_cookies: bool = False


@dataclass
class AppConfig:
    """Full application configuration loaded from YAML."""

    servers: list[ServerInfo] = field(default_factory=list)
    vad: VadConfig = field(default_factory=VadConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    web: WebConfig = field(default_factory=WebConfig)
    log_level: str = "DEBUG"

    def get_server_url(self, name: str) -> str:
        """Get server URL by name."""
        for s in self.servers:
            if s.name == name:
                return s.url
        raise ConfigurationError(f"Server '{name}' not found in configuration")

    def get_vad_url(self) -> str:
        """Get the VAD server URL."""
        return self.get_server_url(self.vad.server)

    def get_embeddings_url(self) -> str:
        """Get the embeddings server URL."""
        return self.get_server_url(self.embeddings.server)

    def get_transcription_urls(self) -> list[str]:
        """Get transcription server URLs."""
        return [self.get_server_url(name) for name in self.transcription.servers]

    def validate(self) -> None:
        """Validate that all required sections reference existing servers."""
        if not self.servers:
            raise ConfigurationError("No servers configured. See config.example.yaml")
        server_names = {s.name for s in self.servers}
        if not self.vad.server:
            raise ConfigurationError("vad.server not configured. See config.example.yaml")
        if self.vad.server not in server_names:
            raise ConfigurationError(f"VAD server '{self.vad.server}' not found in servers list")
        if not self.embeddings.server:
            raise ConfigurationError("embeddings.server not configured. See config.example.yaml")
        if self.embeddings.server not in server_names:
            raise ConfigurationError(
                f"Embeddings server '{self.embeddings.server}' not found in servers list"
            )
        if not self.transcription.servers:
            raise ConfigurationError(
                "transcription.servers not configured. See config.example.yaml"
            )
        for name in self.transcription.servers:
            if name not in server_names:
                raise ConfigurationError(f"Transcription server '{name}' not found in servers list")


def load_config(config_path: Path) -> AppConfig:
    """Load application configuration from YAML file.

    Applies defaults for optional fields not specified in the file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Parsed AppConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ConfigurationError: If config is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Copy config.example.yaml to config.yaml and configure your settings."
        )

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ConfigurationError(f"Empty config file: {config_path}")
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a mapping, got {type(data).__name__}")

    for section in ("servers", "vad", "embeddings", "transcription", "web"):
        if section in data and not isinstance(data[section], (dict, list)):
            raise ConfigurationError(f"Config section '{section}' must be a mapping")

    servers_raw = data.get("servers", [])
    if not isinstance(servers_raw, list):
        raise ConfigurationError("Config section 'servers' must be a list")
    servers = [ServerInfo(url=s["url"], name=s["name"]) for s in servers_raw]

    vad = VadConfig()
    if "vad" in data:
        d = data["vad"]
        vad = VadConfig(
            server=d.get("server", ""),
            timeout=d.get("timeout", 120.0),
            min_silence_duration_ms=d.get("min_silence_duration_ms", 1200),
            speech_pad_ms=d.get("speech_pad_ms", 30),
            threshold=d.get("threshold", 0.5),
        )

    embeddings = EmbeddingsConfig()
    if "embeddings" in data:
        d = data["embeddings"]
        embeddings = EmbeddingsConfig(
            server=d.get("server", ""),
            model=d.get("model", "Wespeaker/wespeaker-voxceleb-resnet34-LM"),
            timeout=d.get("timeout", 60.0),
            threshold=d.get("threshold", 0.6),
            min_duration_ms=d.get("min_duration_ms", 1500),
            unknown_cluster_threshold=d.get("unknown_cluster_threshold", 0.25),
            confident_gap=d.get("confident_gap", 0.2),
            min_threshold=d.get("min_threshold", 0.45),
            max_workers=d.get("max_workers", 4),
        )

    transcription = TranscriptionConfig()
    if "transcription" in data:
        d = data["transcription"]
        transcription = TranscriptionConfig(
            servers=d.get("servers", []),
            model=d.get("model", "Systran/faster-whisper-medium"),
            language=d.get("language", "ru"),
            timeout=d.get("timeout", 120.0),
            max_gap_ms=d.get("max_gap_ms", 500),
            max_chunk_ms=d.get("max_chunk_ms", 30000),
        )

    web = WebConfig()
    if "web" in data:
        d = data["web"]
        web = WebConfig(
            host=d.get("host", "127.0.0.1"),
            port=d.get("port", 8080),
            session_ttl_days=d.get("session_ttl_days", 7),
            secure_cookies=d.get("secure_cookies", False),
        )

    log_level = str(data.get("log_level", "DEBUG")).upper()

    cfg = AppConfig(
        servers=servers,
        vad=vad,
        embeddings=embeddings,
        transcription=transcription,
        web=web,
        log_level=log_level,
    )
    cfg.validate()

    _logger.info("Config loaded", extra={"servers": len(servers)})
    return cfg


_app_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get cached application config. Loads from CONFIG_FILE on first call."""
    global _app_config
    if _app_config is None:
        _app_config = load_config(CONFIG_FILE)
    return _app_config
