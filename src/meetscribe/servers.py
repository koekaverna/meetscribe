"""Application configuration loaded from config.yaml."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigurationError

logger = logging.getLogger(__name__)


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
        if not self.transcription.servers:
            raise ConfigurationError("Transcription servers not configured")
        return [self.get_server_url(name) for name in self.transcription.servers]

    def validate(self) -> None:
        """Validate that all required sections reference existing servers."""
        if not self.servers:
            raise ConfigurationError("No servers configured. See config.example.yaml")
        server_names = {s.name for s in self.servers}
        if self.vad.server and self.vad.server not in server_names:
            raise ConfigurationError(f"VAD server '{self.vad.server}' not found in servers list")
        if self.embeddings.server and self.embeddings.server not in server_names:
            raise ConfigurationError(
                f"Embeddings server '{self.embeddings.server}' not found in servers list"
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
        ValueError: If config is invalid.
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

    servers = [ServerInfo(url=s["url"], name=s["name"]) for s in data.get("servers", [])]

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

    cfg = AppConfig(
        servers=servers,
        vad=vad,
        embeddings=embeddings,
        transcription=transcription,
        web=web,
    )
    cfg.validate()

    logger.info("Config loaded", extra={"servers": len(servers)})
    return cfg
