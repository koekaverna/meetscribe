"""Application configuration loaded from config.yaml."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """A remote processing server."""

    url: str
    name: str


@dataclass
class VadConfig:
    """VAD endpoint configuration."""

    server: str
    timeout: float


@dataclass
class EmbeddingsConfig:
    """Speaker embeddings configuration."""

    server: str
    model: str
    timeout: float
    threshold: float
    min_duration_ms: int
    unknown_cluster_threshold: float
    confident_gap: float
    min_threshold: float
    max_workers: int


@dataclass
class TranscriptionConfig:
    """Transcription configuration."""

    servers: list[str]
    model: str
    language: str
    timeout: float
    max_gap_ms: int
    max_chunk_ms: int


@dataclass
class WebConfig:
    """Web UI configuration."""

    host: str
    port: int
    session_ttl_days: int
    secure_cookies: bool


@dataclass
class AppConfig:
    """Full application configuration loaded from YAML."""

    servers: list[ServerInfo] = field(default_factory=list)
    vad: VadConfig | None = None
    embeddings: EmbeddingsConfig | None = None
    transcription: TranscriptionConfig | None = None
    web: WebConfig | None = None

    def get_server_url(self, name: str) -> str:
        """Get server URL by name."""
        for s in self.servers:
            if s.name == name:
                return s.url
        raise ValueError(f"Server '{name}' not found in configuration")

    def get_vad_url(self) -> str:
        """Get the VAD server URL."""
        if not self.vad:
            raise ValueError("VAD server not configured")
        return self.get_server_url(self.vad.server)

    def get_embeddings_url(self) -> str:
        """Get the embeddings server URL."""
        if not self.embeddings:
            raise ValueError("Embeddings server not configured")
        return self.get_server_url(self.embeddings.server)

    def get_transcription_urls(self) -> list[str]:
        """Get transcription server URLs."""
        if not self.transcription or not self.transcription.servers:
            raise ValueError("Transcription servers not configured")
        return [self.get_server_url(name) for name in self.transcription.servers]

    def validate(self) -> None:
        """Validate that configured servers reference existing server names."""
        if not self.servers:
            raise ValueError("No servers configured. See config.example.yaml")
        server_names = {s.name for s in self.servers}
        if self.vad and self.vad.server not in server_names:
            raise ValueError(f"VAD server '{self.vad.server}' not found in servers list")
        if self.embeddings and self.embeddings.server not in server_names:
            raise ValueError(
                f"Embeddings server '{self.embeddings.server}' not found in servers list"
            )
        if self.transcription:
            for name in self.transcription.servers:
                if name not in server_names:
                    raise ValueError(
                        f"Transcription server '{name}' not found in servers list"
                    )


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
        raise ValueError(f"Empty config file: {config_path}")

    servers = [ServerInfo(url=s["url"], name=s["name"]) for s in data.get("servers", [])]

    vad = None
    if "vad" in data:
        d = data["vad"]
        vad = VadConfig(
            server=d["server"],
            timeout=d.get("timeout", 120.0),
        )

    embeddings = None
    if "embeddings" in data:
        d = data["embeddings"]
        embeddings = EmbeddingsConfig(
            server=d["server"],
            model=d.get("model", "pyannote/wespeaker-voxceleb-resnet34-LM"),
            timeout=d.get("timeout", 60.0),
            threshold=d.get("threshold", 0.6),
            min_duration_ms=d.get("min_duration_ms", 1500),
            unknown_cluster_threshold=d.get("unknown_cluster_threshold", 0.7),
            confident_gap=d.get("confident_gap", 0.2),
            min_threshold=d.get("min_threshold", 0.45),
            max_workers=d.get("max_workers", 4),
        )

    transcription = None
    if "transcription" in data:
        d = data["transcription"]
        transcription = TranscriptionConfig(
            servers=d["servers"],
            model=d.get("model", "Systran/faster-whisper-medium"),
            language=d.get("language", "ru"),
            timeout=d.get("timeout", 120.0),
            max_gap_ms=d.get("max_gap_ms", 500),
            max_chunk_ms=d.get("max_chunk_ms", 30000),
        )

    web = None
    if "web" in data:
        d = data["web"]
        web = WebConfig(
            host=d.get("host", "127.0.0.1"),
            port=d.get("port", 8080),
            session_ttl_days=d.get("session_ttl_days", 7),
            secure_cookies=d.get("secure_cookies", False),
        )
    else:
        web = WebConfig(host="127.0.0.1", port=8080, session_ttl_days=7, secure_cookies=False)

    cfg = AppConfig(
        servers=servers,
        vad=vad,
        embeddings=embeddings,
        transcription=transcription,
        web=web,
    )
    cfg.validate()

    logger.info("Loaded config: %d servers", len(servers))
    return cfg
