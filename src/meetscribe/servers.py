"""Server configuration for remote processing."""

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

    server: str  # Server name reference


@dataclass
class EmbeddingsConfig:
    """Speaker embeddings endpoint configuration."""

    server: str  # Server name reference


@dataclass
class TranscribeConfig:
    """Transcription configuration."""

    servers: list[str]  # Server name references


@dataclass
class ServersConfig:
    """Full server configuration loaded from YAML."""

    servers: list[ServerInfo] = field(default_factory=list)
    vad: VadConfig | None = None
    embeddings: EmbeddingsConfig | None = None
    transcribe: TranscribeConfig | None = None

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
        if not self.transcribe or not self.transcribe.servers:
            raise ValueError("Transcription servers not configured")
        return [self.get_server_url(name) for name in self.transcribe.servers]

    def validate(self) -> None:
        """Validate that configured servers reference existing server names."""
        if not self.servers:
            raise ValueError("No servers configured. See servers.example.yaml")
        server_names = {s.name for s in self.servers}
        if self.vad and self.vad.server not in server_names:
            raise ValueError(f"VAD server '{self.vad.server}' not found in servers list")
        if self.embeddings and self.embeddings.server not in server_names:
            raise ValueError(
                f"Embeddings server '{self.embeddings.server}' not found in servers list"
            )
        if self.transcribe:
            for name in self.transcribe.servers:
                if name not in server_names:
                    raise ValueError(f"Transcription server '{name}' not found in servers list")


def load_config(config_path: Path) -> ServersConfig:
    """Load server configuration from YAML file.

    Args:
        config_path: Path to servers.yaml file.

    Returns:
        Parsed ServersConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Server config not found: {config_path}\n"
            "Copy servers.example.yaml to servers.yaml and configure your servers."
        )

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty config file: {config_path}")

    servers = [ServerInfo(url=s["url"], name=s["name"]) for s in data.get("servers", [])]

    vad = None
    if "vad" in data:
        vad = VadConfig(server=data["vad"]["server"])

    embeddings = None
    if "embeddings" in data:
        embeddings = EmbeddingsConfig(server=data["embeddings"]["server"])

    transcribe = None
    if "transcribe" in data:
        transcribe = TranscribeConfig(servers=data["transcribe"]["servers"])

    cfg = ServersConfig(servers=servers, vad=vad, embeddings=embeddings, transcribe=transcribe)
    cfg.validate()

    logger.info("Loaded config: %d servers", len(servers))
    return cfg
