"""Tests for config.py — config loading, validation, defaults matching config.yaml."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from meetscribe.config import (
    AppConfig,
    EmbeddingsConfig,
    ServerInfo,
    TranscriptionConfig,
    VadConfig,
    ValidatedConfig,
    WebConfig,
    load_config,
)
from meetscribe.errors import ConfigurationError


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


MINIMAL_CONFIG = {
    "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
    "vad": {"server": "gpu1"},
    "embeddings": {"server": "gpu1"},
    "transcription": {"servers": ["gpu1"]},
}


class TestLoadConfig:
    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file_raises(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Empty config"):
            load_config(p)

    def test_minimal_config_loads(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "config.yaml", MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.servers[0].name == "gpu1"
        assert cfg.servers[0].url == "http://localhost:8000"
        assert cfg.vad.server == "gpu1"
        assert cfg.embeddings.server == "gpu1"
        assert cfg.transcription.servers == ["gpu1"]

    def test_defaults_applied_when_sections_missing(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        cfg = load_config(p)
        # Defaults from dataclass
        assert cfg.vad.timeout == 120.0
        assert cfg.vad.min_silence_duration_ms == 1200
        assert cfg.vad.speech_pad_ms == 30
        assert cfg.vad.threshold == 0.5
        assert cfg.embeddings.model == "Wespeaker/wespeaker-voxceleb-resnet34-LM"
        assert cfg.embeddings.timeout == 60.0
        assert cfg.embeddings.threshold == 0.6
        assert cfg.transcription.model == "Systran/faster-whisper-medium"
        assert cfg.transcription.language == "ru"
        assert cfg.web.host == "127.0.0.1"
        assert cfg.web.port == 8080


class TestValidate:
    def test_empty_servers_raises(self):
        cfg = AppConfig(servers=[])
        with pytest.raises(ConfigurationError, match="No servers configured"):
            cfg.validate()

    def test_vad_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "nonexistent"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ConfigurationError, match="VAD server 'nonexistent' not found"):
            load_config(p)

    def test_embeddings_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "nonexistent"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ConfigurationError, match="Embeddings server 'nonexistent' not found"):
            load_config(p)

    def test_transcription_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["nonexistent"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ConfigurationError, match="Transcription server 'nonexistent'"):
            load_config(p)


class TestPartialConfig:
    """Test that load_config applies .get() defaults when YAML keys are missing."""

    def test_all_defaults_with_only_servers(self, tmp_path: Path):
        """Minimal config with only server refs → all other fields get defaults."""
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        cfg = load_config(p)

        assert cfg.vad.timeout == 120.0
        assert cfg.vad.min_silence_duration_ms == 1200
        assert cfg.vad.speech_pad_ms == 30
        assert cfg.vad.threshold == 0.5

        assert cfg.embeddings.model == "Wespeaker/wespeaker-voxceleb-resnet34-LM"
        assert cfg.embeddings.timeout == 60.0
        assert cfg.embeddings.threshold == 0.6
        assert cfg.embeddings.min_duration_ms == 1500
        assert cfg.embeddings.unknown_cluster_threshold == 0.25
        assert cfg.embeddings.confident_gap == 0.2
        assert cfg.embeddings.min_threshold == 0.45
        assert cfg.embeddings.max_workers == 4

        assert cfg.transcription.model == "Systran/faster-whisper-medium"
        assert cfg.transcription.language == "ru"
        assert cfg.transcription.timeout == 120.0
        assert cfg.transcription.max_gap_ms == 500
        assert cfg.transcription.max_chunk_ms == 30000

    def test_web_with_partial_keys(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
            "web": {"port": 9090},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        cfg = load_config(p)
        assert cfg.web.port == 9090
        assert cfg.web.host == "127.0.0.1"
        assert cfg.web.session_ttl_days == 7
        assert cfg.web.secure_cookies is False

    def test_no_web_section(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        cfg = load_config(p)
        assert cfg.web.host == "127.0.0.1"
        assert cfg.web.port == 8080

    def test_custom_values_override_defaults(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1", "timeout": 60.0, "threshold": 0.8},
            "embeddings": {"server": "gpu1", "timeout": 30.0, "max_workers": 8},
            "transcription": {"servers": ["gpu1"], "language": "en", "max_gap_ms": 1000},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        cfg = load_config(p)
        assert cfg.vad.timeout == 60.0
        assert cfg.vad.threshold == 0.8
        assert cfg.embeddings.timeout == 30.0
        assert cfg.embeddings.max_workers == 8
        assert cfg.transcription.language == "en"
        assert cfg.transcription.max_gap_ms == 1000


class TestAppConfigMethods:
    def _make_config(self):
        return AppConfig(
            servers=[
                ServerInfo(url="http://a:8000", name="gpu1"),
                ServerInfo(url="http://b:8000", name="gpu2"),
            ],
            vad=VadConfig(server="gpu1"),
            embeddings=EmbeddingsConfig(server="gpu2"),
            transcription=TranscriptionConfig(servers=["gpu1", "gpu2"]),
        )

    def test_get_server_url_found(self):
        cfg = self._make_config()
        assert cfg.get_server_url("gpu1") == "http://a:8000"
        assert cfg.get_server_url("gpu2") == "http://b:8000"

    def test_get_server_url_not_found(self):
        cfg = self._make_config()
        with pytest.raises(ConfigurationError, match="not found"):
            cfg.get_server_url("nonexistent")

    def test_get_vad_url(self):
        cfg = self._make_config()
        assert cfg.get_vad_url() == "http://a:8000"

    def test_get_embeddings_url(self):
        cfg = self._make_config()
        assert cfg.get_embeddings_url() == "http://b:8000"

    def test_get_transcription_urls(self):
        cfg = self._make_config()
        urls = cfg.get_transcription_urls()
        assert urls == ["http://a:8000", "http://b:8000"]

    def test_get_transcription_urls_empty(self):
        cfg = AppConfig(
            servers=[ServerInfo(url="http://a:8000", name="gpu1")],
            transcription=TranscriptionConfig(servers=[]),
        )
        assert cfg.get_transcription_urls() == []

    def test_validate_empty_vad_server_raises(self):
        """Empty vad.server should fail validation."""
        cfg = AppConfig(
            servers=[ServerInfo(url="http://a:8000", name="gpu1")],
            vad=VadConfig(server=""),
            embeddings=EmbeddingsConfig(server="gpu1"),
            transcription=TranscriptionConfig(servers=["gpu1"]),
        )
        with pytest.raises(ConfigurationError, match="vad.server not configured"):
            cfg.validate()

    def test_validate_empty_embeddings_server_raises(self):
        """Empty embeddings.server should fail validation."""
        cfg = AppConfig(
            servers=[ServerInfo(url="http://a:8000", name="gpu1")],
            vad=VadConfig(server="gpu1"),
            embeddings=EmbeddingsConfig(server=""),
            transcription=TranscriptionConfig(servers=["gpu1"]),
        )
        with pytest.raises(ConfigurationError, match="embeddings.server not configured"):
            cfg.validate()

    def test_validate_empty_transcription_servers_raises(self):
        """Empty transcription.servers should fail validation."""
        cfg = AppConfig(
            servers=[ServerInfo(url="http://a:8000", name="gpu1")],
            vad=VadConfig(server="gpu1"),
            embeddings=EmbeddingsConfig(server="gpu1"),
            transcription=TranscriptionConfig(servers=[]),
        )
        with pytest.raises(ConfigurationError, match="transcription.servers not configured"):
            cfg.validate()


@dataclass
class _StubConfig(ValidatedConfig):
    """Test-only config with one field per supported type."""

    name: str = ""
    timeout: float = 1.0
    count: int = 0
    enabled: bool = False
    items: list[str] = field(default_factory=list)


class TestValidatedConfig:
    """Test ValidatedConfig type validation on a test-only dataclass."""

    # -- str --
    def test_str_accepts_str(self):
        assert _StubConfig(name="ok").name == "ok"

    def test_str_rejects_int(self):
        with pytest.raises(ConfigurationError, match="expected string"):
            _StubConfig(name=123)

    # -- float (accepts int too, YAML `120` → int) --
    def test_float_accepts_float(self):
        assert _StubConfig(timeout=1.5).timeout == 1.5

    def test_float_accepts_int(self):
        assert _StubConfig(timeout=120).timeout == 120

    def test_float_rejects_str(self):
        with pytest.raises(ConfigurationError, match="expected number"):
            _StubConfig(timeout="slow")

    def test_float_rejects_bool(self):
        with pytest.raises(ConfigurationError, match="expected number"):
            _StubConfig(timeout=True)

    # -- int --
    def test_int_accepts_int(self):
        assert _StubConfig(count=5).count == 5

    def test_int_rejects_float(self):
        with pytest.raises(ConfigurationError, match="expected integer"):
            _StubConfig(count=1.5)

    def test_int_rejects_bool(self):
        with pytest.raises(ConfigurationError, match="expected integer"):
            _StubConfig(count=True)

    # -- bool --
    def test_bool_accepts_bool(self):
        assert _StubConfig(enabled=True).enabled is True

    def test_bool_rejects_int(self):
        with pytest.raises(ConfigurationError, match="expected boolean"):
            _StubConfig(enabled=1)

    # -- list --
    def test_list_accepts_list(self):
        assert _StubConfig(items=["a"]).items == ["a"]

    def test_list_rejects_str(self):
        with pytest.raises(ConfigurationError, match="expected list"):
            _StubConfig(items="a,b")

    # -- error message format --
    def test_error_includes_class_and_field(self):
        with pytest.raises(ConfigurationError, match=r"_StubConfig\.timeout"):
            _StubConfig(timeout="bad")

    # -- defaults pass validation --
    def test_all_defaults_pass(self):
        cfg = _StubConfig()
        assert cfg.name == ""
        assert cfg.timeout == 1.0
        assert cfg.count == 0
        assert cfg.enabled is False
        assert cfg.items == []

    # -- integration: wrong type in YAML bubbles up through load_config --
    def test_yaml_with_wrong_type_raises(self, tmp_path: Path):
        data = {**MINIMAL_CONFIG, "vad": {"server": "gpu1", "timeout": "slow"}}
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ConfigurationError, match="expected number"):
            load_config(p)

    def test_section_as_list_raises(self, tmp_path: Path):
        data = {**MINIMAL_CONFIG, "vad": ["server", "gpu1"]}
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ConfigurationError, match="must be a mapping"):
            load_config(p)


class TestDefaultsMatchConfigYaml:
    """Verify dataclass defaults match the values in data/config.yaml.

    This test prevents the recurring bug where code defaults diverge from
    the example config (git history: 3 bugs from this pattern).
    """

    @pytest.fixture
    def config_yaml(self) -> dict:
        config_path = Path(__file__).parent.parent.parent / "config.example.yaml"
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_vad_defaults(self, config_yaml: dict):
        d = config_yaml["vad"]
        defaults = VadConfig()
        assert defaults.timeout == d["timeout"]
        assert defaults.min_silence_duration_ms == d["min_silence_duration_ms"]
        assert defaults.speech_pad_ms == d["speech_pad_ms"]
        assert defaults.threshold == d["threshold"]

    def test_embeddings_defaults(self, config_yaml: dict):
        d = config_yaml["embeddings"]
        defaults = EmbeddingsConfig()
        assert defaults.model == d["model"]
        assert defaults.timeout == d["timeout"]
        assert defaults.threshold == d["threshold"]
        assert defaults.min_duration_ms == d["min_duration_ms"]
        assert defaults.unknown_cluster_threshold == d["unknown_cluster_threshold"]
        assert defaults.confident_gap == d["confident_gap"]
        assert defaults.min_threshold == d["min_threshold"]

    def test_transcription_defaults(self, config_yaml: dict):
        d = config_yaml["transcription"]
        defaults = TranscriptionConfig()
        assert defaults.model == d["model"]
        assert defaults.language == d["language"]
        assert defaults.timeout == d["timeout"]
        assert defaults.max_gap_ms == d["max_gap_ms"]
        assert defaults.max_chunk_ms == d["max_chunk_ms"]

    def test_web_defaults(self, config_yaml: dict):
        d = config_yaml.get("web", {})
        defaults = WebConfig()
        assert defaults.host == d.get("host", "127.0.0.1")
        assert defaults.port == d.get("port", 8080)
        assert defaults.session_ttl_days == d.get("session_ttl_days", 7)
