"""Tests for servers.py — config loading, validation, defaults matching config.yaml."""

from pathlib import Path

import pytest
import yaml

from meetscribe.servers import (
    AppConfig,
    EmbeddingsConfig,
    TranscriptionConfig,
    VadConfig,
    WebConfig,
    load_config,
)


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
        with pytest.raises(ValueError, match="Empty config"):
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
        with pytest.raises(ValueError, match="No servers configured"):
            cfg.validate()

    def test_vad_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "nonexistent"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ValueError, match="VAD server 'nonexistent' not found"):
            load_config(p)

    def test_embeddings_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "nonexistent"},
            "transcription": {"servers": ["gpu1"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ValueError, match="Embeddings server 'nonexistent' not found"):
            load_config(p)

    def test_transcription_server_not_in_list_raises(self, tmp_path: Path):
        data = {
            "servers": [{"url": "http://localhost:8000", "name": "gpu1"}],
            "vad": {"server": "gpu1"},
            "embeddings": {"server": "gpu1"},
            "transcription": {"servers": ["nonexistent"]},
        }
        p = _write_yaml(tmp_path / "config.yaml", data)
        with pytest.raises(ValueError, match="Transcription server 'nonexistent' not found"):
            load_config(p)


class TestDefaultsMatchConfigYaml:
    """Verify dataclass defaults match the values in data/config.yaml.

    This test prevents the recurring bug where code defaults diverge from
    the example config (git history: 3 bugs from this pattern).
    """

    @pytest.fixture
    def config_yaml(self) -> dict:
        config_path = Path(__file__).parent.parent / "data" / "config.yaml"
        if not config_path.exists():
            pytest.skip("data/config.yaml not found")
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
