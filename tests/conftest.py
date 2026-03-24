"""Shared test fixtures."""

import io
import sqlite3
import wave
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meetscribe.database import get_db


@pytest.fixture
def db(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """In-memory-like SQLite with all migrations applied."""
    conn = get_db(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def team_id(db: sqlite3.Connection) -> int:
    """ID of the auto-created 'default' team."""
    row = db.execute("SELECT id FROM teams WHERE name = 'default'").fetchone()
    assert row is not None, "default team not found — migrations may be missing"
    return row["id"]


@pytest.fixture
def wav_bytes() -> bytes:
    """1-second 16kHz mono WAV (silence)."""
    sample_rate = 16000
    n_frames = sample_rate  # 1 second
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


@pytest.fixture
def wav_file(tmp_path: Path, wav_bytes: bytes) -> Path:
    """1-second WAV file on disk."""
    p = tmp_path / "test.wav"
    p.write_bytes(wav_bytes)
    return p


@pytest.fixture
def sample_embedding() -> list[float]:
    """Normalized 256-dim embedding vector."""
    import math

    raw = [float(i) for i in range(256)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def mock_extractor(sample_embedding: list[float]) -> MagicMock:
    """EmbeddingExtractor mock that returns sample_embedding for any input."""
    ext = MagicMock()
    ext.extract.return_value = sample_embedding
    ext.extract_from_file.return_value = sample_embedding
    return ext


def make_wav_file(path: Path, duration_s: float = 1.0) -> Path:
    """Create a WAV file with given duration."""
    sample_rate = 16000
    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    path.write_bytes(buf.getvalue())
    return path
