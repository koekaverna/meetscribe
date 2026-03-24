"""Tests for web/services/session.py — session lifecycle, tracks, samples."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from meetscribe.database import get_db
from meetscribe.web.models import SessionStatus
from meetscribe.web.services.session import SessionService

from .conftest import make_wav_file


@pytest.fixture
def session_env(tmp_path: Path):
    """Isolated DB + dirs for SessionService."""
    db_path = tmp_path / "test.db"
    sessions_dir = tmp_path / "sessions"
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        "servers:\n"
        "  - url: http://localhost:8000\n"
        "    name: gpu1\n"
        "vad:\n"
        "  server: gpu1\n"
        "embeddings:\n"
        "  server: gpu1\n"
        "transcription:\n"
        "  servers: [gpu1]\n",
        encoding="utf-8",
    )

    conn = get_db(db_path)

    with patch("meetscribe.web.services.session.config.CONFIG_FILE", config_file):
        svc = SessionService(conn, sessions_dir=sessions_dir)
        yield svc, tmp_path
        conn.close()


class TestLifecycle:
    def test_create_get_delete(self, session_env):
        svc, _ = session_env
        state = svc.create("default")
        assert state.id
        assert state.team_name == "default"

        fetched = svc.get(state.id)
        assert fetched is not None
        assert fetched.status == SessionStatus.CREATED

        assert svc._tracks_dir(state.id).exists()
        assert svc._samples_dir(state.id).exists()

        assert svc.delete(state.id) is True
        assert svc.get(state.id) is None
        assert not svc._session_dir(state.id).exists()

    def test_delete_nonexistent_returns_false(self, session_env):
        svc, _ = session_env
        assert svc.delete("no-such-id") is False


class TestTracks:
    def test_add_track_moves_file(self, session_env):
        svc, tmp_path = session_env
        state = svc.create("default")
        source = make_wav_file(tmp_path / "upload.wav")

        track = svc.add_track(state.id, "meeting.wav", source)

        assert track.track_num == 1
        assert track.filename == "meeting.wav"
        assert not source.exists()
        assert svc.get_track_path(state.id, 1) is not None

        fetched = svc.get(state.id)
        assert fetched.status == SessionStatus.UPLOADED

    def test_renumber_on_middle_delete(self, session_env):
        svc, tmp_path = session_env
        state = svc.create("default")

        for i in range(3):
            src = make_wav_file(tmp_path / f"t{i}.wav")
            svc.add_track(state.id, f"track{i}.wav", src)

        assert svc.remove_track(state.id, 2) is True

        fetched = svc.get(state.id)
        nums = [t.track_num for t in fetched.tracks]
        assert nums == [1, 2]

        assert svc.get_track_path(state.id, 2) is not None
        assert svc.get_track_path(state.id, 3) is None

    def test_configure_track(self, session_env):
        svc, tmp_path = session_env
        state = svc.create("default")
        src = make_wav_file(tmp_path / "t.wav")
        svc.add_track(state.id, "t.wav", src)

        assert svc.update_track_config(state.id, 1, speaker_name="Alice", diarize=False) is True

        fetched = svc.get(state.id)
        assert fetched.tracks[0].speaker_name == "Alice"
        assert fetched.tracks[0].diarize is False
        assert fetched.status == SessionStatus.CONFIGURED

    def test_remove_all_tracks_resets_status(self, session_env):
        svc, tmp_path = session_env
        state = svc.create("default")
        src = make_wav_file(tmp_path / "t.wav")
        svc.add_track(state.id, "t.wav", src)
        svc.remove_track(state.id, 1)

        fetched = svc.get(state.id)
        assert fetched.status == SessionStatus.CREATED

    def test_rollback_on_fk_error_preserves_source(self, session_env):
        svc, tmp_path = session_env
        source = make_wav_file(tmp_path / "upload.wav")

        # FK violation: session_id doesn't exist in sessions table
        with pytest.raises(sqlite3.IntegrityError):
            svc.add_track("nonexistent-session", "t.wav", source)

        assert source.exists()


class TestSamples:
    def test_add_and_get_path(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        sample = svc.add_sample(
            session_id=state.id,
            track_num=1,
            cluster_id=0,
            filename="sample.wav",
            duration_ms=1500,
            content=b"\x00" * 100,
        )

        assert sample.id
        assert sample.duration_ms == 1500
        assert sample.filename == "sample.wav"

        path = svc.get_sample_path(state.id, sample.id)
        assert path is not None
        assert path.read_bytes() == b"\x00" * 100

    def test_move_sample_between_speakers(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        sp1 = svc.add_speaker(state.id, "Alice")
        sp2 = svc.add_speaker(state.id, "Bob")

        sample = svc.add_sample(
            session_id=state.id,
            track_num=1,
            cluster_id=0,
            filename="s.wav",
            duration_ms=1000,
            content=b"\x00" * 50,
        )

        assert svc.move_sample(state.id, sample.id, sp1.id) is True
        fetched = svc.get(state.id)
        assert fetched.samples[0].speaker_id == sp1.id

        assert svc.move_sample(state.id, sample.id, sp2.id) is True
        fetched = svc.get(state.id)
        assert fetched.samples[0].speaker_id == sp2.id

    def test_delete_sample_cleans_file(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        sample = svc.add_sample(
            session_id=state.id,
            track_num=1,
            cluster_id=0,
            filename="s.wav",
            duration_ms=1000,
            content=b"\x00" * 50,
        )
        sample_path = svc.get_sample_path(state.id, sample.id)
        assert sample_path.exists()

        assert svc.delete_sample(state.id, sample.id) is True
        assert not sample_path.exists()

        fetched = svc.get(state.id)
        assert len(fetched.samples) == 0


class TestSpeakers:
    def test_add_rename_delete(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        sp = svc.add_speaker(state.id, "Alice")
        assert sp.name == "Alice"

        assert svc.rename_speaker(state.id, sp.id, "Bob") is True
        fetched = svc.get(state.id)
        assert fetched.speakers[0].name == "Bob"

        assert svc.delete_speaker(state.id, sp.id) is True
        fetched = svc.get(state.id)
        assert len(fetched.speakers) == 0

    def test_delete_speaker_unassigns_samples(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        sp = svc.add_speaker(state.id, "Alice")
        sample = svc.add_sample(
            session_id=state.id,
            track_num=1,
            cluster_id=0,
            filename="s.wav",
            duration_ms=1000,
            content=b"\x00" * 50,
        )
        svc.move_sample(state.id, sample.id, sp.id)

        svc.delete_speaker(state.id, sp.id)
        fetched = svc.get(state.id)
        assert fetched.samples[0].speaker_id is None


class TestTranscript:
    def test_set_transcript(self, session_env):
        svc, _ = session_env
        state = svc.create("default")

        svc.set_transcript(state.id, "Hello, world!")
        fetched = svc.get(state.id)
        assert fetched.transcript == "Hello, world!"
        assert fetched.status == SessionStatus.TRANSCRIBED

    def test_set_transcript_nonexistent_raises(self, session_env):
        svc, _ = session_env
        with pytest.raises(ValueError, match="Session not found"):
            svc.set_transcript("no-such-id", "text")
