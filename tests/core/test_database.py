"""Tests for database.py — SQLite CRUD operations."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from meetscribe.database import (
    _get_schema_version,
    _run_migrations,
    _validate_team_name,
    close_db,
    count_voiceprints,
    create_auth_session,
    create_team,
    create_user,
    delete_auth_session,
    delete_expired_sessions,
    delete_team,
    delete_voiceprint,
    ensure_default_team,
    get_auth_session,
    get_db,
    get_schema_version_expected,
    get_team,
    init_db,
    load_voiceprints,
    save_voiceprint,
)


class TestMigrations:
    def test_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_db()
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "teams" in tables
        assert "voiceprints" in tables
        assert "session_segments" in tables
        close_db()
        # Re-init should be idempotent
        init_db(db_path)
        conn2 = get_db()
        tables2 = {
            r[0]
            for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert tables == tables2
        close_db()

    def test_schema_version_set(self, tmp_path: Path):
        init_db(tmp_path / "test.db")
        assert _get_schema_version(get_db()) == get_schema_version_expected()
        close_db()

    def test_schema_version_table_exists(self, tmp_path: Path):
        init_db(tmp_path / "test.db")
        conn = get_db()
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "schema_version" in tables
        close_db()

    def test_no_rerun_on_current_version(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        version_before = _get_schema_version(get_db())
        close_db()
        # Re-open — migrations should not re-run
        init_db(db_path)
        assert _get_schema_version(get_db()) == version_before
        close_db()

    def test_pre_versioning_db_upgraded(self, tmp_path: Path):
        """A database created before versioning gets upgraded correctly."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Create tables as old code would (no schema_version)
        conn.executescript("""
            CREATE TABLE teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                description TEXT
            );
            INSERT INTO teams (name, description) VALUES ('default', 'Default team');
        """)
        conn.commit()
        assert _get_schema_version(conn) == 0
        conn.close()
        # Now open with versioned init_db
        init_db(db_path)
        conn2 = get_db()
        assert _get_schema_version(conn2) == get_schema_version_expected()
        # Old data preserved
        team = conn2.execute("SELECT * FROM teams WHERE name = 'default'").fetchone()
        assert team is not None
        close_db()

    def test_failed_migration_does_not_bump_version(self, tmp_path: Path):
        """If a migration SQL fails, schema_version must not be updated."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Copy the real 001_initial.sql
        real_001 = Path(__file__).parent.parent.parent / "src/meetscribe/migrations/001_initial.sql"
        (migrations_dir / "001_initial.sql").write_text(real_001.read_text())

        # Create a broken second migration
        (migrations_dir / "002_broken.sql").write_text(
            "CREATE TABLE migration_test_ok (id INTEGER);\nINVALID SQL THAT WILL FAIL;\n"
        )

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        with patch("meetscribe.database._MIGRATIONS_DIR", migrations_dir):
            with pytest.raises(Exception):
                _run_migrations(conn)

        # Version should be 1 (only first migration applied), not 2
        assert _get_schema_version(conn) == 1
        # The partial table from broken migration should NOT exist
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "migration_test_ok" not in tables
        conn.close()


class TestValidateTeamName:
    @pytest.mark.parametrize("name", ["default", "my-team", "Team_1", "a" * 64])
    def test_valid_names(self, name: str):
        _validate_team_name(name)  # should not raise

    @pytest.mark.parametrize("name", ["", "a" * 65, "has space", "special!", "слово"])
    def test_invalid_names(self, name: str):
        with pytest.raises(ValueError, match="Invalid team name"):
            _validate_team_name(name)


class TestTeamCrud:
    def test_create_and_get(self, db: sqlite3.Connection):
        tid = create_team(db, "test-team", "A test team")
        assert tid is not None
        team = get_team(db, "test-team")
        assert team["name"] == "test-team"
        assert team["description"] == "A test team"

    def test_delete_team(self, db: sqlite3.Connection):
        create_team(db, "doomed")
        assert delete_team(db, "doomed") is True
        assert get_team(db, "doomed") is None

    def test_delete_default_raises(self, db: sqlite3.Connection):
        with pytest.raises(ValueError, match="Cannot delete"):
            delete_team(db, "default")

    def test_delete_nonexistent_returns_false(self, db: sqlite3.Connection):
        assert delete_team(db, "nope") is False

    def test_cascade_deletes_voiceprints(self, db: sqlite3.Connection):
        tid = create_team(db, "temp-team")
        save_voiceprint(db, tid, "speaker1", [0.1, 0.2], "model-v1")
        assert count_voiceprints(db, tid) == 1
        delete_team(db, "temp-team")
        assert count_voiceprints(db, tid) == 0


class TestVoiceprintCrud:
    def test_save_and_load(self, db: sqlite3.Connection, team_id: int):
        embedding = [0.1, 0.2, 0.3]
        save_voiceprint(db, team_id, "Alice", embedding, "wespeaker-v1")
        vps = load_voiceprints(db, team_id)
        assert "Alice" in vps
        assert vps["Alice"] == pytest.approx(embedding)

    def test_upsert_updates_embedding(self, db: sqlite3.Connection, team_id: int):
        save_voiceprint(db, team_id, "Alice", [0.1, 0.2], "model-v1")
        save_voiceprint(db, team_id, "Alice", [0.9, 0.8], "model-v2")
        vps = load_voiceprints(db, team_id)
        assert vps["Alice"] == pytest.approx([0.9, 0.8])

    def test_upsert_updates_model(self, db: sqlite3.Connection, team_id: int):
        save_voiceprint(db, team_id, "Alice", [0.1], "old-model")
        save_voiceprint(db, team_id, "Alice", [0.1], "new-model")
        row = db.execute(
            "SELECT model FROM voiceprints WHERE team_id = ? AND name = ?",
            (team_id, "Alice"),
        ).fetchone()
        assert row["model"] == "new-model"

    def test_delete(self, db: sqlite3.Connection, team_id: int):
        save_voiceprint(db, team_id, "Alice", [0.1], "model")
        assert delete_voiceprint(db, team_id, "Alice") is True
        assert load_voiceprints(db, team_id) == {}

    def test_delete_nonexistent_returns_false(self, db: sqlite3.Connection, team_id: int):
        assert delete_voiceprint(db, team_id, "nobody") is False

    def test_scoped_by_team(self, db: sqlite3.Connection):
        t1 = create_team(db, "team1")
        t2 = create_team(db, "team2")
        save_voiceprint(db, t1, "Alice", [1.0], "model")
        save_voiceprint(db, t2, "Bob", [2.0], "model")
        assert list(load_voiceprints(db, t1).keys()) == ["Alice"]
        assert list(load_voiceprints(db, t2).keys()) == ["Bob"]


class TestEnsureDefaultTeam:
    def test_idempotent(self, db: sqlite3.Connection):
        ensure_default_team(db)
        ensure_default_team(db)
        teams = db.execute("SELECT COUNT(*) as cnt FROM teams WHERE name = 'default'").fetchone()
        assert teams["cnt"] == 1


class TestAuthSessions:
    def _make_user(self, db: sqlite3.Connection, team_id: int) -> int:
        uid = create_user(db, "testuser", "hash123", team_id)
        db.commit()
        return uid

    def test_create_and_get(self, db: sqlite3.Connection, team_id: int):
        uid = self._make_user(db, team_id)
        expires = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        create_auth_session(db, uid, "tok123", expires)
        db.commit()
        session = get_auth_session(db, "tok123")
        assert session is not None
        assert session["username"] == "testuser"
        assert session["team_name"] == "default"

    def test_expired_session_returns_none(self, db: sqlite3.Connection, team_id: int):
        uid = self._make_user(db, team_id)
        expires = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        create_auth_session(db, uid, "expired_tok", expires)
        db.commit()
        assert get_auth_session(db, "expired_tok") is None

    def test_delete_session(self, db: sqlite3.Connection, team_id: int):
        uid = self._make_user(db, team_id)
        expires = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        create_auth_session(db, uid, "tok_del", expires)
        db.commit()
        assert delete_auth_session(db, "tok_del") is True
        db.commit()
        assert get_auth_session(db, "tok_del") is None

    def test_delete_expired(self, db: sqlite3.Connection, team_id: int):
        uid = self._make_user(db, team_id)
        past = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        future = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        create_auth_session(db, uid, "old", past)
        create_auth_session(db, uid, "new", future)
        db.commit()
        deleted = delete_expired_sessions(db)
        db.commit()
        assert deleted == 1
        assert get_auth_session(db, "new") is not None
