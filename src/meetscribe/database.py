"""SQLite database for team and voiceprint management."""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

TEAM_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_team_name(name: str) -> None:
    """Validate team name: alphanumeric, hyphens, underscores, 1-64 chars."""
    if not TEAM_NAME_RE.match(name):
        raise ValueError(
            f"Invalid team name '{name}'. "
            "Use only letters, digits, hyphens, underscores (1-64 chars)."
        )


def get_db(db_path: Path) -> sqlite3.Connection:
    """Open (and create if needed) the database, run migrations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _run_migrations(conn)
    ensure_default_team(conn)
    return conn


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version. Returns 0 if schema_version table doesn't exist."""
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row["version"] if row else 0
    except sqlite3.OperationalError:
        return 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set schema version (upsert single row)."""
    conn.execute(
        "INSERT INTO schema_version (id, version) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
        (version,),
    )


# Each migration brings the schema from version N-1 to version N.
# Migrations are append-only — never modify existing entries.
_MIGRATIONS: list[str] = [
    # Migration 1: initial schema + schema_version table
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        id      INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS teams (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        description TEXT
    );

    CREATE TABLE IF NOT EXISTS voiceprints (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id     INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        embedding   TEXT NOT NULL,
        model       TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(team_id, name)
    );

    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        team_id       INTEGER NOT NULL REFERENCES teams(id),
        is_admin      INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
        token      TEXT PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        expires_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        team_id    INTEGER NOT NULL REFERENCES teams(id),
        status     TEXT NOT NULL DEFAULT 'created',
        language   TEXT NOT NULL DEFAULT 'ru',
        transcript TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS session_tracks (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        track_num    INTEGER NOT NULL,
        filename     TEXT NOT NULL,
        speaker_name TEXT,
        diarize      INTEGER NOT NULL DEFAULT 1,
        UNIQUE(session_id, track_num)
    );

    CREATE TABLE IF NOT EXISTS session_speakers (
        id         TEXT NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        name       TEXT NOT NULL,
        PRIMARY KEY (session_id, id)
    );

    CREATE TABLE IF NOT EXISTS session_samples (
        id                 TEXT NOT NULL,
        session_id         TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        track_num          INTEGER NOT NULL,
        cluster_id         INTEGER NOT NULL,
        filename           TEXT NOT NULL,
        duration_ms        INTEGER NOT NULL,
        speaker_id         TEXT,
        is_known           INTEGER NOT NULL DEFAULT 0,
        known_speaker_name TEXT,
        PRIMARY KEY (session_id, id)
    );
    """,
]

SCHEMA_VERSION = len(_MIGRATIONS)


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations to bring the database up to date."""
    current = _get_schema_version(conn)

    if current >= SCHEMA_VERSION:
        return

    for version in range(current + 1, SCHEMA_VERSION + 1):
        logger.info("Applying migration %d/%d", version, SCHEMA_VERSION)
        conn.executescript(_MIGRATIONS[version - 1])
        _set_schema_version(conn, version)
        conn.commit()

    logger.info("Database schema is at version %d", SCHEMA_VERSION)


def ensure_default_team(conn: sqlite3.Connection) -> None:
    """Create the 'default' team if it doesn't exist."""
    conn.execute(
        "INSERT OR IGNORE INTO teams (name, description) VALUES (?, ?)",
        ("default", "Default team"),
    )
    conn.commit()


def create_team(conn: sqlite3.Connection, name: str, description: str | None = None) -> int | None:
    """Create a team. Returns its id."""
    _validate_team_name(name)
    cursor = conn.execute(
        "INSERT INTO teams (name, description) VALUES (?, ?)",
        (name, description),
    )
    conn.commit()
    return cursor.lastrowid


def get_team(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    """Fetch a team by name."""
    row = conn.execute("SELECT * FROM teams WHERE name = ?", (name,)).fetchone()
    return cast("sqlite3.Row | None", row)


def list_teams(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """List all teams."""
    return conn.execute("SELECT * FROM teams ORDER BY name").fetchall()


def delete_team(conn: sqlite3.Connection, name: str) -> bool:
    """Delete a team by name. Refuses to delete 'default'. Returns True if deleted."""
    if name == "default":
        raise ValueError("Cannot delete the 'default' team")
    cursor = conn.execute("DELETE FROM teams WHERE name = ?", (name,))
    conn.commit()
    return cursor.rowcount > 0


# --- Voiceprint CRUD ---


def save_voiceprint(
    conn: sqlite3.Connection,
    team_id: int,
    name: str,
    embedding: list[float],
    model: str,
) -> int:
    """Save or update a voiceprint. Returns its id."""
    embedding_json = json.dumps(embedding)
    cursor = conn.execute(
        """INSERT INTO voiceprints (team_id, name, embedding, model)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(team_id, name) DO UPDATE SET
               embedding = excluded.embedding,
               model = excluded.model,
               created_at = datetime('now')""",
        (team_id, name, embedding_json, model),
    )
    conn.commit()
    logger.info("Saved voiceprint for '%s' (team_id=%d)", name, team_id)
    return cursor.lastrowid  # type: ignore[return-value]


def load_voiceprints(conn: sqlite3.Connection, team_id: int) -> dict[str, list[float]]:
    """Load all voiceprints for a team. Returns dict mapping name -> embedding."""
    rows = conn.execute(
        "SELECT name, embedding FROM voiceprints WHERE team_id = ?",
        (team_id,),
    ).fetchall()
    return {row["name"]: json.loads(row["embedding"]) for row in rows}


def delete_voiceprint(conn: sqlite3.Connection, team_id: int, name: str) -> bool:
    """Delete a voiceprint by team and name. Returns True if deleted."""
    cursor = conn.execute(
        "DELETE FROM voiceprints WHERE team_id = ? AND name = ?",
        (team_id, name),
    )
    conn.commit()
    return cursor.rowcount > 0


def count_voiceprints(conn: sqlite3.Connection, team_id: int) -> int:
    """Count voiceprints for a team."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM voiceprints WHERE team_id = ?",
        (team_id,),
    ).fetchone()
    return cast(int, row["cnt"])


# --- User CRUD ---


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    team_id: int,
    is_admin: bool = False,
) -> int:
    """Create a user. Returns its id."""
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, team_id, is_admin) VALUES (?, ?, ?, ?)",
        (username, password_hash, team_id, int(is_admin)),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    """Fetch a user by username (with team name)."""
    return cast(
        "sqlite3.Row | None",
        conn.execute(
            "SELECT u.*, t.name as team_name FROM users u JOIN teams t ON u.team_id = t.id "
            "WHERE u.username = ?",
            (username,),
        ).fetchone(),
    )


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    """Fetch a user by id (with team name)."""
    return cast(
        "sqlite3.Row | None",
        conn.execute(
            "SELECT u.*, t.name as team_name FROM users u JOIN teams t ON u.team_id = t.id "
            "WHERE u.id = ?",
            (user_id,),
        ).fetchone(),
    )


def list_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """List all users with team names."""
    return conn.execute(
        "SELECT u.id, u.username, t.name as team_name, u.created_at "
        "FROM users u JOIN teams t ON u.team_id = t.id ORDER BY u.username"
    ).fetchall()


def delete_user(conn: sqlite3.Connection, username: str) -> bool:
    """Delete a user by username. Returns True if deleted."""
    cursor = conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    return cursor.rowcount > 0


# --- Auth session CRUD ---


def create_auth_session(
    conn: sqlite3.Connection, user_id: int, token: str, expires_at: str
) -> None:
    """Create an auth session."""
    conn.execute(
        "INSERT INTO auth_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at),
    )
    conn.commit()


def get_auth_session(conn: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    """Get auth session with user and team info. Returns None if expired or not found."""
    return cast(
        "sqlite3.Row | None",
        conn.execute(
            "SELECT s.token, s.expires_at, u.id as user_id, u.username, "
            "u.team_id, u.is_admin, t.name as team_name "
            "FROM auth_sessions s "
            "JOIN users u ON s.user_id = u.id "
            "JOIN teams t ON u.team_id = t.id "
            "WHERE s.token = ? AND s.expires_at > datetime('now')",
            (token,),
        ).fetchone(),
    )


def delete_auth_session(conn: sqlite3.Connection, token: str) -> bool:
    """Delete an auth session. Returns True if deleted."""
    cursor = conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
    conn.commit()
    return cursor.rowcount > 0


def delete_expired_sessions(conn: sqlite3.Connection) -> int:
    """Delete expired auth sessions. Returns count deleted."""
    cursor = conn.execute("DELETE FROM auth_sessions WHERE expires_at <= datetime('now')")
    conn.commit()
    return cursor.rowcount
