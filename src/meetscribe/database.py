"""SQLite database for team and voiceprint management."""

import json
import logging
import re
import sqlite3
from pathlib import Path

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


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
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
    """)
    conn.commit()


def ensure_default_team(conn: sqlite3.Connection) -> None:
    """Create the 'default' team if it doesn't exist."""
    conn.execute(
        "INSERT OR IGNORE INTO teams (name, description) VALUES (?, ?)",
        ("default", "Default team"),
    )
    conn.commit()


def create_team(conn: sqlite3.Connection, name: str, description: str | None = None) -> int:
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
    return conn.execute("SELECT * FROM teams WHERE name = ?", (name,)).fetchone()


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
    return cursor.lastrowid


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
    return row["cnt"]
