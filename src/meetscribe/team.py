"""Team context resolution."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config
from .database import ensure_default_team, get_db, get_team


@dataclass
class TeamContext:
    """Resolved team with its id and scoped paths."""

    id: int
    name: str
    conn: sqlite3.Connection
    samples_dir: Path
    enrolled_samples_dir: Path
    unknown_samples_dir: Path


def resolve_team(team_name: str | None = None) -> TeamContext:
    """Resolve a team name into a TeamContext.

    If team_name is None, uses 'default'.
    Validates the team exists in the DB and ensures its directories exist.
    """
    name = team_name or "default"
    conn = get_db(config.DB_PATH)

    team = get_team(conn, name)
    if team is None:
        if name == "default":
            ensure_default_team(conn)
            team = get_team(conn, name)
            if team is None:
                raise RuntimeError("Failed to create default team")
        else:
            raise ValueError(
                f"Team '{name}' not found. Create it with: meetscribe team create {name}"
            )

    config.ensure_team_dirs(name)

    return TeamContext(
        id=team["id"],
        name=name,
        conn=conn,
        samples_dir=config.get_team_samples_dir(name),
        enrolled_samples_dir=config.get_team_enrolled_dir(name),
        unknown_samples_dir=config.get_team_unknown_dir(name),
    )
