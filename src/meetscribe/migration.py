"""Migration from legacy flat file layout to team-scoped layout."""

import json
import logging
import shutil
import sqlite3

from . import config
from .database import get_db, get_team, save_voiceprint

logger = logging.getLogger(__name__)


def needs_migration() -> bool:
    """Check if there is legacy data that should be migrated."""
    if not config.VOICEPRINTS_DIR.exists():
        return False
    legacy_voiceprints = list(config.VOICEPRINTS_DIR.glob("*.json"))
    if not legacy_voiceprints:
        return False
    # Check if we already migrated (default team has voiceprints in DB)
    if config.DB_PATH.exists():
        conn = get_db(config.DB_PATH)
        team = get_team(conn, "default")
        if team:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM voiceprints WHERE team_id = ?",
                (team["id"],),
            ).fetchone()
            if row["cnt"] > 0:
                conn.close()
                return False
        conn.close()
    return True


MigrationResult = tuple[int, int]
"""(migrated_voiceprints, migrated_samples)"""


def migrate(conn: sqlite3.Connection) -> MigrationResult:
    """Migrate legacy voiceprints and samples to the 'default' team.

    Returns (migrated_voiceprints, migrated_samples).
    """
    team = get_team(conn, "default")
    if not team:
        return 0, 0

    team_id = team["id"]
    migrated_vp = 0
    migrated_samples = 0

    # Migrate voiceprints (JSON -> SQLite)
    if config.VOICEPRINTS_DIR.exists():
        for path in sorted(config.VOICEPRINTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data["name"]
                embedding = data["embedding"]
                model = data.get("model", "pyannote/wespeaker-voxceleb-resnet34-LM")
                save_voiceprint(conn, team_id, name, embedding, model)
                migrated_vp += 1
                logger.info("Migrated voiceprint: %s", name)
            except Exception:
                logger.warning("Failed to migrate voiceprint %s", path, exc_info=True)

    # Migrate enrolled samples (copy directories)
    if config.ENROLLED_SAMPLES_DIR.exists():
        dest = config.get_team_enrolled_dir("default")
        dest.mkdir(parents=True, exist_ok=True)
        for speaker_dir in config.ENROLLED_SAMPLES_DIR.iterdir():
            if speaker_dir.is_dir():
                dest_dir = dest / speaker_dir.name
                if not dest_dir.exists():
                    shutil.copytree(speaker_dir, dest_dir)
                    migrated_samples += 1
                    logger.info("Migrated enrolled samples: %s", speaker_dir.name)

    # Migrate unknown samples
    unknown_dir = config.SAMPLES_DIR / "unknown"
    if unknown_dir.exists():
        dest = config.get_team_unknown_dir("default")
        dest.mkdir(parents=True, exist_ok=True)
        for sample_dir in unknown_dir.iterdir():
            if sample_dir.is_dir():
                dest_dir = dest / sample_dir.name
                if not dest_dir.exists():
                    shutil.copytree(sample_dir, dest_dir)

    return migrated_vp, migrated_samples
