"""Session state management service backed by SQLite."""

import shutil
import sqlite3
import uuid
from pathlib import Path

from meetscribe import config
from meetscribe.config import get_config
from meetscribe.database import get_team

from ..models import Sample, SessionState, SessionStatus, SpeakerBin, TrackConfig

# Session TTL in seconds (2 hours)
SESSION_TTL = 2 * 60 * 60


class SessionService:
    """Manages session state and files via SQLite + filesystem."""

    def __init__(self, conn: sqlite3.Connection, sessions_dir: Path | None = None):
        self.conn = conn
        self.sessions_dir = sessions_dir or config.DATA_DIR / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _tracks_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "tracks"

    def _samples_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "samples"

    def _conn(self) -> sqlite3.Connection:
        return self.conn

    # --- Core CRUD ---

    def create(self, team_name: str = "default") -> SessionState:
        """Create a new session."""
        session_id = str(uuid.uuid4())

        # Create filesystem dirs for audio files
        self._tracks_dir(session_id).mkdir(parents=True, exist_ok=True)
        self._samples_dir(session_id).mkdir(parents=True, exist_ok=True)

        team = get_team(self.conn, team_name)
        if not team:
            raise ValueError(f"Team '{team_name}' not found")
        cfg = get_config()
        language = cfg.transcription.language if cfg.transcription else "ru"
        self.conn.execute(
            "INSERT INTO sessions (id, team_id, status, language) VALUES (?, ?, ?, ?)",
            (session_id, team["id"], SessionStatus.CREATED.value, language),
        )
        self.conn.commit()

        return SessionState(id=session_id, team_name=team_name)

    def get(self, session_id: str) -> SessionState | None:
        """Get full session state."""
        row = self.conn.execute(
            "SELECT s.*, t.name as team_name FROM sessions s "
            "JOIN teams t ON s.team_id = t.id WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None

        tracks = [
            TrackConfig(
                track_num=t["track_num"],
                filename=t["filename"],
                speaker_name=t["speaker_name"],
                diarize=bool(t["diarize"]),
            )
            for t in self.conn.execute(
                "SELECT * FROM session_tracks WHERE session_id = ? ORDER BY track_num",
                (session_id,),
            ).fetchall()
        ]

        speakers = [
            SpeakerBin(id=sp["id"], name=sp["name"])
            for sp in self.conn.execute(
                "SELECT * FROM session_speakers WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        ]

        samples = [
            Sample(
                id=sa["id"],
                track_num=sa["track_num"],
                cluster_id=sa["cluster_id"],
                filename=sa["filename"],
                duration_ms=sa["duration_ms"],
                speaker_id=sa["speaker_id"],
                is_known=bool(sa["is_known"]),
                known_speaker_name=sa["known_speaker_name"],
            )
            for sa in self.conn.execute(
                "SELECT * FROM session_samples WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        ]

        return SessionState(
            id=row["id"],
            status=SessionStatus(row["status"]),
            team_name=row["team_name"],
            tracks=tracks,
            speakers=speakers,
            samples=samples,
            transcript=row["transcript"],
            language=row["language"],
        )

    def update(self, state: SessionState) -> None:
        """Update session status, language, transcript."""
        self.conn.execute(
            "UPDATE sessions SET status = ?, language = ?, transcript = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (state.status.value, state.language, state.transcript, state.id),
        )
        self.conn.commit()

    def delete(self, session_id: str) -> bool:
        """Delete a session and its files."""
        cursor = self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()
        deleted = cursor.rowcount > 0

        # Clean up filesystem
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

        return deleted

    # --- Tracks ---

    def add_track(self, session_id: str, filename: str, source_path: Path) -> TrackConfig:
        """Add a track file to the session by moving from source_path."""
        track_path = None
        try:
            # Exclusive lock prevents concurrent track_num assignment
            self.conn.execute("BEGIN EXCLUSIVE")
            row = self.conn.execute(
                "SELECT COALESCE(MAX(track_num), 0) + 1 as next_num "
                "FROM session_tracks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            track_num = row["next_num"]

            self.conn.execute(
                "INSERT INTO session_tracks (session_id, track_num, filename) VALUES (?, ?, ?)",
                (session_id, track_num, filename),
            )
            self.conn.execute(
                "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (SessionStatus.UPLOADED.value, session_id),
            )

            # Move file after DB insert succeeds (still within exclusive lock)
            track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
            source_path.rename(track_path)

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            # If file was moved but transaction rolled back, move it back
            if track_path and track_path.exists() and not source_path.exists():
                track_path.rename(source_path)
            raise

        return TrackConfig(track_num=track_num, filename=filename)

    def remove_track(self, session_id: str, track_num: int) -> bool:
        """Remove a track from the session."""
        try:
            cursor = self.conn.execute(
                "DELETE FROM session_tracks WHERE session_id = ? AND track_num = ?",
                (session_id, track_num),
            )
            if cursor.rowcount == 0:
                return False

            # Remove file
            track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
            if track_path.exists():
                track_path.unlink()

            # Renumber remaining tracks
            remaining = self.conn.execute(
                "SELECT track_num FROM session_tracks WHERE session_id = ? ORDER BY track_num",
                (session_id,),
            ).fetchall()

            tracks_dir = self._tracks_dir(session_id)
            for new_idx, row in enumerate(remaining, 1):
                old_num = row["track_num"]
                if old_num != new_idx:
                    old_path = tracks_dir / f"track_{old_num}.wav"
                    new_path = tracks_dir / f"track_{new_idx}.wav"
                    if old_path.exists():
                        old_path.rename(new_path)
                    self.conn.execute(
                        "UPDATE session_tracks SET track_num = ? "
                        "WHERE session_id = ? AND track_num = ?",
                        (new_idx, session_id, old_num),
                    )

            # Update status if no tracks left
            count = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM session_tracks WHERE session_id = ?",
                (session_id,),
            ).fetchone()["cnt"]
            if count == 0:
                self.conn.execute(
                    "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                    (SessionStatus.CREATED.value, session_id),
                )

            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def get_track_path(self, session_id: str, track_num: int) -> Path | None:
        """Get path to a track file."""
        path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
        return path if path.exists() else None

    def update_track_config(
        self, session_id: str, track_num: int, speaker_name: str | None, diarize: bool
    ) -> bool:
        """Update track configuration."""
        try:
            cursor = self.conn.execute(
                "UPDATE session_tracks SET speaker_name = ?, diarize = ? "
                "WHERE session_id = ? AND track_num = ?",
                (speaker_name, int(diarize), session_id, track_num),
            )
            if cursor.rowcount == 0:
                return False
            self.conn.execute(
                "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (SessionStatus.CONFIGURED.value, session_id),
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    # --- Speakers ---

    def add_speaker(self, session_id: str, name: str) -> SpeakerBin:
        """Create a speaker bin."""
        speaker_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO session_speakers (id, session_id, name) VALUES (?, ?, ?)",
            (speaker_id, session_id, name),
        )
        self.conn.commit()
        return SpeakerBin(id=speaker_id, name=name)

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> bool:
        """Rename a speaker bin."""
        cursor = self.conn.execute(
            "UPDATE session_speakers SET name = ? WHERE session_id = ? AND id = ?",
            (name, session_id, speaker_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_speaker(self, session_id: str, speaker_id: str) -> bool:
        """Delete a speaker bin and unassign its samples."""
        try:
            cursor = self.conn.execute(
                "DELETE FROM session_speakers WHERE session_id = ? AND id = ?",
                (session_id, speaker_id),
            )
            if cursor.rowcount == 0:
                return False
            self.conn.execute(
                "UPDATE session_samples SET speaker_id = NULL "
                "WHERE session_id = ? AND speaker_id = ?",
                (session_id, speaker_id),
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    # --- Samples ---

    def add_sample(
        self,
        session_id: str,
        track_num: int,
        cluster_id: int,
        filename: str,
        duration_ms: int,
        content: bytes,
        is_known: bool = False,
        known_speaker_name: str | None = None,
    ) -> Sample:
        """Add a sample."""
        sample_id = str(uuid.uuid4())[:8]
        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"

        with open(sample_path, "wb") as f:
            f.write(content)

        try:
            self.conn.execute(
                "INSERT INTO session_samples "
                "(id, session_id, track_num, cluster_id, filename,"
                " duration_ms, is_known, known_speaker_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    session_id,
                    track_num,
                    cluster_id,
                    filename,
                    duration_ms,
                    int(is_known),
                    known_speaker_name,
                ),
            )
            self.conn.commit()
        except Exception:
            sample_path.unlink(missing_ok=True)
            self.conn.rollback()
            raise

        return Sample(
            id=sample_id,
            track_num=track_num,
            cluster_id=cluster_id,
            filename=filename,
            duration_ms=duration_ms,
            is_known=is_known,
            known_speaker_name=known_speaker_name,
        )

    def move_sample(
        self,
        session_id: str,
        sample_id: str,
        speaker_id: str | None,
        speaker_name: str | None = None,
    ) -> bool:
        """Move sample to speaker bin."""
        cursor = self.conn.execute(
            "UPDATE session_samples SET speaker_id = ? WHERE session_id = ? AND id = ?",
            (speaker_id, session_id, sample_id),
        )
        if cursor.rowcount == 0:
            return False
        self.conn.commit()

        return True

    def delete_sample(self, session_id: str, sample_id: str) -> bool:
        """Delete a sample."""
        cursor = self.conn.execute(
            "DELETE FROM session_samples WHERE session_id = ? AND id = ?",
            (session_id, sample_id),
        )
        if cursor.rowcount == 0:
            self.conn.rollback()
            return False
        self.conn.commit()

        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
        sample_path.unlink(missing_ok=True)
        return True

    def get_sample_path(self, session_id: str, sample_id: str) -> Path | None:
        """Get path to a sample file."""
        path = self._samples_dir(session_id) / f"{sample_id}.wav"
        return path if path.exists() else None

    def set_transcript(self, session_id: str, transcript: str) -> None:
        """Set the transcript."""
        cursor = self.conn.execute(
            "UPDATE sessions SET transcript = ?, status = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (transcript, SessionStatus.TRANSCRIBED.value, session_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Session not found: {session_id}")
        self.conn.commit()

    # --- Cleanup ---

    def cleanup_old_sessions(self) -> int:
        """Remove sessions older than TTL. Returns count of removed sessions."""
        ttl_minutes = SESSION_TTL // 60
        expired = self.conn.execute(
            "SELECT id FROM sessions WHERE updated_at < datetime('now', ?)",
            (f"-{ttl_minutes} minutes",),
        ).fetchall()

        if not expired:
            return 0

        ids = [r["id"] for r in expired]
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids)  # nosec B608
        self.conn.commit()

        # Clean up filesystem
        for session_id in ids:
            session_dir = self._session_dir(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir)

        return len(ids)


# Singleton instance
_session_service: SessionService | None = None


def init_session_service(conn: sqlite3.Connection) -> SessionService:
    """Initialize the session service singleton with a shared connection."""
    global _session_service
    _session_service = SessionService(conn)
    return _session_service


def get_session_service() -> SessionService:
    """Get the session service singleton."""
    if _session_service is None:
        raise RuntimeError("SessionService not initialized — call init_session_service(conn) first")
    return _session_service
