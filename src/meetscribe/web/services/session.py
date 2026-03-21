"""Session state management service backed by SQLite."""

import shutil
import uuid
from pathlib import Path

from meetscribe import config
from meetscribe.database import get_db, get_team, load_voiceprints
from meetscribe.team import resolve_team

from ..models import Sample, SessionState, SessionStatus, SpeakerBin, TrackConfig

# Session TTL in seconds (2 hours)
SESSION_TTL = 2 * 60 * 60


class SessionService:
    """Manages session state and files via SQLite + filesystem."""

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or config.DATA_DIR / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _tracks_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "tracks"

    def _samples_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "samples"

    def _conn(self):
        return get_db(config.DB_PATH)

    # --- Core CRUD ---

    def create(self, team_name: str = "default") -> SessionState:
        """Create a new session."""
        session_id = str(uuid.uuid4())

        # Create filesystem dirs for audio files
        self._tracks_dir(session_id).mkdir(parents=True, exist_ok=True)
        self._samples_dir(session_id).mkdir(parents=True, exist_ok=True)

        conn = self._conn()
        try:
            team = get_team(conn, team_name)
            if not team:
                raise ValueError(f"Team '{team_name}' not found")
            conn.execute(
                "INSERT INTO sessions (id, team_id, status, language) VALUES (?, ?, ?, ?)",
                (session_id, team["id"], SessionStatus.CREATED.value, "ru"),
            )
            conn.commit()
        finally:
            conn.close()

        return SessionState(id=session_id, team_name=team_name)

    def get(self, session_id: str) -> SessionState | None:
        """Get full session state."""
        conn = self._conn()
        try:
            row = conn.execute(
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
                for t in conn.execute(
                    "SELECT * FROM session_tracks WHERE session_id = ? ORDER BY track_num",
                    (session_id,),
                ).fetchall()
            ]

            speakers = [
                SpeakerBin(id=sp["id"], name=sp["name"])
                for sp in conn.execute(
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
                for sa in conn.execute(
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
        finally:
            conn.close()

    def update(self, state: SessionState) -> None:
        """Update session status, language, transcript."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE sessions SET status = ?, language = ?, transcript = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (state.status.value, state.language, state.transcript, state.id),
            )
            conn.commit()
        finally:
            conn.close()

    def delete(self, session_id: str) -> bool:
        """Delete a session and its files."""
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        finally:
            conn.close()

        # Clean up filesystem
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

        return deleted

    # --- Tracks ---

    def add_track(self, session_id: str, filename: str, source_path: Path) -> TrackConfig:
        """Add a track file to the session by moving from source_path."""
        conn = self._conn()
        track_path = None
        try:
            # Exclusive lock prevents concurrent track_num assignment
            conn.execute("BEGIN EXCLUSIVE")
            row = conn.execute(
                "SELECT COALESCE(MAX(track_num), 0) + 1 as next_num "
                "FROM session_tracks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            track_num = row["next_num"]

            conn.execute(
                "INSERT INTO session_tracks (session_id, track_num, filename) VALUES (?, ?, ?)",
                (session_id, track_num, filename),
            )
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (SessionStatus.UPLOADED.value, session_id),
            )

            # Move file after DB insert succeeds (still within exclusive lock)
            track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
            source_path.rename(track_path)

            conn.commit()
        except Exception:
            conn.rollback()
            # If file was moved but transaction rolled back, move it back
            if track_path and track_path.exists() and not source_path.exists():
                track_path.rename(source_path)
            raise
        finally:
            conn.close()

        return TrackConfig(track_num=track_num, filename=filename)

    def remove_track(self, session_id: str, track_num: int) -> bool:
        """Remove a track from the session."""
        conn = self._conn()
        try:
            cursor = conn.execute(
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
            remaining = conn.execute(
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
                    conn.execute(
                        "UPDATE session_tracks SET track_num = ? "
                        "WHERE session_id = ? AND track_num = ?",
                        (new_idx, session_id, old_num),
                    )

            # Update status if no tracks left
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM session_tracks WHERE session_id = ?",
                (session_id,),
            ).fetchone()["cnt"]
            if count == 0:
                conn.execute(
                    "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                    (SessionStatus.CREATED.value, session_id),
                )

            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_track_path(self, session_id: str, track_num: int) -> Path | None:
        """Get path to a track file."""
        path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
        return path if path.exists() else None

    def update_track_config(
        self, session_id: str, track_num: int, speaker_name: str | None, diarize: bool
    ) -> bool:
        """Update track configuration."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE session_tracks SET speaker_name = ?, diarize = ? "
                "WHERE session_id = ? AND track_num = ?",
                (speaker_name, int(diarize), session_id, track_num),
            )
            if cursor.rowcount == 0:
                return False
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (SessionStatus.CONFIGURED.value, session_id),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Speakers ---

    def add_speaker(self, session_id: str, name: str) -> SpeakerBin:
        """Create a speaker bin."""
        speaker_id = str(uuid.uuid4())[:8]
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO session_speakers (id, session_id, name) VALUES (?, ?, ?)",
                (speaker_id, session_id, name),
            )
            conn.commit()
        finally:
            conn.close()
        return SpeakerBin(id=speaker_id, name=name)

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> bool:
        """Rename a speaker bin."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE session_speakers SET name = ? WHERE session_id = ? AND id = ?",
                (name, session_id, speaker_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_speaker(self, session_id: str, speaker_id: str) -> bool:
        """Delete a speaker bin and unassign its samples."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM session_speakers WHERE session_id = ? AND id = ?",
                (session_id, speaker_id),
            )
            if cursor.rowcount == 0:
                return False
            conn.execute(
                "UPDATE session_samples SET speaker_id = NULL "
                "WHERE session_id = ? AND speaker_id = ?",
                (session_id, speaker_id),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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

        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO session_samples "
                "(id, session_id, track_num, cluster_id, filename, duration_ms, is_known, known_speaker_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, session_id, track_num, cluster_id, filename, duration_ms, int(is_known), known_speaker_name),
            )
            conn.commit()
        except Exception:
            sample_path.unlink(missing_ok=True)
            conn.rollback()
            raise
        finally:
            conn.close()

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
        self, session_id: str, sample_id: str, speaker_id: str | None, speaker_name: str | None = None
    ) -> bool:
        """Move sample to speaker bin."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE session_samples SET speaker_id = ? WHERE session_id = ? AND id = ?",
                (speaker_id, session_id, sample_id),
            )
            if cursor.rowcount == 0:
                return False
            conn.commit()
        finally:
            conn.close()

        # If speaker is enrolled, copy sample to team-scoped enrolled folder
        if speaker_name:
            state = self.get(session_id)
            if state:
                team_ctx = resolve_team(state.team_name)
                try:
                    voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)
                    if speaker_name in voiceprints:
                        enrolled_dir = team_ctx.enrolled_samples_dir / speaker_name
                        enrolled_dir.mkdir(parents=True, exist_ok=True)
                        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
                        if sample_path.exists():
                            dest = enrolled_dir / f"manual_{sample_id}.wav"
                            shutil.copy2(sample_path, dest)
                finally:
                    team_ctx.conn.close()

        return True

    def delete_sample(self, session_id: str, sample_id: str) -> bool:
        """Delete a sample."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM session_samples WHERE session_id = ? AND id = ?",
                (session_id, sample_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return False
        finally:
            conn.close()

        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
        sample_path.unlink(missing_ok=True)
        return True

    def get_sample_path(self, session_id: str, sample_id: str) -> Path | None:
        """Get path to a sample file."""
        path = self._samples_dir(session_id) / f"{sample_id}.wav"
        return path if path.exists() else None

    def set_transcript(self, session_id: str, transcript: str) -> None:
        """Set the transcript."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE sessions SET transcript = ?, status = ?, updated_at = datetime('now') "
                "WHERE id = ?",
                (transcript, SessionStatus.TRANSCRIBED.value, session_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Session not found: {session_id}")
            conn.commit()
        finally:
            conn.close()

    # --- Cleanup ---

    def cleanup_old_sessions(self) -> int:
        """Remove sessions older than TTL. Returns count of removed sessions."""
        conn = self._conn()
        try:
            ttl_minutes = SESSION_TTL // 60
            expired = conn.execute(
                "SELECT id FROM sessions WHERE updated_at < datetime('now', ?)",
                (f"-{ttl_minutes} minutes",),
            ).fetchall()

            if not expired:
                return 0

            ids = [r["id"] for r in expired]
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids)
            conn.commit()
        finally:
            conn.close()

        # Clean up filesystem
        for session_id in ids:
            session_dir = self._session_dir(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir)

        return len(ids)


# Singleton instance
_session_service: SessionService | None = None


def get_session_service() -> SessionService:
    """Get the session service singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
