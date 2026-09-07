"""Session state management service backed by SQLite."""

import shutil
import sqlite3
import uuid
from pathlib import Path

from meetscribe import config
from meetscribe.config import get_config
from meetscribe.database import get_db, get_team
from meetscribe.pipeline.models import TranscriptSegment, format_transcript_markdown

from ..models import (
    Sample,
    SessionState,
    SessionStatus,
    SessionSummary,
    SpeakerBin,
    TrackConfig,
    TranscriptSegmentModel,
)


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

    # --- Core CRUD ---

    def create(self, team_name: str = "default", creator_id: int | None = None) -> SessionState:
        """Create a new session."""
        session_id = str(uuid.uuid4())

        # Create filesystem dirs for audio files
        self._tracks_dir(session_id).mkdir(parents=True, exist_ok=True)
        self._samples_dir(session_id).mkdir(parents=True, exist_ok=True)

        conn = get_db()
        team = get_team(conn, team_name)
        if not team:
            raise ValueError(f"Team '{team_name}' not found")
        cfg = get_config()
        language = cfg.transcription.language if cfg.transcription else "ru"
        conn.execute(
            "INSERT INTO sessions (id, team_id, status, language, creator_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, team["id"], SessionStatus.CREATED.value, language, creator_id),
        )
        conn.commit()

        return SessionState(id=session_id, team_name=team_name, creator_id=creator_id)

    def get(self, session_id: str) -> SessionState | None:
        """Get full session state."""
        conn = get_db()
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
                open_space=bool(t["open_space"]),
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

        segments = [
            TranscriptSegmentModel(
                id=seg["id"],
                track_num=seg["track_num"],
                start_ms=seg["start_ms"],
                end_ms=seg["end_ms"],
                speaker=seg["speaker"],
                text=seg["text"],
            )
            for seg in conn.execute(
                "SELECT * FROM session_segments WHERE session_id = ? ORDER BY sort_order",
                (session_id,),
            ).fetchall()
        ]

        return SessionState(
            id=row["id"],
            status=SessionStatus(row["status"]),
            team_name=row["team_name"],
            creator_id=row["creator_id"],
            tracks=tracks,
            speakers=speakers,
            samples=samples,
            transcript=row["transcript"],
            segments=segments,
            language=row["language"],
        )

    def update(self, state: SessionState) -> None:
        """Update session status, language, transcript."""
        conn = get_db()
        conn.execute(
            "UPDATE sessions SET status = ?, language = ?, transcript = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (state.status.value, state.language, state.transcript, state.id),
        )
        conn.commit()

    def delete(self, session_id: str) -> bool:
        """Delete a session and its files."""
        conn = get_db()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

        # Clean up filesystem
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

        return deleted

    def delete_many_for_user(
        self, ids: list[str], team_id: int, creator_id: int | None = None
    ) -> int:
        """Delete the subset of ids the caller may access. Returns count deleted.

        Access predicate matches list_summaries: NULL creator_id means no creator
        filter (admin acting on the whole team). Inaccessible ids are skipped.
        """
        if not ids:
            return 0
        conn = get_db()
        placeholders = ",".join("?" * len(ids))
        # B608: the interpolation is "?" placeholders only; values are bound.
        rows = conn.execute(
            f"SELECT id FROM sessions WHERE id IN ({placeholders}) "  # nosec B608
            f"AND team_id = ? AND (? IS NULL OR creator_id = ?)",
            (*ids, team_id, creator_id, creator_id),
        ).fetchall()
        return sum(1 for row in rows if self.delete(row["id"]))

    # ORDER BY whitelist — never interpolate user input into SQL.
    # rowid breaks ties: created_at has 1-second resolution.
    _SORT_SQL = {
        ("date", "asc"): "s.created_at ASC, s.rowid ASC",
        ("date", "desc"): "s.created_at DESC, s.rowid DESC",
        ("duration", "asc"): "duration_ms ASC NULLS LAST, s.created_at DESC, s.rowid DESC",
        ("duration", "desc"): "duration_ms DESC NULLS LAST, s.created_at DESC, s.rowid DESC",
    }

    def list_summaries(
        self,
        team_id: int,
        page: int = 1,
        per_page: int = 20,
        sort: str = "date",
        order: str = "desc",
        creator_id: int | None = None,
    ) -> tuple[list[SessionSummary], int]:
        """Return one page of session summaries for a team, plus the total count."""
        order_by = self._SORT_SQL[(sort, order)]

        # NULL creator_id disables the filter (admin viewing the whole team)
        params = (team_id, creator_id, creator_id)

        conn = get_db()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions s "
            "WHERE s.team_id = ? AND (? IS NULL OR s.creator_id = ?)",
            params,
        ).fetchone()["cnt"]

        # B608: order_by comes from the _SORT_SQL whitelist; all values are bound.
        rows = conn.execute(
            f"""
            SELECT
                s.id,
                s.status,
                s.created_at,
                u.username AS creator,
                SUBSTR(s.transcript, 1, 150) AS preview,
                (SELECT COUNT(*) FROM session_tracks tr
                  WHERE tr.session_id = s.id) AS track_count,
                (SELECT MAX(seg.end_ms) FROM session_segments seg
                  WHERE seg.session_id = s.id) AS duration_ms
            FROM sessions s
            LEFT JOIN users u ON u.id = s.creator_id
            WHERE s.team_id = ? AND (? IS NULL OR s.creator_id = ?)
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,  # nosec B608
            params + (per_page, (page - 1) * per_page),
        ).fetchall()

        # Speakers per session in one query for the page's ids (not GROUP_CONCAT:
        # speaker names are arbitrary text, splitting on a separator is unsafe).
        # B608: the interpolation is "?" placeholders only; ids are bound.
        speakers: dict[str, list[str]] = {}
        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            for r in conn.execute(
                f"SELECT DISTINCT session_id, speaker FROM session_segments "  # nosec B608
                f"WHERE session_id IN ({placeholders}) AND speaker IS NOT NULL "
                f"ORDER BY session_id, speaker",
                ids,
            ).fetchall():
                speakers.setdefault(r["session_id"], []).append(r["speaker"])

        summaries = [
            SessionSummary(
                id=r["id"],
                status=SessionStatus(r["status"]),
                created_at=r["created_at"],
                creator=r["creator"],
                track_count=r["track_count"],
                duration_ms=r["duration_ms"],
                speakers=speakers.get(r["id"], []),
                preview=r["preview"],
            )
            for r in rows
        ]
        return summaries, total

    # --- Tracks ---

    def add_track(self, session_id: str, filename: str, source_path: Path) -> TrackConfig:
        """Add a track file to the session by moving from source_path."""
        conn = get_db()
        track_path = None
        try:
            conn.execute("BEGIN IMMEDIATE")
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

            track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
            source_path.rename(track_path)

            conn.commit()
        except Exception:
            conn.rollback()
            if track_path and track_path.exists() and not source_path.exists():
                track_path.rename(source_path)
            raise

        return TrackConfig(track_num=track_num, filename=filename)

    def remove_track(self, session_id: str, track_num: int) -> bool:
        """Remove a track from the session."""
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM session_tracks WHERE session_id = ? AND track_num = ?",
                (session_id, track_num),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return False

            track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
            if track_path.exists():
                track_path.unlink()

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

    def get_track_path(self, session_id: str, track_num: int) -> Path | None:
        """Get path to a track file."""
        path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
        return path if path.exists() else None

    def update_track_config(
        self,
        session_id: str,
        track_num: int,
        speaker_name: str | None,
        diarize: bool,
        open_space: bool = False,
    ) -> bool:
        """Update track configuration."""
        # Open-space filtering only applies to named (non-diarized) tracks.
        open_space = open_space and not diarize
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE session_tracks SET speaker_name = ?, diarize = ?, open_space = ? "
                "WHERE session_id = ? AND track_num = ?",
                (speaker_name, int(diarize), int(open_space), session_id, track_num),
            )
            if cursor.rowcount == 0:
                conn.rollback()
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

    # --- Speakers ---

    def add_speaker(self, session_id: str, name: str) -> SpeakerBin:
        """Create a speaker bin."""
        conn = get_db()
        speaker_id = str(uuid.uuid4())[:8]
        conn.execute(
            "INSERT INTO session_speakers (id, session_id, name) VALUES (?, ?, ?)",
            (speaker_id, session_id, name),
        )
        conn.commit()
        return SpeakerBin(id=speaker_id, name=name)

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> bool:
        """Rename a speaker bin."""
        conn = get_db()
        cursor = conn.execute(
            "UPDATE session_speakers SET name = ? WHERE session_id = ? AND id = ?",
            (name, session_id, speaker_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_speaker(self, session_id: str, speaker_id: str) -> bool:
        """Delete a speaker bin and unassign its samples."""
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM session_speakers WHERE session_id = ? AND id = ?",
                (session_id, speaker_id),
            )
            if cursor.rowcount == 0:
                conn.rollback()
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

        conn = get_db()
        try:
            conn.execute(
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
            conn.commit()
        except Exception:
            sample_path.unlink(missing_ok=True)
            conn.rollback()
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
        conn = get_db()
        cursor = conn.execute(
            "UPDATE session_samples SET speaker_id = ? WHERE session_id = ? AND id = ?",
            (speaker_id, session_id, sample_id),
        )
        if cursor.rowcount == 0:
            return False
        conn.commit()

        return True

    def delete_sample(self, session_id: str, sample_id: str) -> bool:
        """Delete a sample."""
        conn = get_db()
        cursor = conn.execute(
            "DELETE FROM session_samples WHERE session_id = ? AND id = ?",
            (session_id, sample_id),
        )
        if cursor.rowcount == 0:
            return False
        conn.commit()

        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
        sample_path.unlink(missing_ok=True)
        return True

    def get_sample_path(self, session_id: str, sample_id: str) -> Path | None:
        """Get path to a sample file."""
        path = self._samples_dir(session_id) / f"{sample_id}.wav"
        return path if path.exists() else None

    def set_transcript(self, session_id: str, transcript: str) -> None:
        """Set the transcript."""
        conn = get_db()
        cursor = conn.execute(
            "UPDATE sessions SET transcript = ?, status = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (transcript, SessionStatus.TRANSCRIBED.value, session_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Session not found: {session_id}")
        conn.commit()

    # --- Segment editing ---

    def _renumber_segments(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Rewrite sort_order as a contiguous 0..n-1 sequence (ties broken by id)."""
        rows = conn.execute(
            "SELECT id FROM session_segments WHERE session_id = ? ORDER BY sort_order, id",
            (session_id,),
        ).fetchall()
        conn.executemany(
            "UPDATE session_segments SET sort_order = ? WHERE id = ?",
            [(i, row["id"]) for i, row in enumerate(rows)],
        )

    def _regenerate_transcript(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Rebuild sessions.transcript from the current segments."""
        rows = conn.execute(
            "SELECT * FROM session_segments WHERE session_id = ? ORDER BY sort_order",
            (session_id,),
        ).fetchall()
        segments = [
            TranscriptSegment(
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                text=r["text"],
                speaker=r["speaker"],
                track_num=r["track_num"],
            )
            for r in rows
        ]
        conn.execute(
            "UPDATE sessions SET transcript = ?, updated_at = datetime('now') WHERE id = ?",
            (format_transcript_markdown(segments), session_id),
        )

    def update_segment(
        self,
        session_id: str,
        segment_id: int,
        text: str | None,
        speaker: str | None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        clear_speaker: bool = False,
    ) -> bool:
        """Update a segment's text, speaker and/or timing; regenerate the transcript.

        clear_speaker distinguishes "set speaker to Unknown" from "don't touch it"
        (both arrive as speaker=None). Raises ValueError if the resulting start is
        not before the end.
        """
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, start_ms, end_ms FROM session_segments WHERE session_id = ? AND id = ?",
                (session_id, segment_id),
            ).fetchone()
            if not row:
                conn.rollback()
                return False
            new_start = row["start_ms"] if start_ms is None else start_ms
            new_end = row["end_ms"] if end_ms is None else end_ms
            if new_start >= new_end:
                raise ValueError("Segment start must be before its end")
            if text is not None:
                conn.execute(
                    "UPDATE session_segments SET text = ? WHERE id = ?", (text, segment_id)
                )
            if speaker is not None or clear_speaker:
                conn.execute(
                    "UPDATE session_segments SET speaker = ? WHERE id = ?", (speaker, segment_id)
                )
            if start_ms is not None or end_ms is not None:
                conn.execute(
                    "UPDATE session_segments SET start_ms = ?, end_ms = ? WHERE id = ?",
                    (new_start, new_end, segment_id),
                )
            self._regenerate_transcript(conn, session_id)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    def insert_segment(
        self, session_id: str, after_id: int | None, text: str, speaker: str | None
    ) -> bool:
        """Insert a segment after `after_id` (or before the first one when None).

        Times default to the gap to the next segment; with no gap a 1-second
        stub is used — fix up via the timing edit. Returns False if after_id
        (or, for None, any segment to anchor on) doesn't exist.
        """
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if after_id is not None:
                prev = conn.execute(
                    "SELECT * FROM session_segments WHERE session_id = ? AND id = ?",
                    (session_id, after_id),
                ).fetchone()
                if not prev:
                    conn.rollback()
                    return False
                nxt = conn.execute(
                    "SELECT start_ms FROM session_segments "
                    "WHERE session_id = ? AND (sort_order, id) > (?, ?) "
                    "ORDER BY sort_order, id LIMIT 1",
                    (session_id, prev["sort_order"], prev["id"]),
                ).fetchone()
                start = prev["end_ms"]
                end = nxt["start_ms"] if nxt and nxt["start_ms"] > start else start + 1000
                track_num = prev["track_num"]
                # Same sort_order as prev: _renumber_segments breaks the tie by id,
                # placing the new (higher-id) row right after it.
                sort_order = prev["sort_order"]
            else:
                first = conn.execute(
                    "SELECT start_ms, track_num FROM session_segments "
                    "WHERE session_id = ? ORDER BY sort_order, id LIMIT 1",
                    (session_id,),
                ).fetchone()
                if not first:
                    conn.rollback()
                    return False
                start = max(0, first["start_ms"] - 1000)
                end = first["start_ms"] if first["start_ms"] > start else start + 1000
                track_num = first["track_num"]
                sort_order = -1
            conn.execute(
                "INSERT INTO session_segments "
                "(session_id, track_num, start_ms, end_ms, speaker, text, sort_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, track_num, start, end, speaker, text, sort_order),
            )
            self._renumber_segments(conn, session_id)
            self._regenerate_transcript(conn, session_id)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    def delete_segment(self, session_id: str, segment_id: int) -> bool:
        """Delete a segment; renumber the rest and regenerate the transcript."""
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM session_segments WHERE session_id = ? AND id = ?",
                (session_id, segment_id),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return False
            self._renumber_segments(conn, session_id)
            self._regenerate_transcript(conn, session_id)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    def merge_segment_with_next(self, session_id: str, segment_id: int) -> bool:
        """Merge a segment with the next one by sort_order.

        Returns False if the segment doesn't exist.
        Raises ValueError if it's the last segment.
        """
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            first = conn.execute(
                "SELECT * FROM session_segments WHERE session_id = ? AND id = ?",
                (session_id, segment_id),
            ).fetchone()
            if not first:
                conn.rollback()
                return False
            second = conn.execute(
                "SELECT * FROM session_segments WHERE session_id = ? AND sort_order > ? "
                "ORDER BY sort_order LIMIT 1",
                (session_id, first["sort_order"]),
            ).fetchone()
            if not second:
                raise ValueError("No next segment to merge with")
            merged_text = " ".join(t for t in (first["text"], second["text"]) if t)
            # Full span, not first.start..second.end: timing edits can reorder them
            conn.execute(
                "UPDATE session_segments SET text = ?, start_ms = ?, end_ms = ? WHERE id = ?",
                (
                    merged_text,
                    min(first["start_ms"], second["start_ms"]),
                    max(first["end_ms"], second["end_ms"]),
                    first["id"],
                ),
            )
            conn.execute("DELETE FROM session_segments WHERE id = ?", (second["id"],))
            self._renumber_segments(conn, session_id)
            self._regenerate_transcript(conn, session_id)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    def split_segment(self, session_id: str, segment_id: int, offset: int) -> bool:
        """Split a segment at a char offset; time is divided proportionally.

        Returns False if the segment doesn't exist. Raises ValueError for an
        offset that doesn't leave text on both sides.
        """
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            seg = conn.execute(
                "SELECT * FROM session_segments WHERE session_id = ? AND id = ?",
                (session_id, segment_id),
            ).fetchone()
            if not seg:
                conn.rollback()
                return False
            text = seg["text"]
            first_text = text[:offset].rstrip() if offset > 0 else ""
            second_text = text[offset:].lstrip() if offset > 0 else ""
            if not (0 < offset < len(text)) or not first_text or not second_text:
                raise ValueError("Split offset must leave text on both sides")
            mid_ms = seg["start_ms"] + (seg["end_ms"] - seg["start_ms"]) * offset // len(text)
            if not seg["start_ms"] < mid_ms < seg["end_ms"]:
                raise ValueError("Segment is too short to split")
            conn.execute(
                "UPDATE session_segments SET text = ?, end_ms = ? WHERE id = ?",
                (first_text, mid_ms, seg["id"]),
            )
            # Same sort_order as the first half: _renumber_segments breaks the
            # tie by id, placing the new (higher-id) row right after it.
            conn.execute(
                "INSERT INTO session_segments "
                "(session_id, track_num, start_ms, end_ms, speaker, text, sort_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    seg["track_num"],
                    mid_ms,
                    seg["end_ms"],
                    seg["speaker"],
                    second_text,
                    seg["sort_order"],
                ),
            )
            self._renumber_segments(conn, session_id)
            self._regenerate_transcript(conn, session_id)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    def save_segments(self, session_id: str, segments: list[dict]) -> None:
        """Save structured transcript segments."""
        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM session_segments WHERE session_id = ?", (session_id,))
            rows = [
                (
                    session_id,
                    seg["track_num"],
                    seg["start_ms"],
                    seg["end_ms"],
                    seg.get("speaker"),
                    seg["text"],
                    i,
                )
                for i, seg in enumerate(segments)
            ]
            conn.executemany(
                "INSERT INTO session_segments "
                "(session_id, track_num, start_ms, end_ms, speaker, text, sort_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# Singleton (no state beyond sessions_dir)
_session_service: SessionService | None = None


def init_session_service(sessions_dir: Path | None = None) -> SessionService:
    """Initialize the session service singleton."""
    global _session_service
    _session_service = SessionService(sessions_dir)
    return _session_service


def get_session_service() -> SessionService:
    """Get the session service singleton."""
    if _session_service is None:
        raise RuntimeError("SessionService not initialized — call init_session_service() first")
    return _session_service
