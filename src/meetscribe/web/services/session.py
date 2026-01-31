"""Session state management service."""

import json
import shutil
import time
import uuid
from pathlib import Path

from meetscribe import config

from ..models import Sample, SessionState, SessionStatus, SpeakerBin, TrackConfig

# Session TTL in seconds (2 hours)
SESSION_TTL = 2 * 60 * 60


class SessionService:
    """Manages session state and files."""

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or config.SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _state_file(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _tracks_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "tracks"

    def _samples_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "samples"

    def create(self) -> SessionState:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._tracks_dir(session_id).mkdir(exist_ok=True)
        self._samples_dir(session_id).mkdir(exist_ok=True)

        state = SessionState(id=session_id)
        self._save_state(state)
        return state

    def get(self, session_id: str) -> SessionState | None:
        """Get session state."""
        state_file = self._state_file(session_id)
        if not state_file.exists():
            return None
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)
        return SessionState(**data)

    def _save_state(self, state: SessionState) -> None:
        """Save session state."""
        state_file = self._state_file(state.id)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=2)

    def update(self, state: SessionState) -> None:
        """Update session state."""
        self._save_state(state)

    def delete(self, session_id: str) -> bool:
        """Delete a session and its files."""
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            return True
        return False

    def add_track(self, session_id: str, filename: str, content: bytes) -> TrackConfig:
        """Add a track file to the session."""
        state = self.get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        track_num = len(state.tracks) + 1
        track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"

        # Save file
        with open(track_path, "wb") as f:
            f.write(content)

        track = TrackConfig(track_num=track_num, filename=filename)
        state.tracks.append(track)
        state.status = SessionStatus.UPLOADED
        self._save_state(state)

        return track

    def remove_track(self, session_id: str, track_num: int) -> bool:
        """Remove a track from the session."""
        state = self.get(session_id)
        if not state:
            return False

        # Find and remove track
        track_idx = None
        for i, t in enumerate(state.tracks):
            if t.track_num == track_num:
                track_idx = i
                break

        if track_idx is None:
            return False

        # Remove file
        track_path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
        if track_path.exists():
            track_path.unlink()

        # Remove from state and renumber
        state.tracks.pop(track_idx)
        for i, t in enumerate(state.tracks):
            old_num = t.track_num
            new_num = i + 1
            if old_num != new_num:
                old_path = self._tracks_dir(session_id) / f"track_{old_num}.wav"
                new_path = self._tracks_dir(session_id) / f"track_{new_num}.wav"
                if old_path.exists():
                    old_path.rename(new_path)
                t.track_num = new_num

        if not state.tracks:
            state.status = SessionStatus.CREATED
        self._save_state(state)
        return True

    def get_track_path(self, session_id: str, track_num: int) -> Path | None:
        """Get path to a track file."""
        path = self._tracks_dir(session_id) / f"track_{track_num}.wav"
        return path if path.exists() else None

    def update_track_config(
        self, session_id: str, track_num: int, speaker_name: str | None, diarize: bool
    ) -> bool:
        """Update track configuration."""
        state = self.get(session_id)
        if not state:
            return False

        for track in state.tracks:
            if track.track_num == track_num:
                track.speaker_name = speaker_name
                track.diarize = diarize
                state.status = SessionStatus.CONFIGURED
                self._save_state(state)
                return True
        return False

    def add_speaker(self, session_id: str, name: str) -> SpeakerBin:
        """Create a speaker bin."""
        state = self.get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        speaker_id = str(uuid.uuid4())[:8]
        speaker = SpeakerBin(id=speaker_id, name=name)
        state.speakers.append(speaker)
        self._save_state(state)
        return speaker

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> bool:
        """Rename a speaker bin."""
        state = self.get(session_id)
        if not state:
            return False

        for speaker in state.speakers:
            if speaker.id == speaker_id:
                speaker.name = name
                self._save_state(state)
                return True
        return False

    def delete_speaker(self, session_id: str, speaker_id: str) -> bool:
        """Delete a speaker bin."""
        state = self.get(session_id)
        if not state:
            return False

        # Find and remove speaker
        speaker_idx = None
        for i, s in enumerate(state.speakers):
            if s.id == speaker_id:
                speaker_idx = i
                break

        if speaker_idx is None:
            return False

        # Unassign samples from this speaker
        for sample in state.samples:
            if sample.speaker_id == speaker_id:
                sample.speaker_id = None

        state.speakers.pop(speaker_idx)
        self._save_state(state)
        return True

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
        state = self.get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        sample_id = str(uuid.uuid4())[:8]
        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"

        with open(sample_path, "wb") as f:
            f.write(content)

        sample = Sample(
            id=sample_id,
            track_num=track_num,
            cluster_id=cluster_id,
            filename=filename,
            duration_ms=duration_ms,
            is_known=is_known,
            known_speaker_name=known_speaker_name,
        )
        state.samples.append(sample)
        self._save_state(state)
        return sample

    def move_sample(
        self, session_id: str, sample_id: str, speaker_id: str | None, speaker_name: str | None = None
    ) -> bool:
        """Move sample to speaker bin.

        If the speaker is enrolled (has voiceprint), copy sample to enrolled folder.
        """
        state = self.get(session_id)
        if not state:
            return False

        for sample in state.samples:
            if sample.id == sample_id:
                sample.speaker_id = speaker_id

                # If speaker is enrolled, copy sample to enrolled folder
                if speaker_name:
                    from meetscribe.config import ENROLLED_SAMPLES_DIR, VOICEPRINTS_DIR

                    speaker_file = VOICEPRINTS_DIR / f"{speaker_name}.json"
                    if speaker_file.exists():
                        enrolled_dir = ENROLLED_SAMPLES_DIR / speaker_name
                        enrolled_dir.mkdir(parents=True, exist_ok=True)
                        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
                        if sample_path.exists():
                            dest = enrolled_dir / f"manual_{sample_id}.wav"
                            shutil.copy2(sample_path, dest)

                self._save_state(state)
                return True
        return False

    def delete_sample(self, session_id: str, sample_id: str) -> bool:
        """Delete a sample."""
        state = self.get(session_id)
        if not state:
            return False

        # Find and remove sample
        sample_idx = None
        for i, s in enumerate(state.samples):
            if s.id == sample_id:
                sample_idx = i
                break

        if sample_idx is None:
            return False

        # Remove file
        sample_path = self._samples_dir(session_id) / f"{sample_id}.wav"
        if sample_path.exists():
            sample_path.unlink()

        state.samples.pop(sample_idx)
        self._save_state(state)
        return True

    def get_sample_path(self, session_id: str, sample_id: str) -> Path | None:
        """Get path to a sample file."""
        path = self._samples_dir(session_id) / f"{sample_id}.wav"
        return path if path.exists() else None

    def set_transcript(self, session_id: str, transcript: str) -> None:
        """Set the transcript."""
        state = self.get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        state.transcript = transcript
        state.status = SessionStatus.TRANSCRIBED
        self._save_state(state)

    def cleanup_old_sessions(self) -> int:
        """Remove sessions older than TTL. Returns count of removed sessions."""
        count = 0
        now = time.time()
        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            state_file = session_dir / "state.json"
            if state_file.exists():
                age = now - state_file.stat().st_mtime
                if age > SESSION_TTL:
                    shutil.rmtree(session_dir)
                    count += 1
        return count


# Singleton instance
_session_service: SessionService | None = None


def get_session_service() -> SessionService:
    """Get the session service singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
