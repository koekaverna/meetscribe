"""Pydantic models for web API."""

from enum import Enum

from pydantic import BaseModel


class SessionStatus(str, Enum):
    """Session status."""

    CREATED = "created"
    UPLOADED = "uploaded"
    CONFIGURED = "configured"
    EXTRACTED = "extracted"
    ENROLLED = "enrolled"
    TRANSCRIBED = "transcribed"


class TrackConfig(BaseModel):
    """Track configuration."""

    track_num: int
    filename: str
    speaker_name: str | None = None
    diarize: bool = True


class SpeakerBin(BaseModel):
    """Speaker bin for sample organization."""

    id: str
    name: str
    sample_ids: list[str] = []


class Sample(BaseModel):
    """Audio sample."""

    id: str
    track_num: int
    cluster_id: int
    filename: str
    duration_ms: int
    speaker_id: str | None = None
    is_known: bool = False
    known_speaker_name: str | None = None


class SessionState(BaseModel):
    """Complete session state."""

    id: str
    status: SessionStatus = SessionStatus.CREATED
    tracks: list[TrackConfig] = []
    speakers: list[SpeakerBin] = []
    samples: list[Sample] = []
    transcript: str | None = None
    whisper_model: str = "medium"
    language: str = "ru"


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str


class TrackUploadResponse(BaseModel):
    """Response for track upload."""

    track_num: int
    filename: str


class SpeakerCreate(BaseModel):
    """Request to create a speaker bin."""

    name: str


class SpeakerRename(BaseModel):
    """Request to rename a speaker bin."""

    name: str


class SampleMove(BaseModel):
    """Request to move a sample to a speaker bin."""

    speaker_id: str | None


class TranscribeOptions(BaseModel):
    """Transcription options."""

    model: str = "medium"
    language: str = "ru"


class GlobalSpeaker(BaseModel):
    """Global enrolled speaker."""

    name: str


class ProgressEvent(BaseModel):
    """SSE progress event."""

    step: int
    total_steps: int
    message: str
    progress: float | None = None
