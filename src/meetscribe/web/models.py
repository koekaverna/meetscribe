"""Pydantic models for web API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    """Session status."""

    CREATED = "created"
    UPLOADED = "uploaded"
    CONFIGURED = "configured"
    EXTRACTED = "extracted"
    ENROLLED = "enrolled"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"


class TrackConfig(BaseModel):
    """Track configuration."""

    track_num: int
    filename: str
    speaker_name: str | None = None
    diarize: bool = True
    # Open-space mic: keep only the assigned speaker's voice (named tracks only).
    open_space: bool = False


class SpeakerBin(BaseModel):
    """Speaker bin for sample organization."""

    id: str
    name: str


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


class TranscriptSegmentModel(BaseModel):
    """Structured transcript segment for playback."""

    track_num: int
    start_ms: int
    end_ms: int
    speaker: str | None = None
    text: str


class SessionState(BaseModel):
    """Complete session state."""

    id: str
    status: SessionStatus = SessionStatus.CREATED
    team_name: str = "default"
    creator_id: int | None = None
    tracks: list[TrackConfig] = []
    speakers: list[SpeakerBin] = []
    samples: list[Sample] = []
    transcript: str | None = None
    segments: list[TranscriptSegmentModel] = []
    language: str = "ru"


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str


class SessionSummary(BaseModel):
    """One row in the session archive list."""

    id: str
    status: SessionStatus
    created_at: str
    creator: str | None = None
    track_count: int = 0
    # MAX(end_ms) over segments; None until transcribed
    duration_ms: int | None = None
    speakers: list[str] = []
    preview: str | None = None


class SessionListResponse(BaseModel):
    """Paginated session list."""

    sessions: list[SessionSummary]
    total: int
    page: int
    per_page: int


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple sessions."""

    # Selection is page-bound; per_page is capped at 100
    ids: list[str] = Field(..., max_length=100)


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

    language: str = "ru"


class GlobalSpeaker(BaseModel):
    """Enrolled speaker with voiceprint quality stats."""

    name: str
    model: str
    sample_count: int
    total_duration_ms: int
    created_at: str


class SpeakerSample(BaseModel):
    """Enrolled sample of a speaker (a wav file in the team's enrolled dir)."""

    filename: str
    duration_ms: int


class ProgressEvent(BaseModel):
    """SSE progress event."""

    step: int
    total_steps: int
    message: str
    progress: float | None = None


class AdminUser(BaseModel):
    """User row in the admin panel."""

    id: int
    username: str
    team_name: str
    is_admin: bool
    is_superadmin: bool = False
    created_at: str


class AdminUserCreate(BaseModel):
    """Request to create a user from the admin panel. None team = requester's own."""

    # No "/": users are addressed via path parameters in the manage endpoints
    username: str = Field(..., min_length=1, max_length=64, pattern=r"^[^/]+$")
    password: str
    team_name: str | None = None
    is_admin: bool = False


class AdminUserPatch(BaseModel):
    """Request to change a user's admin role."""

    is_admin: bool


class AdminPasswordReset(BaseModel):
    """Request to set a user's new password."""

    password: str


class AdminTeam(BaseModel):
    """Team row with usage counts."""

    id: int
    name: str
    description: str | None = None
    created_at: str
    user_count: int
    session_count: int
    voiceprint_count: int


class AdminTeamCreate(BaseModel):
    """Request to create a team from the admin panel."""

    name: str
    description: str | None = None


class ServerStatus(BaseModel):
    """Reachability of one configured Speaches server."""

    name: str
    url: str
    reachable: bool
    latency_ms: int | None = None
    error: str | None = None


class DiskUsage(BaseModel):
    """Disk usage of the data directory (bytes)."""

    total_bytes: int
    sessions_bytes: int
    samples_bytes: int


class ErrorLogTail(BaseModel):
    """Recent ERROR lines from the newest log file."""

    file: str | None = None
    lines: list[str] = []
