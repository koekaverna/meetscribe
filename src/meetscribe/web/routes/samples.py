"""Sample management routes."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from ..models import Sample, SampleMove, SpeakerBin, SpeakerCreate, SpeakerRename
from ..services.session import get_session_service

router = APIRouter()


@router.post("/{session_id}/speakers", response_model=SpeakerBin)
async def create_speaker(session_id: str, data: SpeakerCreate):
    """Create a speaker bin."""
    service = get_session_service()
    try:
        speaker = service.add_speaker(session_id, data.name)
        return speaker
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{session_id}/speakers/{speaker_id}")
async def rename_speaker(session_id: str, speaker_id: str, data: SpeakerRename):
    """Rename a speaker bin."""
    service = get_session_service()
    if not service.rename_speaker(session_id, speaker_id, data.name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "updated"}


@router.delete("/{session_id}/speakers/{speaker_id}")
async def delete_speaker(session_id: str, speaker_id: str):
    """Delete a speaker bin."""
    service = get_session_service()
    if not service.delete_speaker(session_id, speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "deleted"}


@router.get("/{session_id}/samples", response_model=list[Sample])
async def list_samples(session_id: str):
    """List samples in session."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.samples


@router.get("/{session_id}/samples/{sample_id}/audio")
async def get_sample_audio(session_id: str, sample_id: str):
    """Stream sample audio."""
    service = get_session_service()
    path = service.get_sample_path(session_id, sample_id)
    if not path:
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="audio/wav")


@router.post("/{session_id}/samples/{sample_id}/move")
async def move_sample(session_id: str, sample_id: str, data: SampleMove):
    """Move sample to a speaker bin."""
    service = get_session_service()

    # Find speaker name by speaker_id
    speaker_name = None
    if data.speaker_id:
        state = service.get(session_id)
        if state:
            for speaker in state.speakers:
                if speaker.id == data.speaker_id:
                    speaker_name = speaker.name
                    break

    if not service.move_sample(session_id, sample_id, data.speaker_id, speaker_name):
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"status": "moved"}


@router.delete("/{session_id}/samples/{sample_id}")
async def delete_sample(session_id: str, sample_id: str):
    """Delete a sample."""
    service = get_session_service()
    if not service.delete_sample(session_id, sample_id):
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"status": "deleted"}


@router.get("/{session_id}/samples-view", response_class=HTMLResponse)
async def get_samples_view(request: Request, session_id: str):
    """Get HTML view of samples organized by speaker."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "partials/samples_grid.html",
        {"session": state},
    )
