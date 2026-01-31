"""Session management routes."""

from fastapi import APIRouter, HTTPException

from ..models import CreateSessionResponse, SessionState
from ..services.session import get_session_service

router = APIRouter()


@router.post("", response_model=CreateSessionResponse)
async def create_session():
    """Create a new session."""
    service = get_session_service()
    state = service.create()
    return CreateSessionResponse(session_id=state.id)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """Get session state."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    service = get_session_service()
    if not service.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
