"""Session management routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_current_user, get_session_for_user
from ..models import CreateSessionResponse, SessionState
from ..services.auth import AuthUser
from ..services.session import get_session_service

router = APIRouter()


@router.post("", response_model=CreateSessionResponse)
async def create_session(user: AuthUser = Depends(get_current_user)) -> CreateSessionResponse:
    """Create a new session scoped to the user's team."""
    service = get_session_service()
    state = service.create(team_name=user.team_name)
    return CreateSessionResponse(session_id=state.id)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str, user: AuthUser = Depends(get_current_user)) -> SessionState:
    """Get session state (team-scoped)."""
    return get_session_for_user(session_id, user)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Delete a session (team-scoped)."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    if not service.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
