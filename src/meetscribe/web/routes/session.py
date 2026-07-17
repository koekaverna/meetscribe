"""Session management routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_current_user, get_session_for_user
from ..models import BulkDeleteRequest, CreateSessionResponse, SessionListResponse, SessionState
from ..services.auth import AuthUser
from ..services.session import get_session_service

router = APIRouter()


@router.post("", response_model=CreateSessionResponse)
def create_session(user: AuthUser = Depends(get_current_user)) -> CreateSessionResponse:
    """Create a new session scoped to the user's team."""
    service = get_session_service()
    state = service.create(team_name=user.team_name, creator_id=user.id)
    return CreateSessionResponse(session_id=state.id)


@router.get("", response_model=SessionListResponse)
def list_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: Literal["date", "duration"] = "date",
    order: Literal["asc", "desc"] = "desc",
    mine: bool = False,
    user: AuthUser = Depends(get_current_user),
) -> SessionListResponse:
    """List sessions (paginated). Non-admins see only their own; admins the whole team."""
    sessions, total = get_session_service().list_summaries(
        team_id=user.team_id,
        page=page,
        per_page=per_page,
        sort=sort,
        order=order,
        creator_id=user.id if mine or not user.is_admin else None,
    )
    return SessionListResponse(sessions=sessions, total=total, page=page, per_page=per_page)


@router.get("/{session_id}", response_model=SessionState)
def get_session(session_id: str, user: AuthUser = Depends(get_current_user)) -> SessionState:
    """Get session state (team-scoped)."""
    return get_session_for_user(session_id, user)


@router.delete("/{session_id}")
def delete_session(session_id: str, user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    """Delete a session (team-scoped)."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    if not service.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/bulk-delete")
def bulk_delete_sessions(
    body: BulkDeleteRequest, user: AuthUser = Depends(get_current_user)
) -> dict[str, int]:
    """Delete multiple sessions with their files; ids the user cannot access are skipped."""
    service = get_session_service()
    deleted = 0
    for session_id in body.ids:
        try:
            get_session_for_user(session_id, user)
        except HTTPException:
            continue
        if service.delete(session_id):
            deleted += 1
    return {"deleted": deleted}
