"""FastAPI dependencies for authentication and session access control."""

from fastapi import Form, HTTPException, Request

from .models import SessionState
from .services.auth import COOKIE_NAME, AuthUser, get_auth_service
from .services.session import get_session_service

CSRF_COOKIE_NAME = "meetscribe_csrf"


async def verify_csrf(request: Request, csrf_token: str = Form(...)) -> None:
    """Verify CSRF token from form matches cookie. Use as dependency on POST routes."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not cookie_token or not csrf_token or cookie_token != csrf_token:
        raise HTTPException(status_code=403, detail="CSRF validation failed")


async def get_current_user(request: Request) -> AuthUser:
    """Extract authenticated user from session cookie. Raises 401 if invalid."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_auth_service().verify_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    return user


async def get_current_user_or_none(request: Request) -> AuthUser | None:
    """Extract authenticated user or return None (for page routes that redirect)."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return get_auth_service().verify_session(token)


def get_session_for_user(session_id: str, user: AuthUser) -> SessionState:
    """Get a session and verify it belongs to the user's team."""
    service = get_session_service()
    state = service.get(session_id)
    if not state or state.team_name != user.team_name:
        raise HTTPException(status_code=404, detail="Session not found")
    return state
