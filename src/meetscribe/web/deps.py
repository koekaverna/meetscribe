"""FastAPI dependencies for authentication and session access control."""

from fastapi import Depends, Form, HTTPException, Request

from .models import SessionState
from .services.auth import COOKIE_NAME, AuthUser, get_auth_service
from .services.session import get_session_service

CSRF_COOKIE_NAME = "meetscribe_csrf"


def verify_csrf(request: Request, csrf_token: str = Form(...)) -> None:
    """Verify CSRF token from form matches cookie. Use as dependency on POST routes."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not cookie_token or not csrf_token or cookie_token != csrf_token:
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def get_current_user(request: Request) -> AuthUser:
    """Extract authenticated user from session cookie. Raises 401 if invalid."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_auth_service().verify_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    return user


def get_admin_user(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Require an admin user. Raises 403 for non-admins."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def get_superadmin_user(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Require a superadmin user. Raises 403 otherwise."""
    if not user.is_superadmin:
        raise HTTPException(status_code=403, detail="Superadmin access required")
    return user


def get_current_user_or_none(request: Request) -> AuthUser | None:
    """Extract authenticated user or return None (for page routes that redirect)."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return get_auth_service().verify_session(token)


def get_session_for_user(session_id: str, user: AuthUser) -> SessionState:
    """Get a session and verify access: own sessions only; admins get any team session.

    404 (not 403) in all deny cases so existence isn't leaked.
    """
    service = get_session_service()
    state = service.get(session_id)
    if not state or state.team_name != user.team_name:
        raise HTTPException(status_code=404, detail="Session not found")
    if not user.is_admin and state.creator_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return state
