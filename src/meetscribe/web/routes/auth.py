"""Authentication routes: login, register, logout."""

import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse

from ..deps import get_current_user, verify_csrf
from ..services.auth import (
    COOKIE_NAME,
    AuthUser,
    get_auth_service,
    get_secure_cookies,
    get_session_ttl_days,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", dependencies=[Depends(verify_csrf)])
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Login with username and password."""
    auth = get_auth_service()
    try:
        user, token = auth.login(username, password)
    except ValueError:
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid username or password"},
            status_code=400,
        )

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="strict",
        secure=get_secure_cookies(),
        max_age=get_session_ttl_days() * 86400,
        path="/",
    )
    return response


@router.post("/register", dependencies=[Depends(verify_csrf)])
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    admin: AuthUser = Depends(get_current_user),
):
    """Register a new user in admin's team."""
    templates = request.app.state.templates

    def _render_error(error: str):
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": error, "team_name": admin.team_name},
            status_code=400,
        )

    if not admin.is_admin:
        return _render_error("Only admins can register new users")

    if password != password_confirm:
        return _render_error("Passwords do not match")

    if len(password) < 8:
        return _render_error("Password must be at least 8 characters")

    auth = get_auth_service()
    try:
        auth.register(username, password, admin.team_name)
    except ValueError as e:
        logger.warning("Registration failed for user '%s': %s", username, e)
        return _render_error("Registration failed. Please try a different username.")

    return templates.TemplateResponse(
        request,
        "register.html",
        {"team_name": admin.team_name, "success": username},
    )


@router.post("/logout", dependencies=[Depends(verify_csrf)])
async def logout(request: Request):
    """Logout: delete session and clear cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        get_auth_service().logout(token)

    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return response
