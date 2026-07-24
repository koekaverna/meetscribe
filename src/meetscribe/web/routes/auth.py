"""Authentication routes: login, logout. Users are created via the admin panel or CLI."""

import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ..deps import verify_csrf
from ..services.auth import (
    COOKIE_NAME,
    get_auth_service,
    get_secure_cookies,
    get_session_ttl_days,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", dependencies=[Depends(verify_csrf)])
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    """Login with username and password."""
    auth = get_auth_service()
    try:
        user, token = auth.login(username, password)
    except ValueError:
        templates: Jinja2Templates = request.app.state.templates
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
        samesite="lax",
        secure=get_secure_cookies(),
        max_age=get_session_ttl_days() * 86400,
        path="/",
    )
    return response


@router.post("/logout", dependencies=[Depends(verify_csrf)])
def logout(request: Request) -> RedirectResponse:
    """Logout: delete session and clear cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        get_auth_service().logout(token)

    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return response
