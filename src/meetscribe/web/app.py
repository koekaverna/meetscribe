"""FastAPI application for MeetScribe Web UI."""

import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .deps import get_current_user, get_current_user_or_none
from .routes import auth, samples, session, speakers, tasks, tracks
from .routes.tasks import shutdown_threads
from .services.auth import get_secure_cookies

CSRF_COOKIE_NAME = "meetscribe_csrf"
CSRF_FORM_FIELD = "csrf_token"

# Package directories
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Paths that don't require authentication
PUBLIC_PREFIXES = ("/auth", "/static", "/login", "/health")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MeetScribe",
        description="Meeting transcription with speaker diarization",
        version="0.1.0",
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Setup templates
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    def csrf_token_for_request(request: Request) -> str:
        """Get or generate CSRF token for the current request.

        Stores generated token on request.state so the middleware
        can set the same value as a cookie.
        """
        token = request.cookies.get(CSRF_COOKIE_NAME)
        if not token:
            # Reuse token if already generated during this request
            token = getattr(request.state, "csrf_token", None)
            if not token:
                token = secrets.token_hex(32)
                request.state.csrf_token = token
        return token

    # Make csrf_token() available in all templates
    templates.env.globals["csrf_token"] = csrf_token_for_request
    templates.env.globals["csrf_field_name"] = CSRF_FORM_FIELD
    app.state.templates = templates

    @app.middleware("http")
    async def csrf_cookie_middleware(request: Request, call_next):
        """Ensure CSRF cookie is set on every response."""
        response: Response = await call_next(request)
        if not request.cookies.get(CSRF_COOKIE_NAME):
            # Use the same token that was rendered in the template
            token = getattr(request.state, "csrf_token", None) or secrets.token_hex(32)
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=token,
                httponly=False,  # Must be readable by templates
                samesite="strict",
                secure=get_secure_cookies(),
                path="/",
            )
        return response

    # Auth middleware: redirect unauthenticated page requests to /login
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path

        # Skip public paths
        if any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES):
            return await call_next(request)

        # API routes: let the dependency handle auth (returns 401)
        if path.startswith("/api/"):
            return await call_next(request)

        # Page routes: redirect to login if no valid session
        user = await get_current_user_or_none(request)
        if not user:
            return RedirectResponse("/login", status_code=303)

        return await call_next(request)

    # Auth routes (no auth required)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])

    # API routes (auth required via dependency)
    app.include_router(
        session.router,
        prefix="/api/session",
        tags=["session"],
        dependencies=[Depends(get_current_user)],
    )
    app.include_router(
        tracks.router,
        prefix="/api/session",
        tags=["tracks"],
        dependencies=[Depends(get_current_user)],
    )
    app.include_router(
        samples.router,
        prefix="/api/session",
        tags=["samples"],
        dependencies=[Depends(get_current_user)],
    )
    app.include_router(
        tasks.router,
        prefix="/api/session",
        tags=["tasks"],
        dependencies=[Depends(get_current_user)],
    )
    app.include_router(
        speakers.router,
        prefix="/api/speakers",
        tags=["speakers"],
        dependencies=[Depends(get_current_user)],
    )

    # Page routes

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        """Render login page."""
        # Redirect if already authenticated
        user = await get_current_user_or_none(request)
        if user:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "login.html")

    @app.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):
        """Render registration page. Only admins can access."""
        user = await get_current_user_or_none(request)
        if not user:
            return RedirectResponse("/login", status_code=303)
        if not user.is_admin:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "register.html", {"team_name": user.team_name})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Render the main page."""
        return templates.TemplateResponse(request, "index.html")

    @app.get("/step/{step_num}", response_class=HTMLResponse)
    async def get_step(request: Request, step_num: int):
        """Render a specific step."""
        step_templates = {
            1: "steps/step1_upload.html",
            2: "steps/step2_config.html",
            3: "steps/step3_extract.html",
            4: "steps/step4_samples.html",
            5: "steps/step5_enroll.html",
            6: "steps/step6_transcribe.html",
        }
        template_name = step_templates.get(step_num, "steps/step1_upload.html")
        return templates.TemplateResponse(request, template_name)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def on_shutdown():
        shutdown_threads()

    return app


def run(host: str = "127.0.0.1", port: int = 8080) -> None:  # defaults for direct invocation
    """Run the web server."""
    import uvicorn

    from meetscribe import config

    # Ensure directories exist
    config.ensure_dirs()

    # Create static directory if it doesn't exist
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    app = create_app()
    print("\n  MeetScribe Web UI")
    print(f"  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
