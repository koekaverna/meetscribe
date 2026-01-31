"""FastAPI application for MeetScribe Web UI."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import samples, session, speakers, tasks, tracks

# Package directories
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


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
    app.state.templates = templates

    # Include routers
    app.include_router(session.router, prefix="/api/session", tags=["session"])
    app.include_router(tracks.router, prefix="/api/session", tags=["tracks"])
    app.include_router(samples.router, prefix="/api/session", tags=["samples"])
    app.include_router(tasks.router, prefix="/api/session", tags=["tasks"])
    app.include_router(speakers.router, prefix="/api/speakers", tags=["speakers"])

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

    return app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
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
