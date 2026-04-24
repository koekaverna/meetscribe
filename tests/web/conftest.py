"""Shared fixtures for web/route tests using FastAPI TestClient."""

import io
import wave
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import meetscribe.config as config_mod
import meetscribe.web.services.auth as auth_mod
import meetscribe.web.services.session as session_mod
from meetscribe.config import AppConfig, WebConfig
from meetscribe.database import close_db, get_db, init_db
from meetscribe.web.services.auth import AuthService, AuthUser
from meetscribe.web.services.session import SessionService


@pytest.fixture
def web_db(tmp_path: Path) -> Generator[None, None, None]:
    """Initialize DB for web tests with config patching and singleton cleanup."""
    old_web_cfg = auth_mod._web_cfg
    old_app_cfg = config_mod._app_config
    old_session_svc = session_mod._session_service

    auth_mod._web_cfg = WebConfig()
    config_mod._app_config = AppConfig()

    db_path = tmp_path / "web_test.db"
    init_db(db_path)

    yield

    close_db()

    auth_mod._web_cfg = old_web_cfg
    config_mod._app_config = old_app_cfg
    session_mod._session_service = old_session_svc


@pytest.fixture
def web_auth_service(web_db) -> AuthService:
    return AuthService()


@pytest.fixture
def web_session_service(web_db, tmp_path: Path) -> SessionService:
    svc = SessionService(sessions_dir=tmp_path / "sessions")
    session_mod._session_service = svc
    return svc


@pytest.fixture
def admin_user(web_auth_service: AuthService) -> tuple[AuthUser, str]:
    """Create an admin user. Returns (AuthUser, session_token)."""
    user, token = web_auth_service.register("admin", "test-pass-000", "default")
    conn = get_db()
    conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user.id,))
    conn.commit()
    user.is_admin = True
    return user, token


@pytest.fixture
def regular_user(web_auth_service: AuthService) -> tuple[AuthUser, str]:
    """Create a regular (non-admin) user. Returns (AuthUser, session_token)."""
    return web_auth_service.register("regular", "test-pass-000", "default")


@pytest.fixture
def app(web_auth_service: AuthService, web_session_service: SessionService):
    """Create a FastAPI app with test services already initialized."""
    from meetscribe.web.app import create_app

    with (
        patch("meetscribe.web.app.init_db"),
        patch("meetscribe.web.app.init_session_service", return_value=web_session_service),
    ):
        return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Unauthenticated test client."""
    return TestClient(app)


@pytest.fixture
def auth_client(client: TestClient, regular_user: tuple[AuthUser, str]) -> TestClient:
    """Test client with a valid session cookie."""
    _, token = regular_user
    client.cookies.set("meetscribe_session", token)
    return client


@pytest.fixture
def admin_client(client: TestClient, admin_user: tuple[AuthUser, str]) -> TestClient:
    """Test client with an admin session cookie."""
    _, token = admin_user
    client.cookies.set("meetscribe_session", token)
    return client


@pytest.fixture
def session_id(auth_client: TestClient) -> str:
    """Create a session and return its ID."""
    return auth_client.post("/api/session").json()["session_id"]


@pytest.fixture
def wav_upload_bytes() -> bytes:
    """1-second 16kHz mono WAV (silence) for upload tests."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()
