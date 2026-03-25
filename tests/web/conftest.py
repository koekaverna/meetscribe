"""Shared fixtures for web/route tests using FastAPI TestClient."""

import io
import sqlite3
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
from meetscribe.database import get_db
from meetscribe.web.services.auth import AuthService, AuthUser, init_auth_service
from meetscribe.web.services.session import SessionService


@pytest.fixture
def web_db(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Database connection for web tests with config patching and singleton cleanup."""
    # Save old singletons
    old_web_cfg = auth_mod._web_cfg
    old_app_cfg = config_mod._app_config
    old_auth_svc = auth_mod._auth_service
    old_session_svc = session_mod._session_service

    # Set default configs so tests don't need config.yaml
    auth_mod._web_cfg = WebConfig()
    config_mod._app_config = AppConfig()

    # Use get_db for migrations, then reopen with check_same_thread=False
    # because TestClient runs requests in a different thread
    db_path = tmp_path / "web_test.db"
    init_conn = get_db(db_path)
    init_conn.close()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    yield conn

    conn.close()

    # Restore all singletons
    auth_mod._web_cfg = old_web_cfg
    config_mod._app_config = old_app_cfg
    auth_mod._auth_service = old_auth_svc
    session_mod._session_service = old_session_svc


@pytest.fixture
def web_auth_service(web_db: sqlite3.Connection) -> AuthService:
    return init_auth_service(web_db)


@pytest.fixture
def web_session_service(web_db: sqlite3.Connection, tmp_path: Path) -> SessionService:
    svc = SessionService(web_db, sessions_dir=tmp_path / "sessions")
    session_mod._session_service = svc
    return svc


@pytest.fixture
def admin_user(web_auth_service: AuthService) -> tuple[AuthUser, str]:
    """Create an admin user. Returns (AuthUser, session_token)."""
    user, token = web_auth_service.register("admin", "adminpass123", "default")
    web_auth_service.conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user.id,))
    web_auth_service.conn.commit()
    user.is_admin = True
    return user, token


@pytest.fixture
def regular_user(web_auth_service: AuthService) -> tuple[AuthUser, str]:
    """Create a regular (non-admin) user. Returns (AuthUser, session_token)."""
    return web_auth_service.register("regular", "userpass1234", "default")


@pytest.fixture
def app(web_auth_service: AuthService, web_session_service: SessionService):
    """Create a FastAPI app with test services already initialized."""
    from meetscribe.web.app import create_app

    with patch("meetscribe.web.app.get_db", return_value=web_auth_service.conn):
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
