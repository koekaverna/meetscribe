"""Shared fixtures for web/route tests using FastAPI TestClient."""

import sqlite3
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


def _setup_config() -> tuple:
    """Set default configs and return old values for teardown."""
    old_web = auth_mod._web_cfg
    old_app = config_mod._app_config
    auth_mod._web_cfg = WebConfig()
    config_mod._app_config = AppConfig()
    return old_web, old_app


def _teardown_config(old: tuple) -> None:
    old_web, old_app = old
    auth_mod._web_cfg = old_web
    config_mod._app_config = old_app


@pytest.fixture
def web_db(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Database connection for web tests with config patching."""
    old = _setup_config()
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
    _teardown_config(old)


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
