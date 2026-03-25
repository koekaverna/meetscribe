"""Tests for web auth service: password hashing, login, registration, sessions."""

import sqlite3
from unittest.mock import patch

import pytest

import meetscribe.web.services.auth as auth_mod
from meetscribe.config import WebConfig
from meetscribe.web.services.auth import (
    AuthService,
    AuthUser,
    get_auth_service,
    hash_password,
    init_auth_service,
    verify_password,
)


@pytest.fixture(autouse=True)
def _patch_web_config():
    """Provide a default WebConfig so auth doesn't need config.yaml."""
    old = auth_mod._web_cfg
    auth_mod._web_cfg = WebConfig()
    yield
    auth_mod._web_cfg = old


@pytest.fixture
def auth_service(db: sqlite3.Connection) -> AuthService:
    """AuthService backed by the test database."""
    return AuthService(db)


@pytest.fixture
def registered_user(auth_service: AuthService) -> tuple[AuthUser, str]:
    """Register a test user and return (user, token)."""
    return auth_service.register("testuser", "securepass123", "default")


# --- Password hashing ---


class TestPasswordHashing:
    def test_hash_verify_roundtrip(self) -> None:
        h = hash_password("my-password")
        assert verify_password("my-password", h)

    def test_wrong_password_rejected(self) -> None:
        h = hash_password("correct")
        assert not verify_password("wrong", h)

    def test_hash_format(self) -> None:
        h = hash_password("x")
        parts = h.split("$")
        assert len(parts) == 3
        assert parts[0].startswith("pbkdf2:sha256:")

    def test_different_salts(self) -> None:
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # different salts
        assert verify_password("same", h1)
        assert verify_password("same", h2)

    def test_malformed_hash_returns_false(self) -> None:
        assert not verify_password("x", "garbage")
        assert not verify_password("x", "a$b")
        assert not verify_password("x", "a$b$c$d")
        assert not verify_password("x", "")

    def test_non_hex_salt_returns_false(self) -> None:
        assert not verify_password("x", "pbkdf2:sha256:100$ZZZZ$abcd")

    def test_empty_password(self) -> None:
        h = hash_password("")
        assert verify_password("", h)
        assert not verify_password("notempty", h)


# --- AuthService.register ---


class TestRegister:
    def test_register_success(self, auth_service: AuthService) -> None:
        user, token = auth_service.register("alice", "password123", "default")
        assert user.username == "alice"
        assert user.team_name == "default"
        assert len(token) == 64  # hex of 32 bytes

    def test_register_duplicate_username(self, auth_service: AuthService) -> None:
        auth_service.register("alice", "pass1234", "default")
        with pytest.raises(ValueError, match="already taken"):
            auth_service.register("alice", "pass5678", "default")

    def test_register_nonexistent_team(self, auth_service: AuthService) -> None:
        with pytest.raises(ValueError, match="not found"):
            auth_service.register("bob", "pass1234", "no_such_team")


# --- AuthService.login ---


class TestLogin:
    def test_login_success(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        user, token = auth_service.login("testuser", "securepass123")
        assert user.username == "testuser"
        assert len(token) == 64

    def test_login_wrong_password(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("testuser", "wrongpassword")

    def test_login_nonexistent_user(self, auth_service: AuthService) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("ghost", "whatever")


# --- AuthService.verify_session ---


class TestVerifySession:
    def test_verify_valid_token(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        _, token = registered_user
        user = auth_service.verify_session(token)
        assert user is not None
        assert user.username == "testuser"

    def test_verify_invalid_token(self, auth_service: AuthService) -> None:
        assert auth_service.verify_session("nonexistent_token") is None


# --- AuthService.logout ---


class TestLogout:
    def test_logout_invalidates_session(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        _, token = registered_user
        auth_service.logout(token)
        assert auth_service.verify_session(token) is None

    def test_logout_unknown_token_no_error(self, auth_service: AuthService) -> None:
        auth_service.logout("does_not_exist")  # should not raise


# --- Singleton management ---


class TestSingleton:
    def test_get_auth_service_before_init_raises(self) -> None:
        with patch("meetscribe.web.services.auth._auth_service", None):
            with pytest.raises(RuntimeError, match="not initialized"):
                get_auth_service()

    def test_init_and_get(self, db: sqlite3.Connection) -> None:
        svc = init_auth_service(db)
        with patch("meetscribe.web.services.auth._auth_service", svc):
            assert get_auth_service() is svc
