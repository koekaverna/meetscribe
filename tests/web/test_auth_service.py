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
def _isolate_auth_globals():
    """Save and restore auth globals so tests don't leak state."""
    old_cfg = auth_mod._web_cfg
    old_svc = auth_mod._auth_service
    auth_mod._web_cfg = WebConfig()
    yield
    auth_mod._web_cfg = old_cfg
    auth_mod._auth_service = old_svc


@pytest.fixture
def auth_service(db: sqlite3.Connection) -> AuthService:
    """AuthService backed by the test database."""
    return AuthService(db)


@pytest.fixture
def registered_user(auth_service: AuthService) -> tuple[AuthUser, str]:
    """Register a test user and return (user, token)."""
    return auth_service.register("testuser", "securepass123", "default")


class TestPasswordHashing:
    def test_roundtrip_verifies_correct_password(self) -> None:
        h = hash_password("my-password")
        assert verify_password("my-password", h)

    def test_rejects_wrong_password(self) -> None:
        h = hash_password("correct")
        assert not verify_password("wrong", h)

    def test_same_password_produces_different_hashes(self) -> None:
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        assert verify_password("same", h1)
        assert verify_password("same", h2)

    @pytest.mark.parametrize("malformed_hash", [
        "garbage",
        "a$b",
        "a$b$c$d",
        "",
        "pbkdf2:sha256:100$ZZZZ$abcd",
    ])
    def test_malformed_hash_returns_false(self, malformed_hash: str) -> None:
        assert not verify_password("x", malformed_hash)

    def test_empty_password_hashes_and_verifies(self) -> None:
        h = hash_password("")
        assert verify_password("", h)
        assert not verify_password("notempty", h)


class TestRegister:
    def test_returns_user_with_correct_fields(self, auth_service: AuthService) -> None:
        user, token = auth_service.register("alice", "password123", "default")
        assert user.username == "alice"
        assert user.team_name == "default"
        assert len(token) == 64

    def test_duplicate_username_raises(self, auth_service: AuthService) -> None:
        auth_service.register("alice", "pass1234", "default")
        with pytest.raises(ValueError, match="already taken"):
            auth_service.register("alice", "pass5678", "default")

    def test_nonexistent_team_raises(self, auth_service: AuthService) -> None:
        with pytest.raises(ValueError, match="not found"):
            auth_service.register("bob", "pass1234", "no_such_team")


class TestLogin:
    def test_valid_credentials_returns_user_and_token(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        user, token = auth_service.login("testuser", "securepass123")
        assert user.username == "testuser"
        assert len(token) == 64

    def test_wrong_password_raises(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("testuser", "wrongpassword")

    def test_nonexistent_user_raises(self, auth_service: AuthService) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("ghost", "whatever")

    def test_wrong_and_nonexistent_return_same_error(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        """Prevents user enumeration via different error messages."""
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("testuser", "wrong")
        with pytest.raises(ValueError, match="Invalid"):
            auth_service.login("nonexistent", "wrong")


class TestVerifySession:
    def test_valid_token_returns_user(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        _, token = registered_user
        user = auth_service.verify_session(token)
        assert user is not None
        assert user.username == "testuser"

    def test_invalid_token_returns_none(self, auth_service: AuthService) -> None:
        assert auth_service.verify_session("nonexistent_token") is None


class TestLogout:
    def test_invalidates_session_token(
        self, auth_service: AuthService, registered_user: tuple[AuthUser, str]
    ) -> None:
        _, token = registered_user
        auth_service.logout(token)
        assert auth_service.verify_session(token) is None

    def test_unknown_token_does_not_raise(self, auth_service: AuthService) -> None:
        auth_service.logout("does_not_exist")


class TestSingleton:
    def test_get_before_init_raises_runtime_error(self) -> None:
        with patch("meetscribe.web.services.auth._auth_service", None):
            with pytest.raises(RuntimeError, match="not initialized"):
                get_auth_service()

    def test_init_then_get_returns_same_instance(self, db: sqlite3.Connection) -> None:
        svc = init_auth_service(db)
        with patch("meetscribe.web.services.auth._auth_service", svc):
            assert get_auth_service() is svc
