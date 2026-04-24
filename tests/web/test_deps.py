"""Tests for FastAPI dependencies: CSRF validation and auth extraction."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import meetscribe.web.services.auth as auth_mod
from meetscribe.config import WebConfig
from meetscribe.web.deps import (
    get_current_user,
    get_current_user_or_none,
    get_session_for_user,
    verify_csrf,
)
from meetscribe.web.models import SessionState
from meetscribe.web.services.auth import AuthService, AuthUser, init_auth_service


def _make_request(cookies: dict[str, str] | None = None) -> MagicMock:
    """Create a fake Request with given cookies."""
    request = MagicMock()
    request.cookies = cookies or {}
    return request


@pytest.fixture(autouse=True)
def _patch_web_config():
    old = auth_mod._web_cfg
    auth_mod._web_cfg = WebConfig()
    yield
    auth_mod._web_cfg = old


@pytest.fixture
def auth_service(db: sqlite3.Connection) -> AuthService:
    svc = init_auth_service(db)
    return svc


@pytest.fixture
def user_and_token(auth_service: AuthService) -> tuple[AuthUser, str]:
    return auth_service.register("testuser", "test-pass-000", "default")


# --- verify_csrf ---


class TestVerifyCSRF:
    async def test_valid_csrf(self) -> None:
        request = _make_request({"meetscribe_csrf": "tok123"})
        await verify_csrf(request, csrf_token="tok123")  # should not raise

    async def test_mismatched_csrf_raises_403(self) -> None:
        from fastapi import HTTPException

        request = _make_request({"meetscribe_csrf": "good"})
        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf(request, csrf_token="bad")
        assert exc_info.value.status_code == 403

    async def test_missing_cookie_raises_403(self) -> None:
        from fastapi import HTTPException

        request = _make_request({})
        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf(request, csrf_token="anything")
        assert exc_info.value.status_code == 403

    async def test_empty_token_raises_403(self) -> None:
        from fastapi import HTTPException

        request = _make_request({"meetscribe_csrf": "tok"})
        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf(request, csrf_token="")
        assert exc_info.value.status_code == 403


# --- get_current_user ---


class TestGetCurrentUser:
    def test_no_cookie_raises_401(self) -> None:
        from fastapi import HTTPException

        request = _make_request({})
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(request)
        assert exc_info.value.status_code == 401

    def test_invalid_token_raises_401(self, auth_service: AuthService) -> None:
        from fastapi import HTTPException

        request = _make_request({"meetscribe_session": "bad_token"})
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(request)
        assert exc_info.value.status_code == 401

    def test_valid_token_returns_user(
        self, auth_service: AuthService, user_and_token: tuple[AuthUser, str]
    ) -> None:
        _, token = user_and_token
        request = _make_request({"meetscribe_session": token})
        user = get_current_user(request)
        assert user.username == "testuser"


# --- get_current_user_or_none ---


class TestGetCurrentUserOrNone:
    def test_no_cookie_returns_none(self, auth_service: AuthService) -> None:
        request = _make_request({})
        assert get_current_user_or_none(request) is None

    def test_invalid_token_returns_none(self, auth_service: AuthService) -> None:
        request = _make_request({"meetscribe_session": "expired"})
        assert get_current_user_or_none(request) is None

    def test_valid_token_returns_user(
        self, auth_service: AuthService, user_and_token: tuple[AuthUser, str]
    ) -> None:
        _, token = user_and_token
        request = _make_request({"meetscribe_session": token})
        user = get_current_user_or_none(request)
        assert user is not None
        assert user.username == "testuser"


# --- get_session_for_user ---


class TestGetSessionForUser:
    def test_session_not_found_raises_404(self) -> None:
        from fastapi import HTTPException

        user = AuthUser(id=1, username="u", team_id=1, team_name="default")
        with patch("meetscribe.web.deps.get_session_service") as mock_svc:
            mock_svc.return_value.get.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                get_session_for_user("no-such-id", user)
            assert exc_info.value.status_code == 404

    def test_wrong_team_raises_404(self) -> None:
        from fastapi import HTTPException

        user = AuthUser(id=1, username="u", team_id=1, team_name="team_a")
        state = SessionState(id="s1", team_name="team_b")
        with patch("meetscribe.web.deps.get_session_service") as mock_svc:
            mock_svc.return_value.get.return_value = state
            with pytest.raises(HTTPException) as exc_info:
                get_session_for_user("s1", user)
            assert exc_info.value.status_code == 404

    def test_correct_team_returns_state(self) -> None:
        user = AuthUser(id=1, username="u", team_id=1, team_name="default")
        state = SessionState(id="s1", team_name="default")
        with patch("meetscribe.web.deps.get_session_service") as mock_svc:
            mock_svc.return_value.get.return_value = state
            result = get_session_for_user("s1", user)
            assert result.id == "s1"
