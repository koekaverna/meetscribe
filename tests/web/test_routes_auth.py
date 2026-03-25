"""Tests for auth routes: login, register, logout."""

from fastapi.testclient import TestClient

from meetscribe.web.services.auth import get_auth_service


def _get_csrf_token(client: TestClient) -> str:
    """Get CSRF token from cookie after a GET request."""
    client.get("/login")
    token = client.cookies.get("meetscribe_csrf")
    assert token, "CSRF cookie not set after GET /login"
    return token


class TestLogin:
    def test_valid_credentials_redirects_to_home(self, client: TestClient, regular_user) -> None:
        csrf_token = _get_csrf_token(client)
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "userpass1234", "csrf_token": csrf_token},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"
        assert "meetscribe_session" in resp.cookies

    def test_wrong_password_returns_400_without_session(
        self, client: TestClient, regular_user
    ) -> None:
        csrf_token = _get_csrf_token(client)
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "wrong", "csrf_token": csrf_token},
        )
        assert resp.status_code == 400
        assert "meetscribe_session" not in resp.cookies

    def test_invalid_csrf_returns_403_without_session(
        self, client: TestClient, regular_user
    ) -> None:
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "userpass1234", "csrf_token": "bad"},
        )
        assert resp.status_code == 403
        assert "meetscribe_session" not in resp.cookies


class TestRegister:
    def test_admin_creates_user_persisted_in_db(self, admin_client: TestClient, web_db) -> None:
        csrf_token = _get_csrf_token(admin_client)
        resp = admin_client.post(
            "/auth/register",
            data={
                "username": "newuser",
                "password": "newpass1234",
                "password_confirm": "newpass1234",
                "csrf_token": csrf_token,
            },
        )
        assert resp.status_code == 200

        row = web_db.execute(
            "SELECT username FROM users WHERE username = ?", ("newuser",)
        ).fetchone()
        assert row is not None

    def test_password_mismatch_returns_400_and_no_user_created(
        self, admin_client: TestClient, web_db
    ) -> None:
        csrf_token = _get_csrf_token(admin_client)
        resp = admin_client.post(
            "/auth/register",
            data={
                "username": "newuser",
                "password": "pass1234",
                "password_confirm": "different",
                "csrf_token": csrf_token,
            },
        )
        assert resp.status_code == 400

        row = web_db.execute(
            "SELECT username FROM users WHERE username = ?", ("newuser",)
        ).fetchone()
        assert row is None

    def test_short_password_returns_400(self, admin_client: TestClient) -> None:
        csrf_token = _get_csrf_token(admin_client)
        resp = admin_client.post(
            "/auth/register",
            data={
                "username": "newuser",
                "password": "short",
                "password_confirm": "short",
                "csrf_token": csrf_token,
            },
        )
        assert resp.status_code == 400

    def test_non_admin_cannot_register_users(self, auth_client: TestClient, web_db) -> None:
        csrf_token = _get_csrf_token(auth_client)
        resp = auth_client.post(
            "/auth/register",
            data={
                "username": "newuser",
                "password": "pass12345",
                "password_confirm": "pass12345",
                "csrf_token": csrf_token,
            },
        )
        assert resp.status_code == 400

        row = web_db.execute(
            "SELECT username FROM users WHERE username = ?", ("newuser",)
        ).fetchone()
        assert row is None


class TestLogout:
    def test_invalidates_token_and_deletes_cookie(
        self, auth_client: TestClient, regular_user
    ) -> None:
        _, token = regular_user
        csrf_token = _get_csrf_token(auth_client)
        resp = auth_client.post(
            "/auth/logout",
            data={"csrf_token": csrf_token},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]

        # Verify token invalidated server-side
        assert get_auth_service().verify_session(token) is None

        # Verify cookie deleted in response
        session_cookie = resp.cookies.get("meetscribe_session")
        assert session_cookie is None or session_cookie == ""
