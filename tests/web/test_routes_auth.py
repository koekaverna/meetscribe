"""Tests for auth routes: login, register, logout."""

from fastapi.testclient import TestClient

from meetscribe.web.services.auth import get_auth_service


def _get_csrf_token(client: TestClient) -> str:
    """Get CSRF token from cookie after a GET request."""
    client.get("/login")  # triggers CSRF cookie
    return client.cookies.get("meetscribe_csrf", "")


class TestLoginRoute:
    def test_login_success(self, client: TestClient, regular_user) -> None:
        csrf_token = _get_csrf_token(client)
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "userpass1234", "csrf_token": csrf_token},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"
        assert "meetscribe_session" in resp.cookies

    def test_login_wrong_password(self, client: TestClient, regular_user) -> None:
        csrf_token = _get_csrf_token(client)
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "wrong", "csrf_token": csrf_token},
        )
        assert resp.status_code == 400

    def test_login_missing_csrf_rejected(self, client: TestClient, regular_user) -> None:
        resp = client.post(
            "/auth/login",
            data={"username": "regular", "password": "userpass1234", "csrf_token": "bad"},
        )
        assert resp.status_code == 403


class TestRegisterRoute:
    def test_register_creates_user_in_db(self, admin_client: TestClient, web_db) -> None:
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

        # Verify user actually exists in DB
        row = web_db.execute(
            "SELECT username FROM users WHERE username = ?", ("newuser",)
        ).fetchone()
        assert row is not None
        assert row["username"] == "newuser"

    def test_register_password_mismatch(self, admin_client: TestClient) -> None:
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

    def test_register_short_password(self, admin_client: TestClient) -> None:
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

    def test_register_non_admin_rejected(self, auth_client: TestClient) -> None:
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


class TestLogoutRoute:
    def test_logout_invalidates_token(self, auth_client: TestClient, regular_user) -> None:
        _, token = regular_user
        csrf_token = _get_csrf_token(auth_client)
        resp = auth_client.post(
            "/auth/logout",
            data={"csrf_token": csrf_token},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]

        # Verify the old token no longer works
        auth = get_auth_service()
        assert auth.verify_session(token) is None
