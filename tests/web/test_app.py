"""Tests for FastAPI app: middleware, CSRF, auth, health, and page routes."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuthMiddleware:
    def test_unauthenticated_page_request_redirects_to_login(self, client: TestClient) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]

    def test_public_path_accessible_without_auth(self, client: TestClient) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_api_route_returns_401_instead_of_redirect(self, client: TestClient) -> None:
        resp = client.post("/api/session")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]

    def test_authenticated_user_can_access_protected_page(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 200


class TestCSRFMiddleware:
    def test_sets_csrf_cookie_on_first_request(self, client: TestClient) -> None:
        client.get("/login")
        assert "meetscribe_csrf" in client.cookies


class TestPageRoutes:
    def test_login_page_returns_html(self, client: TestClient) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_login_page_redirects_authenticated_user_to_home(
        self, auth_client: TestClient
    ) -> None:
        resp = auth_client.get("/login", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    def test_register_page_redirects_non_admin_to_home(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/register", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    def test_register_page_accessible_by_admin(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/register")
        assert resp.status_code == 200

    @pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
    def test_step_page_renders(self, auth_client: TestClient, step: int) -> None:
        resp = auth_client.get(f"/step/{step}")
        assert resp.status_code == 200

    def test_index_page_renders(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/")
        assert resp.status_code == 200
