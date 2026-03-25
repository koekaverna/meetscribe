"""Tests for FastAPI app: middleware, CSRF, auth, health, and page routes."""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuthMiddleware:
    def test_unauthenticated_page_redirects_to_login(self, client: TestClient) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]

    def test_public_paths_accessible(self, client: TestClient) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_api_routes_return_401_not_redirect(self, client: TestClient) -> None:
        resp = client.post("/api/session")
        assert resp.status_code == 401

    def test_authenticated_user_can_access_pages(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 200


class TestCSRFMiddleware:
    def test_csrf_cookie_set_on_first_request(self, client: TestClient) -> None:
        client.get("/login")
        assert "meetscribe_csrf" in client.cookies

    def test_csrf_cookie_is_64_hex_chars(self, client: TestClient) -> None:
        client.get("/login")
        token = client.cookies["meetscribe_csrf"]
        assert len(token) == 64
        int(token, 16)  # validates hex format


class TestPageRoutes:
    def test_login_page_renders(self, client: TestClient) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_login_page_redirects_if_authenticated(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/login", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    def test_register_page_requires_admin(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/register", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    def test_register_page_accessible_by_admin(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/register")
        assert resp.status_code == 200

    def test_step_pages(self, auth_client: TestClient) -> None:
        for step in range(1, 7):
            resp = auth_client.get(f"/step/{step}")
            assert resp.status_code == 200, f"step {step} failed"

    def test_index_page(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/")
        assert resp.status_code == 200
