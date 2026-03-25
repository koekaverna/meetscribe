"""Tests for session management routes."""

from fastapi.testclient import TestClient

from meetscribe.database import create_team
from meetscribe.web.services.auth import AuthService


class TestCreateSession:
    def test_returns_session_id(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        resp = client.post("/api/session")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]


class TestGetSession:
    def test_returns_session_with_created_status(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert data["status"] == "created"

    def test_nonexistent_returns_404(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/api/session/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestDeleteSession:
    def test_removes_session_from_db_and_api(self, auth_client: TestClient, web_db) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]

        resp = auth_client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        row = web_db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        assert row is None

        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 404

    def test_nonexistent_returns_404(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/session/nonexistent")
        assert resp.status_code == 404


class TestTeamIsolation:
    """Users from different teams cannot access each other's sessions."""

    def _create_other_team_session(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> str:
        """Helper: create a session belonging to 'other_team'."""
        create_team(web_db, "other_team")
        _, other_token = web_auth_service.register("other_user", "password123", "other_team")
        client.cookies.set("meetscribe_session", other_token)
        return client.post("/api/session").json()["session_id"]

    def _switch_to_default_team(self, client: TestClient, web_auth_service: AuthService) -> None:
        """Helper: switch client to a user from 'default' team."""
        _, token = web_auth_service.register("default_user", "password123", "default")
        client.cookies.set("meetscribe_session", token)

    def test_cannot_read_other_teams_session(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> None:
        other_session_id = self._create_other_team_session(client, web_db, web_auth_service)
        self._switch_to_default_team(client, web_auth_service)

        resp = client.get(f"/api/session/{other_session_id}")
        assert resp.status_code == 404

    def test_cannot_delete_other_teams_session(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> None:
        other_session_id = self._create_other_team_session(client, web_db, web_auth_service)
        self._switch_to_default_team(client, web_auth_service)

        resp = client.delete(f"/api/session/{other_session_id}")
        assert resp.status_code == 404
