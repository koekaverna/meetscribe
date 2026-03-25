"""Tests for session management routes."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from meetscribe.database import create_team
from meetscribe.web.services.auth import AuthService, AuthUser, init_auth_service
from meetscribe.web.services.session import get_session_service


class TestCreateSession:
    def test_create_session(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_create_session_unauthenticated(self, client: TestClient) -> None:
        resp = client.post("/api/session")
        assert resp.status_code == 401


class TestGetSession:
    def test_get_session(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert data["status"] == "created"

    def test_get_nonexistent_session(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/api/session/nonexistent")
        assert resp.status_code == 404


class TestDeleteSession:
    def test_delete_session_removes_from_db(self, auth_client: TestClient, web_db) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        resp = auth_client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone from DB
        row = web_db.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        assert row is None

        # Verify API also returns 404
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/session/nonexistent")
        assert resp.status_code == 404


class TestTeamIsolation:
    """Verify that users from different teams cannot access each other's sessions."""

    def test_user_cannot_access_other_teams_session(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> None:
        # Create a second team and user
        create_team(web_db, "other_team")
        other_user, other_token = web_auth_service.register(
            "other_user", "password123", "other_team"
        )

        # Create a session as the "other" user
        client.cookies.set("meetscribe_session", other_token)
        resp = client.post("/api/session")
        assert resp.status_code == 200
        other_session_id = resp.json()["session_id"]

        # Verify the other user can see their own session
        resp = client.get(f"/api/session/{other_session_id}")
        assert resp.status_code == 200

        # Now switch to the "regular" user (default team)
        regular_user, regular_token = web_auth_service.register(
            "regular_iso", "password123", "default"
        )
        client.cookies.set("meetscribe_session", regular_token)

        # Regular user should NOT see the other team's session
        resp = client.get(f"/api/session/{other_session_id}")
        assert resp.status_code == 404

        # Regular user should NOT be able to delete it
        resp = client.delete(f"/api/session/{other_session_id}")
        assert resp.status_code == 404
