"""Tests for session management routes."""

import pytest
from fastapi.testclient import TestClient

from .conftest_web import *  # noqa: F401, F403 — import web fixtures


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
    def test_delete_session(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        resp = auth_client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/session/nonexistent")
        assert resp.status_code == 404
