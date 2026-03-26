"""Tests for enrolled speakers routes."""

import json

from fastapi.testclient import TestClient


def _insert_voiceprint(web_db, name: str) -> int:
    """Helper: insert a voiceprint and return team_id."""
    team = web_db.execute("SELECT id FROM teams WHERE name = 'default'").fetchone()
    web_db.execute(
        "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
        (team["id"], name, json.dumps([0.1] * 256), "test"),
    )
    web_db.commit()
    return team["id"]


class TestListSpeakers:
    def test_no_voiceprints_returns_empty_list(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/api/speakers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_enrolled_speaker_names(self, auth_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = auth_client.get("/api/speakers")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "Alice" in names


class TestDeleteSpeaker:
    def test_removes_voiceprint_from_db(self, auth_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Bob")
        resp = auth_client.delete("/api/speakers/Bob")
        assert resp.status_code == 200

        row = web_db.execute(
            "SELECT name FROM voiceprints WHERE team_id = ? AND name = ?",
            (team_id, "Bob"),
        ).fetchone()
        assert row is None

    def test_nonexistent_speaker_returns_404(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/speakers/NoSuchPerson")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        resp = client.delete("/api/speakers/Bob")
        assert resp.status_code == 401
