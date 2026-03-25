"""Tests for enrolled speakers routes."""

import json

from fastapi.testclient import TestClient


class TestListSpeakers:
    def test_list_speakers_empty(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/api/speakers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_speakers_with_voiceprints(self, auth_client: TestClient, web_db) -> None:
        team = web_db.execute("SELECT id FROM teams WHERE name = 'default'").fetchone()
        web_db.execute(
            "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
            (team["id"], "Alice", json.dumps([0.1] * 256), "test"),
        )
        web_db.commit()
        resp = auth_client.get("/api/speakers")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "Alice" in names


class TestDeleteSpeaker:
    def test_delete_nonexistent_speaker(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/speakers/NoSuchPerson")
        assert resp.status_code == 404

    def test_delete_removes_from_db(self, auth_client: TestClient, web_db) -> None:
        team = web_db.execute("SELECT id FROM teams WHERE name = 'default'").fetchone()
        web_db.execute(
            "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
            (team["id"], "Bob", json.dumps([0.2] * 256), "test"),
        )
        web_db.commit()
        resp = auth_client.delete("/api/speakers/Bob")
        assert resp.status_code == 200

        # Verify actually deleted from DB
        row = web_db.execute(
            "SELECT name FROM voiceprints WHERE team_id = ? AND name = ?",
            (team["id"], "Bob"),
        ).fetchone()
        assert row is None

    def test_unauthenticated_delete(self, client: TestClient) -> None:
        resp = client.delete("/api/speakers/Bob")
        assert resp.status_code == 401
