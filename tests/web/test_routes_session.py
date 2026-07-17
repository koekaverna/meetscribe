"""Tests for session management routes."""

import pytest
from fastapi.testclient import TestClient

from meetscribe.database import create_team, get_db
from meetscribe.web.services.auth import AuthService, AuthUser


def _seed_transcript(session_id: str, transcript: str, segments: list[tuple]) -> None:
    """Mark a session transcribed with segments: (track_num, start_ms, end_ms, speaker, text)."""
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET status = 'transcribed', transcript = ? WHERE id = ?",
        (transcript, session_id),
    )
    conn.executemany(
        "INSERT INTO session_segments "
        "(session_id, track_num, start_ms, end_ms, speaker, text, sort_order) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(session_id, *seg, i) for i, seg in enumerate(segments)],
    )
    conn.commit()


def _backdate(session_id: str, created_at: str) -> None:
    conn = get_db()
    conn.execute("UPDATE sessions SET created_at = ? WHERE id = ?", (created_at, session_id))
    conn.commit()


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

    def test_persists_creator_id(
        self, auth_client: TestClient, regular_user: tuple[AuthUser, str]
    ) -> None:
        user, _ = regular_user
        session_id = auth_client.post("/api/session").json()["session_id"]
        row = (
            get_db()
            .execute("SELECT creator_id FROM sessions WHERE id = ?", (session_id,))
            .fetchone()
        )
        assert row["creator_id"] == user.id


class TestListSessions:
    def test_empty_list(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/api/session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 20

    def test_created_session_summary_fields(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        data = auth_client.get("/api/session").json()
        assert data["total"] == 1
        row = data["sessions"][0]
        assert row["id"] == session_id
        assert row["status"] == "created"
        assert row["creator"] == "regular"
        assert row["track_count"] == 0
        assert row["duration_ms"] is None
        assert row["speakers"] == []
        assert row["preview"] is None

    def test_transcribed_session_computed_fields(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        transcript = "т" * 200
        _seed_transcript(
            session_id,
            transcript,
            [
                (1, 0, 30000, "Alice", "hello"),
                (1, 30000, 61000, "Unknown-1", "world"),
            ],
        )

        row = auth_client.get("/api/session").json()["sessions"][0]
        assert row["status"] == "transcribed"
        assert row["duration_ms"] == 61000
        assert row["speakers"] == ["Alice", "Unknown-1"]
        assert row["preview"] == transcript[:150]

    def test_pagination(self, auth_client: TestClient) -> None:
        for _ in range(3):
            auth_client.post("/api/session")

        page1 = auth_client.get("/api/session?page=1&per_page=2").json()
        assert len(page1["sessions"]) == 2
        assert page1["total"] == 3

        page2 = auth_client.get("/api/session?page=2&per_page=2").json()
        assert len(page2["sessions"]) == 1
        assert page2["total"] == 3

    def test_sort_by_date(self, auth_client: TestClient) -> None:
        ids = [auth_client.post("/api/session").json()["session_id"] for _ in range(3)]
        for i, session_id in enumerate(ids, 1):
            _backdate(session_id, f"2026-01-0{i} 10:00:00")

        asc = auth_client.get("/api/session?sort=date&order=asc").json()
        assert [s["id"] for s in asc["sessions"]] == ids

        desc = auth_client.get("/api/session?sort=date&order=desc").json()
        assert [s["id"] for s in desc["sessions"]] == list(reversed(ids))

    def test_sort_by_duration_nulls_last(self, auth_client: TestClient) -> None:
        with_duration = auth_client.post("/api/session").json()["session_id"]
        without_duration = auth_client.post("/api/session").json()["session_id"]
        _seed_transcript(with_duration, "text", [(1, 0, 5000, "Alice", "hi")])

        for order in ("asc", "desc"):
            data = auth_client.get(f"/api/session?sort=duration&order={order}").json()
            assert [s["id"] for s in data["sessions"]] == [with_duration, without_duration]

    def test_regular_user_sees_only_own_sessions(
        self, auth_client: TestClient, web_auth_service: AuthService
    ) -> None:
        my_session = auth_client.post("/api/session").json()["session_id"]

        _, other_token = web_auth_service.register("teammate", "test-pass-000", "default")
        my_token = auth_client.cookies["meetscribe_session"]
        auth_client.cookies.set("meetscribe_session", other_token)
        auth_client.post("/api/session")
        auth_client.cookies.set("meetscribe_session", my_token)

        data = auth_client.get("/api/session").json()
        assert data["total"] == 1
        assert data["sessions"][0]["id"] == my_session

    def test_admin_sees_all_team_sessions(
        self, admin_client: TestClient, web_auth_service: AuthService
    ) -> None:
        admin_session = admin_client.post("/api/session").json()["session_id"]

        _, other_token = web_auth_service.register("teammate", "test-pass-000", "default")
        admin_token = admin_client.cookies["meetscribe_session"]
        admin_client.cookies.set("meetscribe_session", other_token)
        admin_client.post("/api/session")
        admin_client.cookies.set("meetscribe_session", admin_token)

        all_team = admin_client.get("/api/session").json()
        assert all_team["total"] == 2

        mine = admin_client.get("/api/session?mine=true").json()
        assert mine["total"] == 1
        assert mine["sessions"][0]["id"] == admin_session

    def test_team_isolation(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> None:
        create_team(get_db(), "other_team")
        _, other_token = web_auth_service.register("other_user", "test-pass-000", "other_team")
        client.cookies.set("meetscribe_session", other_token)
        client.post("/api/session")

        _, token = web_auth_service.register("default_user", "test-pass-000", "default")
        client.cookies.set("meetscribe_session", token)
        data = client.get("/api/session").json()
        assert data["total"] == 0
        assert data["sessions"] == []

    def test_null_creator_visible_to_admin_only(
        self, admin_client: TestClient, web_auth_service: AuthService
    ) -> None:
        """Sessions created before migration 004 have creator_id NULL — only admins see them."""
        session_id = admin_client.post("/api/session").json()["session_id"]
        conn = get_db()
        conn.execute("UPDATE sessions SET creator_id = NULL WHERE id = ?", (session_id,))
        conn.commit()

        row = admin_client.get("/api/session").json()["sessions"][0]
        assert row["id"] == session_id
        assert row["creator"] is None

        _, token = web_auth_service.register("teammate", "test-pass-000", "default")
        admin_client.cookies.set("meetscribe_session", token)
        assert admin_client.get("/api/session").json()["total"] == 0

    @pytest.mark.parametrize(
        "query", ["sort=bogus", "order=sideways", "per_page=0", "per_page=101", "page=0"]
    )
    def test_invalid_params_rejected(self, auth_client: TestClient, query: str) -> None:
        resp = auth_client.get(f"/api/session?{query}")
        assert resp.status_code == 422


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

        row = get_db().execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        assert row is None

        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 404

    def test_nonexistent_returns_404(self, auth_client: TestClient) -> None:
        resp = auth_client.delete("/api/session/nonexistent")
        assert resp.status_code == 404


class TestBulkDelete:
    def test_deletes_own_sessions(self, auth_client: TestClient) -> None:
        ids = [auth_client.post("/api/session").json()["session_id"] for _ in range(3)]

        resp = auth_client.post("/api/session/bulk-delete", json={"ids": ids[:2]})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2

        remaining = get_db().execute("SELECT id FROM sessions").fetchall()
        assert [r["id"] for r in remaining] == [ids[2]]

    def test_skips_teammates_sessions_for_regular_user(
        self, auth_client: TestClient, web_auth_service: AuthService
    ) -> None:
        my_session = auth_client.post("/api/session").json()["session_id"]

        my_token = auth_client.cookies["meetscribe_session"]
        _, other_token = web_auth_service.register("teammate", "test-pass-000", "default")
        auth_client.cookies.set("meetscribe_session", other_token)
        other_session = auth_client.post("/api/session").json()["session_id"]
        auth_client.cookies.set("meetscribe_session", my_token)

        resp = auth_client.post(
            "/api/session/bulk-delete", json={"ids": [my_session, other_session]}
        )
        assert resp.json()["deleted"] == 1

        remaining = get_db().execute("SELECT id FROM sessions").fetchall()
        assert [r["id"] for r in remaining] == [other_session]

    def test_admin_deletes_any_team_session(
        self, admin_client: TestClient, web_auth_service: AuthService
    ) -> None:
        admin_session = admin_client.post("/api/session").json()["session_id"]

        admin_token = admin_client.cookies["meetscribe_session"]
        _, other_token = web_auth_service.register("teammate", "test-pass-000", "default")
        admin_client.cookies.set("meetscribe_session", other_token)
        other_session = admin_client.post("/api/session").json()["session_id"]
        admin_client.cookies.set("meetscribe_session", admin_token)

        resp = admin_client.post(
            "/api/session/bulk-delete", json={"ids": [admin_session, other_session]}
        )
        assert resp.json()["deleted"] == 2
        assert get_db().execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()["cnt"] == 0

    def test_skips_nonexistent_ids(self, auth_client: TestClient) -> None:
        session_id = auth_client.post("/api/session").json()["session_id"]
        resp = auth_client.post(
            "/api/session/bulk-delete", json={"ids": [session_id, "no-such-id"]}
        )
        assert resp.json()["deleted"] == 1

    def test_empty_ids_deletes_nothing(self, auth_client: TestClient) -> None:
        auth_client.post("/api/session")
        resp = auth_client.post("/api/session/bulk-delete", json={"ids": []})
        assert resp.json()["deleted"] == 0
        assert get_db().execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()["cnt"] == 1

    def test_too_many_ids_rejected(self, auth_client: TestClient) -> None:
        resp = auth_client.post(
            "/api/session/bulk-delete", json={"ids": [f"id-{i}" for i in range(101)]}
        )
        assert resp.status_code == 422


class TestOwnership:
    """Non-admins can access only their own sessions; admins any session in the team."""

    def _create_teammate_session(
        self, client: TestClient, web_auth_service: AuthService
    ) -> tuple[str, str]:
        """Create a session as a new 'teammate' user. Returns (session_id, original_token)."""
        original_token = client.cookies["meetscribe_session"]
        _, token = web_auth_service.register("teammate", "test-pass-000", "default")
        client.cookies.set("meetscribe_session", token)
        session_id = client.post("/api/session").json()["session_id"]
        client.cookies.set("meetscribe_session", original_token)
        return session_id, original_token

    def test_regular_user_cannot_read_teammates_session(
        self, auth_client: TestClient, web_auth_service: AuthService
    ) -> None:
        session_id, _ = self._create_teammate_session(auth_client, web_auth_service)
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 404

    def test_regular_user_cannot_delete_teammates_session(
        self, auth_client: TestClient, web_auth_service: AuthService
    ) -> None:
        session_id, _ = self._create_teammate_session(auth_client, web_auth_service)
        resp = auth_client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 404

        row = get_db().execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        assert row["id"] == session_id

    def test_admin_can_read_and_delete_teammates_session(
        self, admin_client: TestClient, web_auth_service: AuthService
    ) -> None:
        session_id, _ = self._create_teammate_session(admin_client, web_auth_service)

        resp = admin_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == session_id

        resp = admin_client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestTeamIsolation:
    """Users from different teams cannot access each other's sessions."""

    def _create_other_team_session(
        self, client: TestClient, web_db, web_auth_service: AuthService
    ) -> str:
        """Helper: create a session belonging to 'other_team'."""
        create_team(get_db(), "other_team")
        _, other_token = web_auth_service.register("other_user", "test-pass-000", "other_team")
        client.cookies.set("meetscribe_session", other_token)
        return client.post("/api/session").json()["session_id"]

    def _switch_to_default_team(self, client: TestClient, web_auth_service: AuthService) -> None:
        """Helper: switch client to a user from 'default' team."""
        _, token = web_auth_service.register("default_user", "test-pass-000", "default")
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
