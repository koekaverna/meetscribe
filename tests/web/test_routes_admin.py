"""Tests for admin panel routes: access control, user/team CRUD, status, disk, error log."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import meetscribe.config as config_mod
from meetscribe.config import ServerInfo
from meetscribe.database import create_team, get_db, get_team, get_user_by_username
from meetscribe.web.services.auth import verify_password

ADMIN_ROUTES = [
    ("GET", "/api/admin/users"),
    ("POST", "/api/admin/users"),
    ("PATCH", "/api/admin/users/someone"),
    ("POST", "/api/admin/users/someone/password"),
    ("DELETE", "/api/admin/users/someone"),
    ("GET", "/api/admin/teams"),
    ("POST", "/api/admin/teams"),
    ("DELETE", "/api/admin/teams/someteam"),
    ("GET", "/api/admin/status"),
    ("GET", "/api/admin/disk"),
    ("GET", "/api/admin/errors"),
]

SUPERADMIN_ROUTES = [
    ("GET", "/api/admin/teams"),
    ("POST", "/api/admin/teams"),
    ("DELETE", "/api/admin/teams/someteam"),
    ("GET", "/api/admin/status"),
    ("GET", "/api/admin/disk"),
    ("GET", "/api/admin/errors"),
]


def _create_user(client: TestClient, username: str, team: str = "default", admin: bool = False):
    return client.post(
        "/api/admin/users",
        json={
            "username": username,
            "password": "test-pass-000",
            "team_name": team,
            "is_admin": admin,
        },
    )


class TestAdminAccess:
    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    def test_non_admin_gets_403(self, auth_client: TestClient, method: str, path: str) -> None:
        resp = auth_client.request(method, path)
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Admin access required"

    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    def test_unauthenticated_gets_401(self, client: TestClient, method: str, path: str) -> None:
        resp = client.request(method, path)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated"

    @pytest.mark.parametrize("method,path", SUPERADMIN_ROUTES)
    def test_team_admin_gets_403_on_superadmin_routes(
        self, admin_client: TestClient, method: str, path: str
    ) -> None:
        resp = admin_client.request(method, path)
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Superadmin access required"


class TestAdminPage:
    def test_redirects_non_admin_to_home(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/admin", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    def test_redirects_unauthenticated_to_login(self, client: TestClient) -> None:
        resp = client.get("/admin", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_renders_for_admin(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/admin")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_nav_link_rendered_only_for_admin(
        self, client: TestClient, admin_user, regular_user
    ) -> None:
        client.cookies.set("meetscribe_session", regular_user[1])
        assert "navigate('/admin')" not in client.get("/").text
        client.cookies.set("meetscribe_session", admin_user[1])
        assert "navigate('/admin')" in client.get("/").text


class TestUsers:
    def test_lists_users_with_fields(self, admin_client: TestClient, regular_user) -> None:
        resp = admin_client.get("/api/admin/users")
        assert resp.status_code == 200
        users = {u["username"]: u for u in resp.json()}
        assert set(users) == {"admin", "regular"}
        assert users["admin"]["is_admin"] is True
        assert users["admin"]["is_superadmin"] is False
        assert users["regular"]["is_admin"] is False
        assert users["regular"]["team_name"] == "default"
        assert users["regular"]["created_at"] != ""

    def test_create_user_stores_hash_and_team(self, admin_client: TestClient) -> None:
        resp = _create_user(admin_client, "newbie")
        assert resp.status_code == 200
        assert resp.json()["username"] == "newbie"
        assert resp.json()["is_admin"] is False
        row = get_user_by_username(get_db(), "newbie")
        assert row["team_name"] == "default"
        assert verify_password("test-pass-000", row["password_hash"]) is True

    def test_create_admin_user_sets_flag(self, admin_client: TestClient) -> None:
        resp = _create_user(admin_client, "second-admin", admin=True)
        assert resp.status_code == 200
        assert resp.json()["is_admin"] is True
        assert get_user_by_username(get_db(), "second-admin")["is_admin"] == 1

    def test_create_duplicate_username_returns_409(self, admin_client: TestClient) -> None:
        resp = _create_user(admin_client, "admin")
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Username already taken"

    def test_create_with_unknown_team_returns_400(self, superadmin_client: TestClient) -> None:
        resp = _create_user(superadmin_client, "newbie", team="no-such-team")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Team 'no-such-team' not found"

    def test_create_with_slash_in_username_returns_422(self, admin_client: TestClient) -> None:
        # "/" would break the /api/admin/users/{username} manage endpoints
        resp = _create_user(admin_client, "bad/name")
        assert resp.status_code == 422

    def test_create_with_short_password_returns_400(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/api/admin/users",
            json={"username": "newbie", "password": "short", "team_name": "default"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Password must be at least 8 characters"

    def test_delete_user_removes_row(self, admin_client: TestClient, regular_user) -> None:
        resp = admin_client.delete("/api/admin/users/regular")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}
        assert get_user_by_username(get_db(), "regular") is None

    def test_delete_missing_user_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.delete("/api/admin/users/ghost")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "User not found"

    def test_delete_last_admin_refused(self, admin_client: TestClient) -> None:
        # "admin" is the only admin; deleting it would lock everyone out
        resp = admin_client.delete("/api/admin/users/admin")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot delete the last admin"
        assert get_user_by_username(get_db(), "admin")["is_admin"] == 1

    def test_delete_self_refused_even_with_other_admin(self, admin_client: TestClient) -> None:
        _create_user(admin_client, "second-admin", admin=True)
        resp = admin_client.delete("/api/admin/users/admin")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot delete your own account"

    def test_delete_other_admin_allowed(self, admin_client: TestClient) -> None:
        _create_user(admin_client, "second-admin", admin=True)
        resp = admin_client.delete("/api/admin/users/second-admin")
        assert resp.status_code == 200
        assert get_user_by_username(get_db(), "second-admin") is None


def _client_as(app, token: str) -> TestClient:
    """Separate client for tests that need two authenticated roles at once."""
    c = TestClient(app)
    c.cookies.set("meetscribe_session", token)
    return c


class TestUserScoping:
    @pytest.fixture
    def teamb_admin(self, web_auth_service):
        """Admin of a second team ('teamb'). Returns (AuthUser, token)."""
        create_team(get_db(), "teamb")
        user, token = web_auth_service.register("badmin", "test-pass-000", "teamb")
        get_db().execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user.id,))
        get_db().commit()
        return user, token

    def test_team_admin_sees_only_own_team(self, admin_client: TestClient, teamb_admin) -> None:
        users = admin_client.get("/api/admin/users").json()
        assert [u["username"] for u in users] == ["admin"]

    def test_superadmin_sees_all_teams(self, superadmin_client: TestClient, teamb_admin) -> None:
        users = superadmin_client.get("/api/admin/users").json()
        assert {u["username"] for u in users} == {"superadmin", "badmin"}

    def test_team_admin_cannot_create_in_other_team(
        self, admin_client: TestClient, teamb_admin
    ) -> None:
        resp = _create_user(admin_client, "spy", team="teamb")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Team admins can only create users in their own team"

    def test_omitted_team_defaults_to_own(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/api/admin/users", json={"username": "mate", "password": "test-pass-000"}
        )
        assert resp.status_code == 200
        assert resp.json()["team_name"] == "default"

    def test_team_admin_cannot_delete_other_team_user(
        self, admin_client: TestClient, teamb_admin
    ) -> None:
        # 404, not 403: other teams' users are hidden entirely
        resp = admin_client.delete("/api/admin/users/badmin")
        assert resp.status_code == 404
        assert get_user_by_username(get_db(), "badmin") is not None

    def test_team_admin_cannot_delete_superadmin(self, app, admin_user, superadmin_user) -> None:
        resp = _client_as(app, admin_user[1]).delete("/api/admin/users/superadmin")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Cannot delete a superadmin"

    def test_delete_last_superadmin_refused(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.delete("/api/admin/users/superadmin")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot delete the last superadmin"

    def test_superadmin_can_delete_other_superadmin(
        self, superadmin_client: TestClient, web_auth_service
    ) -> None:
        user, _ = web_auth_service.register("super2", "test-pass-000", "default")
        get_db().execute(
            "UPDATE users SET is_admin = 1, is_superadmin = 1 WHERE id = ?", (user.id,)
        )
        get_db().commit()
        resp = superadmin_client.delete("/api/admin/users/super2")
        assert resp.status_code == 200
        assert get_user_by_username(get_db(), "super2") is None


class TestPasswordReset:
    def test_resets_password_and_invalidates_sessions(
        self, app, superadmin_client: TestClient, regular_user
    ) -> None:
        target = _client_as(app, regular_user[1])
        assert target.get("/api/session").status_code == 200

        resp = superadmin_client.post(
            "/api/admin/users/regular/password", json={"password": "test-pass-001"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "reset"}
        row = get_user_by_username(get_db(), "regular")
        assert verify_password("test-pass-001", row["password_hash"]) is True
        assert target.get("/api/session").status_code == 401

    def test_short_password_returns_400(self, superadmin_client: TestClient, regular_user) -> None:
        resp = superadmin_client.post(
            "/api/admin/users/regular/password", json={"password": "short"}
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Password must be at least 8 characters"

    def test_team_admin_resets_within_team(self, admin_client: TestClient, regular_user) -> None:
        resp = admin_client.post(
            "/api/admin/users/regular/password", json={"password": "test-pass-001"}
        )
        assert resp.status_code == 200

    def test_team_admin_cannot_reset_superadmin(self, app, admin_user, superadmin_user) -> None:
        resp = _client_as(app, admin_user[1]).post(
            "/api/admin/users/superadmin/password", json={"password": "test-pass-001"}
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Cannot reset a superadmin's password"

    def test_missing_user_returns_404(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.post(
            "/api/admin/users/ghost/password", json={"password": "test-pass-001"}
        )
        assert resp.status_code == 404


class TestRolePatch:
    def test_grant_admin(self, superadmin_client: TestClient, regular_user) -> None:
        resp = superadmin_client.patch("/api/admin/users/regular", json={"is_admin": True})
        assert resp.status_code == 200
        assert resp.json()["is_admin"] is True
        assert get_user_by_username(get_db(), "regular")["is_admin"] == 1

    def test_revoke_admin(self, superadmin_client: TestClient, admin_user) -> None:
        resp = superadmin_client.patch("/api/admin/users/admin", json={"is_admin": False})
        assert resp.status_code == 200
        assert get_user_by_username(get_db(), "admin")["is_admin"] == 0

    def test_team_admin_grants_within_team(self, admin_client: TestClient, regular_user) -> None:
        resp = admin_client.patch("/api/admin/users/regular", json={"is_admin": True})
        assert resp.status_code == 200

    def test_own_role_change_refused(self, admin_client: TestClient) -> None:
        resp = admin_client.patch("/api/admin/users/admin", json={"is_admin": False})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot change your own role"

    def test_superadmin_role_locked(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.patch("/api/admin/users/superadmin", json={"is_admin": False})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Superadmin role is managed via CLI"

    def test_team_admin_cannot_patch_other_team(
        self, admin_client: TestClient, web_auth_service
    ) -> None:
        create_team(get_db(), "teamb")
        web_auth_service.register("bob", "test-pass-000", "teamb")
        resp = admin_client.patch("/api/admin/users/bob", json={"is_admin": True})
        assert resp.status_code == 404


class TestTeams:
    def test_lists_teams_with_counts(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.get("/api/admin/teams")
        assert resp.status_code == 200
        teams = {t["name"]: t for t in resp.json()}
        assert teams["default"]["user_count"] == 1  # only the superadmin fixture user
        assert teams["default"]["session_count"] == 0
        assert teams["default"]["voiceprint_count"] == 0

    def test_voiceprint_count_reflects_db(self, superadmin_client: TestClient) -> None:
        conn = get_db()
        team_id = get_team(conn, "default")["id"]
        conn.execute(
            "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
            (team_id, "Alice", json.dumps([0.1] * 4), "test"),
        )
        conn.commit()
        teams = {t["name"]: t for t in superadmin_client.get("/api/admin/teams").json()}
        assert teams["default"]["voiceprint_count"] == 1

    def test_create_team_creates_row_and_dirs(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(config_mod, "TEAMS_DIR", tmp_path / "teams")
        resp = superadmin_client.post(
            "/api/admin/teams", json={"name": "squad", "description": "the squad"}
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "squad"
        assert resp.json()["description"] == "the squad"
        assert resp.json()["user_count"] == 0
        assert get_team(get_db(), "squad")["description"] == "the squad"
        assert (tmp_path / "teams" / "squad" / "samples" / "enrolled").is_dir()

    def test_create_team_with_invalid_name_returns_400(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.post("/api/admin/teams", json={"name": "bad name!"})
        assert resp.status_code == 400
        assert "Invalid team name" in resp.json()["detail"]

    def test_create_duplicate_team_returns_409(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.post("/api/admin/teams", json={"name": "default"})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Team 'default' already exists"

    def test_delete_team_cascades_voiceprints_and_removes_dir(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(config_mod, "TEAMS_DIR", tmp_path / "teams")
        superadmin_client.post("/api/admin/teams", json={"name": "doomed"})
        conn = get_db()
        team_id = get_team(conn, "doomed")["id"]
        conn.execute(
            "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
            (team_id, "Bob", json.dumps([0.1] * 4), "test"),
        )
        conn.commit()

        resp = superadmin_client.delete("/api/admin/teams/doomed")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}
        assert get_team(conn, "doomed") is None
        vp = conn.execute("SELECT id FROM voiceprints WHERE team_id = ?", (team_id,)).fetchone()
        assert vp is None  # ON DELETE CASCADE
        assert not (tmp_path / "teams" / "doomed").exists()

    def test_delete_team_with_users_returns_409(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(config_mod, "TEAMS_DIR", tmp_path / "teams")
        superadmin_client.post("/api/admin/teams", json={"name": "staffed"})
        _create_user(superadmin_client, "member", team="staffed")

        resp = superadmin_client.delete("/api/admin/teams/staffed")
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Team still has users or sessions. Delete them first."
        assert get_team(get_db(), "staffed") is not None

    def test_delete_default_team_returns_400(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.delete("/api/admin/teams/default")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Cannot delete the 'default' team"

    def test_delete_missing_team_returns_404(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.delete("/api/admin/teams/ghost")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Team not found"


class TestStatus:
    def test_reachable_server_reports_latency(self, superadmin_client: TestClient) -> None:
        config_mod._app_config.servers = [ServerInfo(url="http://speaches:8000", name="main")]
        with patch("meetscribe.web.routes.admin.httpx.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)
            resp = superadmin_client.get("/api/admin/status")
        assert resp.status_code == 200
        [status] = resp.json()
        assert status["name"] == "main"
        assert status["url"] == "http://speaches:8000"
        assert status["reachable"] is True
        assert status["latency_ms"] >= 0
        assert status["error"] is None
        mock_get.assert_called_once_with("http://speaches:8000/health", timeout=3.0)

    def test_unreachable_server_reports_error(self, superadmin_client: TestClient) -> None:
        config_mod._app_config.servers = [ServerInfo(url="http://speaches:8000", name="main")]
        with patch("meetscribe.web.routes.admin.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            resp = superadmin_client.get("/api/admin/status")
        assert resp.status_code == 200
        [status] = resp.json()
        assert status["reachable"] is False
        assert status["latency_ms"] is None
        assert status["error"] == "Connection refused"

    def test_unhealthy_status_code_reports_unreachable(self, superadmin_client: TestClient) -> None:
        config_mod._app_config.servers = [ServerInfo(url="http://speaches:8000", name="main")]
        with patch("meetscribe.web.routes.admin.httpx.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error '503 Service Unavailable'", request=Mock(), response=Mock()
            )
            resp = superadmin_client.get("/api/admin/status")
        assert resp.status_code == 200
        [status] = resp.json()
        assert status["reachable"] is False
        assert "503" in status["error"]

    def test_no_servers_configured_returns_empty(self, superadmin_client: TestClient) -> None:
        resp = superadmin_client.get("/api/admin/status")
        assert resp.status_code == 200
        assert resp.json() == []


class TestDisk:
    def test_sums_seeded_directories(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        data_dir = tmp_path / "data"
        (data_dir / "sessions" / "s1").mkdir(parents=True)
        (data_dir / "teams" / "default" / "samples").mkdir(parents=True)
        (data_dir / "sessions" / "s1" / "track.wav").write_bytes(b"x" * 2048)
        (data_dir / "teams" / "default" / "samples" / "alice.wav").write_bytes(b"y" * 512)
        (data_dir / "meetscribe.db").write_bytes(b"z" * 100)
        monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)
        monkeypatch.setattr(config_mod, "TEAMS_DIR", data_dir / "teams")

        resp = superadmin_client.get("/api/admin/disk")
        assert resp.status_code == 200
        assert resp.json() == {
            "total_bytes": 2660,
            "sessions_bytes": 2048,
            "samples_bytes": 512,
        }

    def test_missing_dirs_report_zero(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "nowhere")
        monkeypatch.setattr(config_mod, "TEAMS_DIR", tmp_path / "nowhere" / "teams")
        resp = superadmin_client.get("/api/admin/disk")
        assert resp.json() == {"total_bytes": 0, "sessions_bytes": 0, "samples_bytes": 0}


class TestErrors:
    def test_tails_error_lines_from_newest_file(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        old = logs / "web_old.log"
        old.write_text("2026-07-16 09:00:00 [ERROR] app: from old file\n", encoding="utf-8")
        new = logs / "web_new.log"
        new.write_text(
            "2026-07-17 10:00:00 [INFO] app: started\n"
            "2026-07-17 10:00:01 [ERROR] app: boom one\n"
            "2026-07-17 10:00:02 [DEBUG] app: detail\n"
            "2026-07-17 10:00:03 [ERROR] app: boom two\n",
            encoding="utf-8",
        )
        os.utime(old, (1000, 1000))  # force "old" to be older regardless of write order
        monkeypatch.setattr(config_mod, "LOGS_DIR", logs)

        resp = superadmin_client.get("/api/admin/errors")
        assert resp.status_code == 200
        assert resp.json() == {
            "file": "web_new.log",
            "lines": [
                "2026-07-17 10:00:01 [ERROR] app: boom one",
                "2026-07-17 10:00:03 [ERROR] app: boom two",
            ],
        }

    def test_limit_returns_last_n_lines(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "web.log").write_text(
            "2026-07-17 10:00:01 [ERROR] app: first\n2026-07-17 10:00:02 [ERROR] app: last\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_mod, "LOGS_DIR", logs)

        resp = superadmin_client.get("/api/admin/errors?limit=1")
        assert resp.json()["lines"] == ["2026-07-17 10:00:02 [ERROR] app: last"]

    def test_no_log_files_returns_empty(
        self, superadmin_client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(config_mod, "LOGS_DIR", logs)
        resp = superadmin_client.get("/api/admin/errors")
        assert resp.json() == {"file": None, "lines": []}
