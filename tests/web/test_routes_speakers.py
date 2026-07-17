"""Tests for enrolled speakers routes."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import meetscribe.config as config_mod
from meetscribe.database import create_team, get_db
from meetscribe.errors import SpeachesAPIError
from meetscribe.web.services.auth import AuthService
from meetscribe.web.services.pipeline import get_speaker_sample_path, list_speaker_samples
from tests.conftest import make_wav_file

EMBEDDING_MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"


@pytest.fixture(autouse=True)
def teams_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect team sample dirs into tmp so tests never touch real data."""
    d = tmp_path / "teams"
    monkeypatch.setattr(config_mod, "TEAMS_DIR", d)
    return d


def _insert_voiceprint(web_db, name: str, team: str = "default") -> int:
    """Helper: insert a voiceprint and return team_id."""
    row = get_db().execute("SELECT id FROM teams WHERE name = ?", (team,)).fetchone()
    get_db().execute(
        "INSERT INTO voiceprints (team_id, name, embedding, model) VALUES (?, ?, ?, ?)",
        (row["id"], name, json.dumps([0.1] * 256), "test"),
    )
    get_db().commit()
    return row["id"]


def _add_samples(name: str, durations_s: list[float], team: str = "default") -> Path:
    """Helper: create the speaker's enrolled dir with s0.wav, s1.wav, ..."""
    d = config_mod.get_team_enrolled_dir(team) / name
    d.mkdir(parents=True, exist_ok=True)
    for i, dur in enumerate(durations_s):
        make_wav_file(d / f"s{i}.wav", dur)
    return d


def _stored_embedding(team_id: int, name: str) -> list[float] | None:
    row = (
        get_db()
        .execute(
            "SELECT embedding FROM voiceprints WHERE team_id = ? AND name = ?",
            (team_id, name),
        )
        .fetchone()
    )
    return json.loads(row["embedding"]) if row else None


@pytest.fixture
def other_team_client(app, web_auth_service: AuthService) -> TestClient:
    """Client authenticated as an admin of a second team ('teamb')."""
    create_team(get_db(), "teamb")
    user, token = web_auth_service.register("bob", "test-pass-000", "teamb")
    get_db().execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user.id,))
    get_db().commit()
    c = TestClient(app)
    c.cookies.set("meetscribe_session", token)
    return c


class TestListSpeakers:
    def test_no_voiceprints_returns_empty_list(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/api/speakers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_enrolled_speaker_names(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = admin_client.get("/api/speakers")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "Alice" in names

    def test_returns_sample_stats_and_model(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        _add_samples("Alice", [1.0, 2.0])
        resp = admin_client.get("/api/speakers")
        assert resp.status_code == 200
        speaker = resp.json()[0]
        assert speaker["name"] == "Alice"
        assert speaker["model"] == "test"
        assert speaker["sample_count"] == 2
        assert speaker["total_duration_ms"] == 3000

    def test_speaker_without_samples_dir_reports_zero(
        self, admin_client: TestClient, web_db
    ) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = admin_client.get("/api/speakers")
        speaker = resp.json()[0]
        assert speaker["sample_count"] == 0
        assert speaker["total_duration_ms"] == 0


class TestListSamples:
    def test_lists_wav_files_with_duration(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        _add_samples("Alice", [1.0, 2.0])
        resp = admin_client.get("/api/speakers/Alice/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert [s["filename"] for s in samples] == ["s0.wav", "s1.wav"]
        assert [s["duration_ms"] for s in samples] == [1000, 2000]

    def test_unknown_speaker_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/api/speakers/NoSuchPerson/samples")
        assert resp.status_code == 404

    def test_unsafe_speaker_name_rejected(self, web_db) -> None:
        with pytest.raises(LookupError):
            list_speaker_samples("..")


class TestSampleAudio:
    def test_streams_wav_bytes(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        d = _add_samples("Alice", [1.0])
        resp = admin_client.get("/api/speakers/Alice/samples/s0.wav/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == (d / "s0.wav").read_bytes()

    def test_missing_sample_returns_404(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = admin_client.get("/api/speakers/Alice/samples/nope.wav/audio")
        assert resp.status_code == 404

    def test_traversal_filename_rejected(self, web_db, tmp_path: Path) -> None:
        _insert_voiceprint(web_db, "Alice")
        _add_samples("Alice", [1.0])
        # Would resolve outside the speaker dir if not rejected
        (config_mod.get_team_enrolled_dir("default") / "secret.wav").write_bytes(b"x")
        with pytest.raises(LookupError):
            get_speaker_sample_path("Alice", "../secret.wav")


class TestDeleteSample:
    def test_recomputes_voiceprint_from_remaining(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        d = _add_samples("Alice", [1.0, 1.0, 1.0])

        extractor = MagicMock()
        extractor.extract_from_file.side_effect = [[1.0, 0.0], [0.0, 1.0]]
        with patch("meetscribe.web.services.pipeline._make_extractor", return_value=extractor):
            resp = admin_client.delete("/api/speakers/Alice/samples/s2.wav")

        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}
        assert not (d / "s2.wav").exists()
        assert (d / "s0.wav").exists()
        assert (d / "s1.wav").exists()
        # Voiceprint recomputed as the average of the two remaining samples
        called_paths = [c.args[0] for c in extractor.extract_from_file.call_args_list]
        assert called_paths == [d / "s0.wav", d / "s1.wav"]
        assert _stored_embedding(team_id, "Alice") == [0.5, 0.5]
        row = (
            get_db()
            .execute(
                "SELECT model FROM voiceprints WHERE team_id = ? AND name = ?",
                (team_id, "Alice"),
            )
            .fetchone()
        )
        assert row["model"] == EMBEDDING_MODEL

    def test_last_sample_rejected(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        d = _add_samples("Alice", [1.0])
        resp = admin_client.delete("/api/speakers/Alice/samples/s0.wav")
        assert resp.status_code == 409
        assert "delete the speaker" in resp.json()["detail"].lower()
        assert (d / "s0.wav").exists()
        assert _stored_embedding(team_id, "Alice") == [0.1] * 256

    def test_missing_sample_returns_404(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        _add_samples("Alice", [1.0])
        resp = admin_client.delete("/api/speakers/Alice/samples/nope.wav")
        assert resp.status_code == 404

    def test_embedding_api_failure_keeps_sample(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        d = _add_samples("Alice", [1.0, 1.0])

        extractor = MagicMock()
        extractor.extract_from_file.side_effect = SpeachesAPIError("server down")
        with patch("meetscribe.web.services.pipeline._make_extractor", return_value=extractor):
            resp = admin_client.delete("/api/speakers/Alice/samples/s1.wav")

        assert resp.status_code == 502
        assert (d / "s1.wav").exists()
        assert _stored_embedding(team_id, "Alice") == [0.1] * 256


class TestRenameSpeaker:
    def test_renames_voiceprint_and_samples_dir(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        old_dir = _add_samples("Alice", [1.0])
        resp = admin_client.patch("/api/speakers/Alice", json={"name": "Alicia"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "renamed"}

        names = [
            r["name"]
            for r in get_db()
            .execute("SELECT name FROM voiceprints WHERE team_id = ?", (team_id,))
            .fetchall()
        ]
        assert names == ["Alicia"]
        new_dir = old_dir.parent / "Alicia"
        assert not old_dir.exists()
        assert (new_dir / "s0.wav").is_file()

    def test_collision_returns_409(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        _insert_voiceprint(web_db, "Bob")
        resp = admin_client.patch("/api/speakers/Alice", json={"name": "Bob"})
        assert resp.status_code == 409
        names = {
            r["name"]
            for r in get_db()
            .execute("SELECT name FROM voiceprints WHERE team_id = ?", (team_id,))
            .fetchall()
        }
        assert names == {"Alice", "Bob"}

    def test_unknown_speaker_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.patch("/api/speakers/NoSuchPerson", json={"name": "X"})
        assert resp.status_code == 404

    def test_invalid_new_name_returns_400(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = admin_client.patch("/api/speakers/Alice", json={"name": "../evil"})
        assert resp.status_code == 400

    def test_same_name_is_noop(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        resp = admin_client.patch("/api/speakers/Alice", json={"name": "Alice"})
        assert resp.status_code == 200


class TestDeleteSpeaker:
    def test_removes_voiceprint_from_db(self, admin_client: TestClient, web_db) -> None:
        team_id = _insert_voiceprint(web_db, "Bob")
        resp = admin_client.delete("/api/speakers/Bob")
        assert resp.status_code == 200

        row = (
            get_db()
            .execute(
                "SELECT name FROM voiceprints WHERE team_id = ? AND name = ?",
                (team_id, "Bob"),
            )
            .fetchone()
        )
        assert row is None

    def test_removes_enrolled_samples_dir(self, admin_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Bob")
        d = _add_samples("Bob", [1.0])
        resp = admin_client.delete("/api/speakers/Bob")
        assert resp.status_code == 200
        assert not d.exists()

    def test_nonexistent_speaker_returns_404(self, admin_client: TestClient) -> None:
        resp = admin_client.delete("/api/speakers/NoSuchPerson")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        resp = client.delete("/api/speakers/Bob")
        assert resp.status_code == 401


class TestAdminOnly:
    def test_regular_user_gets_403(self, auth_client: TestClient, web_db) -> None:
        _insert_voiceprint(web_db, "Alice")
        assert auth_client.get("/api/speakers").status_code == 403
        assert auth_client.get("/api/speakers/Alice/samples").status_code == 403
        assert auth_client.patch("/api/speakers/Alice", json={"name": "X"}).status_code == 403
        assert auth_client.delete("/api/speakers/Alice").status_code == 403

    def test_regular_user_redirected_from_page(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/speakers", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"


class TestTeamIsolation:
    def test_other_team_speakers_invisible(
        self, admin_client: TestClient, other_team_client: TestClient, web_db
    ) -> None:
        _insert_voiceprint(web_db, "Alice")
        _add_samples("Alice", [1.0])
        assert other_team_client.get("/api/speakers").json() == []
        assert [s["name"] for s in admin_client.get("/api/speakers").json()] == ["Alice"]

    def test_other_team_cannot_access_or_mutate(
        self, other_team_client: TestClient, web_db
    ) -> None:
        team_id = _insert_voiceprint(web_db, "Alice")
        d = _add_samples("Alice", [1.0, 1.0])

        assert other_team_client.get("/api/speakers/Alice/samples").status_code == 404
        assert other_team_client.get("/api/speakers/Alice/samples/s0.wav/audio").status_code == 404
        assert other_team_client.delete("/api/speakers/Alice/samples/s0.wav").status_code == 404
        assert other_team_client.patch("/api/speakers/Alice", json={"name": "X"}).status_code == 404
        assert other_team_client.delete("/api/speakers/Alice").status_code == 404
        # Nothing changed for the owning team
        assert _stored_embedding(team_id, "Alice") == [0.1] * 256
        assert (d / "s0.wav").exists()
