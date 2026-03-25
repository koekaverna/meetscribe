"""Tests for sample and speaker bin routes."""

import pytest
from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service


@pytest.fixture
def sample_id(session_id: str) -> str:
    """Create a sample via service and return its ID."""
    service = get_session_service()
    sample = service.add_sample(
        session_id,
        track_num=1,
        cluster_id=0,
        filename="test.wav",
        duration_ms=1000,
        content=b"\x00" * 100,
    )
    return sample.id


@pytest.fixture
def speaker_id(session_id: str) -> str:
    """Create a speaker via service and return its ID."""
    service = get_session_service()
    return service.add_speaker(session_id, "Alice").id


class TestSpeakerCreate:
    def test_returns_speaker_with_name_and_id(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/speakers",
            json={"name": "Alice"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert "id" in data


class TestSpeakerRename:
    def test_new_name_persists_in_session(
        self, auth_client: TestClient, session_id: str, speaker_id: str
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/speakers/{speaker_id}",
            json={"name": "Bob"},
        )
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        names = [s["name"] for s in state["speakers"]]
        assert "Bob" in names
        assert "Alice" not in names

    def test_nonexistent_speaker_returns_404(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/speakers/nope",
            json={"name": "Bob"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestSpeakerDelete:
    def test_removes_speaker_from_session(
        self, auth_client: TestClient, session_id: str, speaker_id: str
    ) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/speakers/{speaker_id}")
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        assert len(state["speakers"]) == 0

    def test_nonexistent_speaker_returns_404(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/speakers/nope")
        assert resp.status_code == 404


class TestSampleList:
    def test_new_session_returns_empty_list(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/samples")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSampleAudio:
    def test_nonexistent_sample_returns_404(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/samples/nope/audio")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestSampleMove:
    def test_assigns_sample_to_speaker(
        self, auth_client: TestClient, session_id: str, sample_id: str, speaker_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample_id}/move",
            json={"speaker_id": speaker_id},
        )
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        moved = [s for s in state["samples"] if s["id"] == sample_id][0]
        assert moved["speaker_id"] == speaker_id

    def test_null_speaker_clears_assignment(
        self, auth_client: TestClient, session_id: str, sample_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample_id}/move",
            json={"speaker_id": None},
        )
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        moved = [s for s in state["samples"] if s["id"] == sample_id][0]
        assert moved["speaker_id"] is None

    def test_nonexistent_speaker_returns_404(
        self, auth_client: TestClient, session_id: str, sample_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample_id}/move",
            json={"speaker_id": "nonexistent"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestSampleDelete:
    def test_removes_sample_and_file(
        self, auth_client: TestClient, session_id: str, sample_id: str
    ) -> None:
        service = get_session_service()
        assert service.get_sample_path(session_id, sample_id) is not None

        resp = auth_client.delete(f"/api/session/{session_id}/samples/{sample_id}")
        assert resp.status_code == 200

        assert service.get_sample_path(session_id, sample_id) is None

    def test_nonexistent_sample_returns_404(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/samples/nope")
        assert resp.status_code == 404
