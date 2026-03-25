"""Tests for sample and speaker bin routes."""

from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service


class TestSpeakerBins:
    def test_create_speaker(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/speakers",
            json={"name": "Alice"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert "id" in data

    def test_rename_speaker_persists(self, auth_client: TestClient, session_id: str) -> None:
        speaker = auth_client.post(
            f"/api/session/{session_id}/speakers", json={"name": "Alice"}
        ).json()
        resp = auth_client.patch(
            f"/api/session/{session_id}/speakers/{speaker['id']}",
            json={"name": "Bob"},
        )
        assert resp.status_code == 200

        # Verify rename persisted in session state
        state = auth_client.get(f"/api/session/{session_id}").json()
        speaker_names = [s["name"] for s in state["speakers"]]
        assert "Bob" in speaker_names
        assert "Alice" not in speaker_names

    def test_rename_nonexistent_speaker(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/speakers/nope",
            json={"name": "Bob"},
        )
        assert resp.status_code == 404

    def test_delete_speaker_removes_from_session(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        speaker = auth_client.post(
            f"/api/session/{session_id}/speakers", json={"name": "Alice"}
        ).json()
        resp = auth_client.delete(f"/api/session/{session_id}/speakers/{speaker['id']}")
        assert resp.status_code == 200

        # Verify speaker is gone
        state = auth_client.get(f"/api/session/{session_id}").json()
        assert len(state["speakers"]) == 0

    def test_delete_nonexistent_speaker(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/speakers/nope")
        assert resp.status_code == 404


class TestSamples:
    def test_list_samples_empty(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/samples")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_nonexistent_sample_audio(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/samples/nope/audio")
        assert resp.status_code == 404

    def test_delete_nonexistent_sample(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/samples/nope")
        assert resp.status_code == 404

    def test_move_sample_to_nonexistent_speaker(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        service = get_session_service()
        sample = service.add_sample(
            session_id, track_num=1, cluster_id=0,
            filename="test.wav", duration_ms=1000, content=b"\x00" * 100,
        )
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample.id}/move",
            json={"speaker_id": "nonexistent"},
        )
        assert resp.status_code == 404

    def test_move_sample_persists_assignment(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        service = get_session_service()
        sample = service.add_sample(
            session_id, track_num=1, cluster_id=0,
            filename="test.wav", duration_ms=1000, content=b"\x00" * 100,
        )
        speaker = service.add_speaker(session_id, "Alice")
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample.id}/move",
            json={"speaker_id": speaker.id},
        )
        assert resp.status_code == 200

        # Verify sample is now assigned to speaker
        state = auth_client.get(f"/api/session/{session_id}").json()
        moved = [s for s in state["samples"] if s["id"] == sample.id]
        assert len(moved) == 1
        assert moved[0]["speaker_id"] == speaker.id

    def test_move_sample_unassign(self, auth_client: TestClient, session_id: str) -> None:
        service = get_session_service()
        sample = service.add_sample(
            session_id, track_num=1, cluster_id=0,
            filename="test.wav", duration_ms=1000, content=b"\x00" * 100,
        )
        resp = auth_client.post(
            f"/api/session/{session_id}/samples/{sample.id}/move",
            json={"speaker_id": None},
        )
        assert resp.status_code == 200

    def test_delete_sample_removes_file(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        service = get_session_service()
        sample = service.add_sample(
            session_id, track_num=1, cluster_id=0,
            filename="test.wav", duration_ms=1000, content=b"\x00" * 100,
        )
        # Verify file exists before delete
        path = service.get_sample_path(session_id, sample.id)
        assert path is not None

        resp = auth_client.delete(f"/api/session/{session_id}/samples/{sample.id}")
        assert resp.status_code == 200

        # Verify file is gone
        assert service.get_sample_path(session_id, sample.id) is None
