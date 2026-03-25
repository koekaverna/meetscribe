"""Tests for track routes."""

import io

import pytest
from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service


@pytest.fixture
def _uploaded_track(auth_client: TestClient, session_id: str, wav_upload_bytes: bytes) -> int:
    """Upload a track and return its track_num."""
    resp = auth_client.post(
        f"/api/session/{session_id}/tracks",
        files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
    )
    return resp.json()[0]["track_num"]


class TestUpload:
    def test_wav_stores_file_and_returns_track_info(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("test.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["track_num"] == 1
        assert data[0]["filename"] == "test.wav"

        path = get_session_service().get_track_path(session_id, 1)
        assert path is not None
        assert path.stat().st_size > 0

    def test_unsupported_format_returns_400_and_no_track_stored(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("test.txt", io.BytesIO(b"not audio"), "text/plain"))],
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

        state = auth_client.get(f"/api/session/{session_id}").json()
        assert len(state["tracks"]) == 0

    def test_nonexistent_session_returns_404(self, auth_client: TestClient) -> None:
        resp = auth_client.post(
            "/api/session/nonexistent/tracks",
            files=[("files", ("test.wav", io.BytesIO(b""), "audio/wav"))],
        )
        assert resp.status_code == 404

    def test_sets_session_status_to_uploaded(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        state = auth_client.get(f"/api/session/{session_id}").json()
        assert state["status"] == "uploaded"


class TestList:
    def test_empty_session_returns_empty_list(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/tracks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_uploaded_tracks(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("a.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        tracks = auth_client.get(f"/api/session/{session_id}/tracks").json()
        assert len(tracks) == 1
        assert tracks[0]["filename"] == "a.wav"


class TestAudio:
    def test_returns_wav_content(
        self, auth_client: TestClient, session_id: str, _uploaded_track: int
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/tracks/{_uploaded_track}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_nonexistent_track_returns_404(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/tracks/999/audio")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestUpdate:
    def test_persists_speaker_name_and_diarize_flag(
        self, auth_client: TestClient, session_id: str, _uploaded_track: int
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/{_uploaded_track}",
            params={"speaker_name": "Host", "diarize": False},
        )
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        track = state["tracks"][0]
        assert track["speaker_name"] == "Host"
        assert track["diarize"] is False

    def test_nonexistent_track_returns_404(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/999",
            params={"speaker_name": "Host", "diarize": False},
        )
        assert resp.status_code == 404


class TestDelete:
    def test_removes_file_and_track_from_session(
        self, auth_client: TestClient, session_id: str, _uploaded_track: int
    ) -> None:
        service = get_session_service()
        assert service.get_track_path(session_id, _uploaded_track) is not None

        resp = auth_client.delete(f"/api/session/{session_id}/tracks/{_uploaded_track}")
        assert resp.status_code == 200

        assert service.get_track_path(session_id, _uploaded_track) is None
        state = auth_client.get(f"/api/session/{session_id}").json()
        assert len(state["tracks"]) == 0

    def test_nonexistent_track_returns_404(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/tracks/999")
        assert resp.status_code == 404
