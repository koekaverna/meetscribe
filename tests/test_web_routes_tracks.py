"""Tests for track routes."""

import io

import pytest
from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service

from .conftest_web import *  # noqa: F401, F403 — import web fixtures


@pytest.fixture
def session_id(auth_client: TestClient) -> str:
    return auth_client.post("/api/session").json()["session_id"]


@pytest.fixture
def wav_upload_bytes() -> bytes:
    """Minimal valid WAV for upload tests."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


class TestTrackUpload:
    def test_upload_wav(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("test.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["track_num"] == 1
        assert data[0]["filename"] == "test.wav"

    def test_upload_unsupported_format(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("test.txt", io.BytesIO(b"not audio"), "text/plain"))],
        )
        assert resp.status_code == 400

    def test_upload_to_nonexistent_session(self, auth_client: TestClient) -> None:
        resp = auth_client.post(
            "/api/session/nonexistent/tracks",
            files=[("files", ("test.wav", io.BytesIO(b""), "audio/wav"))],
        )
        assert resp.status_code == 404


class TestTrackList:
    def test_list_tracks_empty(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/tracks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_tracks_after_upload(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("a.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        resp = auth_client.get(f"/api/session/{session_id}/tracks")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestTrackOperations:
    def _upload_track(self, client, session_id, wav_bytes) -> int:
        resp = client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("t.wav", io.BytesIO(wav_bytes), "audio/wav"))],
        )
        return resp.json()[0]["track_num"]

    def test_get_track_audio(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        track_num = self._upload_track(auth_client, session_id, wav_upload_bytes)
        resp = auth_client.get(f"/api/session/{session_id}/tracks/{track_num}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_get_nonexistent_track_audio(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/tracks/999/audio")
        assert resp.status_code == 404

    def test_update_track(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        track_num = self._upload_track(auth_client, session_id, wav_upload_bytes)
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/{track_num}",
            params={"speaker_name": "Host", "diarize": False},
        )
        assert resp.status_code == 200

    def test_update_nonexistent_track(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/999",
            params={"speaker_name": "Host", "diarize": False},
        )
        assert resp.status_code == 404

    def test_delete_track(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        track_num = self._upload_track(auth_client, session_id, wav_upload_bytes)
        resp = auth_client.delete(f"/api/session/{session_id}/tracks/{track_num}")
        assert resp.status_code == 200

    def test_delete_nonexistent_track(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/tracks/999")
        assert resp.status_code == 404
