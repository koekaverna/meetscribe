"""Tests for task routes: extraction, enrollment, transcription triggers."""

import io

import pytest
from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service

from .conftest_web import *  # noqa: F401, F403 — import web fixtures


@pytest.fixture
def session_id(auth_client: TestClient) -> str:
    return auth_client.post("/api/session").json()["session_id"]


class TestExtraction:
    def test_start_extraction_no_tracks(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(f"/api/session/{session_id}/extract")
        assert resp.status_code == 400
        assert "No tracks" in resp.json()["detail"]

    def test_start_extraction_with_tracks(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        # Upload a track first
        auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        resp = auth_client.post(f"/api/session/{session_id}/extract")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


@pytest.fixture
def wav_upload_bytes() -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


class TestEnrollment:
    def test_start_enrollment_no_speakers(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(f"/api/session/{session_id}/enroll")
        assert resp.status_code == 400
        assert "No speakers" in resp.json()["detail"]

    def test_start_enrollment_with_speakers(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        # Create a speaker first
        auth_client.post(
            f"/api/session/{session_id}/speakers", json={"name": "Alice"}
        )
        resp = auth_client.post(f"/api/session/{session_id}/enroll")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestTranscription:
    def test_start_transcription_no_tracks(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.post(
            f"/api/session/{session_id}/transcribe", json={"language": "en"}
        )
        assert resp.status_code == 400

    def test_start_transcription_with_tracks(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        auth_client.post(
            f"/api/session/{session_id}/tracks",
            files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
        )
        resp = auth_client.post(
            f"/api/session/{session_id}/transcribe", json={"language": "en"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestTranscript:
    def test_get_transcript_not_available(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/transcript")
        assert resp.status_code == 404

    def test_get_transcript_after_set(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        service = get_session_service()
        service.set_transcript(session_id, "Hello world transcript")
        resp = auth_client.get(f"/api/session/{session_id}/transcript")
        assert resp.status_code == 200
        assert resp.json()["transcript"] == "Hello world transcript"
