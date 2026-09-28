"""Tests for task routes: extraction, enrollment, transcription triggers."""

import io
import threading
import time
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from meetscribe.web.services.session import get_session_service


def _upload_track(auth_client: TestClient, session_id: str, wav_upload_bytes: bytes) -> None:
    """Helper: upload a WAV track to the session."""
    resp = auth_client.post(
        f"/api/session/{session_id}/tracks",
        files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
    )
    assert resp.status_code == 200, f"Track upload failed: {resp.text}"


class TestExtraction:
    def test_without_tracks_returns_400(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.post(f"/api/session/{session_id}/extract")
        assert resp.status_code == 400
        assert "No tracks" in resp.json()["detail"]

    def test_with_tracks_returns_started(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        resp = auth_client.post(f"/api/session/{session_id}/extract")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestEnrollment:
    def test_without_speakers_returns_400(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.post(f"/api/session/{session_id}/enroll")
        assert resp.status_code == 400
        assert "No speakers" in resp.json()["detail"]

    def test_with_speakers_returns_started(self, auth_client: TestClient, session_id: str) -> None:
        auth_client.post(f"/api/session/{session_id}/speakers", json={"name": "Alice"})
        resp = auth_client.post(f"/api/session/{session_id}/enroll")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestTranscription:
    def test_without_tracks_returns_400(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.post(f"/api/session/{session_id}/transcribe", json={"language": "en"})
        assert resp.status_code == 400
        assert "No tracks" in resp.json()["detail"]

    def test_saves_language_option(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        resp = auth_client.post(f"/api/session/{session_id}/transcribe", json={"language": "en"})
        assert resp.status_code == 200

        state = auth_client.get(f"/api/session/{session_id}").json()
        assert state["language"] == "en"


class TestTranscribingOverlay:
    """The session list shows "transcribing" while the task runs, from the task registry."""

    def _list_status(self, auth_client: TestClient, session_id: str) -> str:
        sessions = auth_client.get("/api/session").json()["sessions"]
        return next(s["status"] for s in sessions if s["id"] == session_id)

    def test_list_shows_transcribing_only_while_running(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        release = threading.Event()

        def fake_transcribe(*args, **kwargs):
            release.wait(timeout=5)
            yield {"transcript": "**[00:00] A:** hi", "segments": []}

        runner = Mock()
        runner.transcribe = fake_transcribe
        with patch("meetscribe.web.routes.tasks.get_pipeline_runner", return_value=runner):
            resp = auth_client.post(
                f"/api/session/{session_id}/transcribe", json={"language": "ru"}
            )
        assert resp.json() == {"status": "started"}

        assert self._list_status(auth_client, session_id) == "transcribing"
        # The persisted status is untouched by the overlay
        assert auth_client.get(f"/api/session/{session_id}").json()["status"] == "uploaded"

        release.set()
        # /tasks/status reports None once done=True, which is set after callbacks
        for _ in range(100):
            tasks = auth_client.get(f"/api/session/{session_id}/tasks/status").json()
            if tasks["transcribe"] is None:
                break
            time.sleep(0.05)
        assert self._list_status(auth_client, session_id) == "transcribed"


class TestTranscriptionResult:
    """What on_complete stores once the transcription task finishes."""

    DROPPED = {
        "track_num": 1,
        "start_ms": 1000,
        "end_ms": 2000,
        "speaker": "Alice",
        "text": "Продолжение следует...",
        "reason": "blocklist",
        "no_speech_prob": 0.0,
        "avg_logprob": -0.3,
    }

    def _transcribe(self, auth_client: TestClient, session_id: str, final: dict) -> dict:
        runner = Mock()
        runner.transcribe = lambda *args, **kwargs: iter([final])
        with patch("meetscribe.web.routes.tasks.get_pipeline_runner", return_value=runner):
            resp = auth_client.post(
                f"/api/session/{session_id}/transcribe", json={"language": "ru"}
            )
        assert resp.json() == {"status": "started"}
        for _ in range(100):
            tasks = auth_client.get(f"/api/session/{session_id}/tasks/status").json()
            if tasks["transcribe"] is None:
                break
            time.sleep(0.05)
        return auth_client.get(f"/api/session/{session_id}").json()

    def test_everything_filtered_out_is_still_saved(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        state = self._transcribe(
            auth_client,
            session_id,
            {"transcript": "", "segments": [], "dropped": [self.DROPPED]},
        )

        assert state["status"] == "transcribed"
        assert state["segments"] == []
        assert [d["text"] for d in state["dropped_segments"]] == ["Продолжение следует..."]

        dropped_id = state["dropped_segments"][0]["id"]
        resp = auth_client.post(f"/api/session/{session_id}/dropped-segments/{dropped_id}/restore")
        assert resp.status_code == 200
        state = auth_client.get(f"/api/session/{session_id}").json()
        assert [s["text"] for s in state["segments"]] == ["Продолжение следует..."]

    def test_rerun_replaces_earlier_result(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        segment = {
            "track_num": 1,
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": "Alice",
            "text": "Привет",
        }
        self._transcribe(
            auth_client,
            session_id,
            {"transcript": "**[00:00] Alice:** Привет", "segments": [segment], "dropped": []},
        )
        state = self._transcribe(
            auth_client, session_id, {"transcript": "", "segments": [], "dropped": []}
        )

        assert state["segments"] == []
        assert not state["transcript"]

    def test_dropped_list_is_not_streamed(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        _upload_track(auth_client, session_id, wav_upload_bytes)
        self._transcribe(
            auth_client,
            session_id,
            {"transcript": "", "segments": [], "dropped": [self.DROPPED]},
        )
        stream = auth_client.get(f"/api/session/{session_id}/transcribe/stream")
        assert "Продолжение следует" not in stream.text


class TestTranscript:
    def test_missing_transcript_returns_404(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.get(f"/api/session/{session_id}/transcript")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"].lower()

    def test_returns_transcript_after_set(self, auth_client: TestClient, session_id: str) -> None:
        get_session_service().set_transcript(session_id, "Hello world")
        resp = auth_client.get(f"/api/session/{session_id}/transcript")
        assert resp.status_code == 200
        assert resp.json()["transcript"] == "Hello world"


class TestSegments:
    def test_session_includes_segments(self, auth_client: TestClient, session_id: str) -> None:
        svc = get_session_service()
        svc.save_segments(
            session_id,
            [
                {"track_num": 1, "start_ms": 0, "end_ms": 5000, "speaker": "Alice", "text": "Hi"},
                {
                    "track_num": 2,
                    "start_ms": 5000,
                    "end_ms": 10000,
                    "speaker": "Bob",
                    "text": "Hey",
                },
            ],
        )

        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["segments"]) == 2
        assert data["segments"][0]["track_num"] == 1
        assert data["segments"][0]["speaker"] == "Alice"
        assert data["segments"][0]["text"] == "Hi"
        assert data["segments"][1]["track_num"] == 2

    def test_session_without_segments_returns_empty(
        self, auth_client: TestClient, session_id: str
    ) -> None:
        resp = auth_client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["segments"] == []
