"""Tests for track routes."""

import asyncio
import gzip
import io

import pytest
from fastapi.testclient import TestClient

from meetscribe.web.app import GunzipRequests
from meetscribe.web.services.session import get_session_service


@pytest.fixture
def _uploaded_track(auth_client: TestClient, session_id: str, wav_upload_bytes: bytes) -> int:
    """Upload a track and return its track_num."""
    resp = auth_client.post(
        f"/api/session/{session_id}/tracks",
        files=[("files", ("t.wav", io.BytesIO(wav_upload_bytes), "audio/wav"))],
    )
    assert resp.status_code == 200, f"Track upload failed: {resp.text}"
    data = resp.json()
    assert len(data) >= 1, "Expected at least one track in response"
    return data[0]["track_num"]


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

    def test_nonexistent_track_returns_404(self, auth_client: TestClient, session_id: str) -> None:
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

    def test_persists_open_space_for_named_track(
        self, auth_client: TestClient, session_id: str, _uploaded_track: int
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/{_uploaded_track}",
            params={"speaker_name": "Host", "diarize": False, "open_space": True},
        )
        assert resp.status_code == 200

        track = auth_client.get(f"/api/session/{session_id}").json()["tracks"][0]
        assert track["open_space"] is True

    def test_open_space_ignored_when_diarized(
        self, auth_client: TestClient, session_id: str, _uploaded_track: int
    ) -> None:
        resp = auth_client.patch(
            f"/api/session/{session_id}/tracks/{_uploaded_track}",
            params={"diarize": True, "open_space": True},
        )
        assert resp.status_code == 200

        track = auth_client.get(f"/api/session/{session_id}").json()["tracks"][0]
        assert track["open_space"] is False

    def test_nonexistent_track_returns_404(self, auth_client: TestClient, session_id: str) -> None:
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

    def test_nonexistent_track_returns_404(self, auth_client: TestClient, session_id: str) -> None:
        resp = auth_client.delete(f"/api/session/{session_id}/tracks/999")
        assert resp.status_code == 404


def _multipart_wav(name: str, payload: bytes) -> tuple[bytes, str]:
    boundary = "meetscribe-test-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


class TestGzipUpload:
    def test_gzipped_body_is_stored_byte_for_byte(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        body, content_type = _multipart_wav("test.wav", wav_upload_bytes)
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            content=gzip.compress(body),
            headers={"Content-Type": content_type, "Content-Encoding": "gzip"},
        )
        assert resp.status_code == 200, resp.text
        path = get_session_service().get_track_path(session_id, 1)
        assert path is not None
        assert path.read_bytes() == wav_upload_bytes

    def test_corrupt_gzip_returns_400(self, auth_client: TestClient, session_id: str) -> None:
        body, content_type = _multipart_wav("test.wav", b"x")
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            content=b"definitely not gzip",
            headers={"Content-Type": content_type, "Content-Encoding": "gzip"},
        )
        assert resp.status_code == 400

    def test_truncated_gzip_returns_400(
        self, auth_client: TestClient, session_id: str, wav_upload_bytes: bytes
    ) -> None:
        body, content_type = _multipart_wav("test.wav", wav_upload_bytes)
        resp = auth_client.post(
            f"/api/session/{session_id}/tracks",
            content=gzip.compress(body)[:-8],
            headers={"Content-Type": content_type, "Content-Encoding": "gzip"},
        )
        assert resp.status_code == 400

    def test_chunked_body_is_reassembled_without_content_length(self) -> None:
        payload = bytes(range(256)) * 1000
        gz = gzip.compress(payload)
        chunks = [gz[:10], gz[10:100], gz[100:]]
        messages = [
            {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
            for i, chunk in enumerate(chunks)
        ]
        seen: list[bytes] = []
        inner_headers: list[tuple[bytes, bytes]] = []

        async def inner_app(scope, receive, send) -> None:
            inner_headers.extend(scope["headers"])
            while True:
                message = await receive()
                seen.append(message["body"])
                if not message["more_body"]:
                    return

        async def receive():
            return messages.pop(0)

        scope = {
            "type": "http",
            "headers": [(b"content-encoding", b"gzip"), (b"content-length", str(len(gz)).encode())],
        }
        asyncio.run(GunzipRequests(inner_app)(scope, receive, None))  # type: ignore[arg-type]

        assert b"".join(seen) == payload
        assert dict(inner_headers) == {}
