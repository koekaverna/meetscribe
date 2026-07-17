"""Tests for transcript segment editing routes."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from meetscribe.database import get_db
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


def _segments(session_id: str) -> list[sqlite3.Row]:
    return (
        get_db()
        .execute(
            "SELECT * FROM session_segments WHERE session_id = ? ORDER BY sort_order",
            (session_id,),
        )
        .fetchall()
    )


def _transcript(session_id: str) -> str:
    row = get_db().execute("SELECT transcript FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row["transcript"]


@pytest.fixture
def seeded_session(auth_client: TestClient, session_id: str) -> tuple[str, list[int]]:
    """Transcribed session with three segments; returns (session_id, segment_ids)."""
    _seed_transcript(
        session_id,
        "old transcript",
        [
            (1, 0, 5000, "Alice", "hello there"),
            (1, 5000, 9000, "Bob", "hi"),
            (1, 9000, 15000, "Alice", "bye"),
        ],
    )
    ids = [row["id"] for row in _segments(session_id)]
    return session_id, ids


class TestPatchSegment:
    def test_session_state_exposes_segment_ids(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        segments = auth_client.get(f"/api/session/{sid}").json()["segments"]
        assert [s["id"] for s in segments] == ids

    def test_patch_text_updates_segment_and_transcript(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.patch(f"/api/session/{sid}/segments/{ids[0]}", json={"text": "edited"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "updated"}

        rows = _segments(sid)
        assert rows[0]["text"] == "edited"
        assert rows[0]["speaker"] == "Alice"
        assert _transcript(sid) == (
            "**[00:00] Alice:** edited\n\n**[00:05] Bob:** hi\n\n**[00:09] Alice:** bye"
        )

    def test_patch_speaker_updates_segment_and_transcript(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.patch(
            f"/api/session/{sid}/segments/{ids[1]}", json={"speaker": "Charlie"}
        )
        assert resp.status_code == 200

        rows = _segments(sid)
        assert rows[1]["speaker"] == "Charlie"
        assert rows[1]["text"] == "hi"
        assert _transcript(sid) == (
            "**[00:00] Alice:** hello there\n\n**[00:05] Charlie:** hi\n\n**[00:09] Alice:** bye"
        )

    def test_patch_empty_body_returns_400(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.patch(f"/api/session/{sid}/segments/{ids[0]}", json={})
        assert resp.status_code == 400

    def test_patch_invalid_segment_returns_404(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, _ = seeded_session
        resp = auth_client.patch(f"/api/session/{sid}/segments/999999", json={"text": "x"})
        assert resp.status_code == 404


class TestDeleteSegment:
    def test_delete_renumbers_and_regenerates_transcript(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.delete(f"/api/session/{sid}/segments/{ids[1]}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}

        rows = _segments(sid)
        assert [r["text"] for r in rows] == ["hello there", "bye"]
        assert [r["sort_order"] for r in rows] == [0, 1]
        assert _transcript(sid) == "**[00:00] Alice:** hello there\n\n**[00:09] Alice:** bye"

    def test_delete_invalid_segment_returns_404(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, _ = seeded_session
        resp = auth_client.delete(f"/api/session/{sid}/segments/999999")
        assert resp.status_code == 404


class TestMergeSegment:
    def test_merge_concatenates_text_and_spans_time(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.post(f"/api/session/{sid}/segments/{ids[0]}/merge-next")
        assert resp.status_code == 200
        assert resp.json() == {"status": "merged"}

        rows = _segments(sid)
        assert len(rows) == 2
        assert rows[0]["text"] == "hello there hi"
        assert rows[0]["start_ms"] == 0
        assert rows[0]["end_ms"] == 9000
        assert rows[0]["speaker"] == "Alice"
        assert [r["sort_order"] for r in rows] == [0, 1]
        assert rows[1]["text"] == "bye"
        assert _transcript(sid) == "**[00:00] Alice:** hello there hi\n\n**[00:09] Alice:** bye"

    def test_merge_last_segment_returns_400(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.post(f"/api/session/{sid}/segments/{ids[2]}/merge-next")
        assert resp.status_code == 400

    def test_merge_invalid_segment_returns_404(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, _ = seeded_session
        resp = auth_client.post(f"/api/session/{sid}/segments/999999/merge-next")
        assert resp.status_code == 404


class TestSplitSegment:
    def test_split_divides_time_proportionally(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, ids = seeded_session
        # "hello there": len 11, offset 5 → mid = 0 + 5000 * 5 // 11 = 2272
        resp = auth_client.post(f"/api/session/{sid}/segments/{ids[0]}/split", json={"offset": 5})
        assert resp.status_code == 200
        assert resp.json() == {"status": "split"}

        rows = _segments(sid)
        assert len(rows) == 4
        assert rows[0]["text"] == "hello"
        assert rows[0]["start_ms"] == 0
        assert rows[0]["end_ms"] == 2272
        assert rows[1]["text"] == "there"
        assert rows[1]["start_ms"] == 2272
        assert rows[1]["end_ms"] == 5000
        assert rows[1]["speaker"] == "Alice"
        assert rows[1]["track_num"] == 1
        assert [r["sort_order"] for r in rows] == [0, 1, 2, 3]
        assert _transcript(sid) == (
            "**[00:00] Alice:** hello\n\n**[00:02] Alice:** there\n\n"
            "**[00:05] Bob:** hi\n\n**[00:09] Alice:** bye"
        )

    @pytest.mark.parametrize("offset", [0, 11, 100, -3])
    def test_split_offset_out_of_bounds_returns_400(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]], offset: int
    ) -> None:
        sid, ids = seeded_session
        resp = auth_client.post(
            f"/api/session/{sid}/segments/{ids[0]}/split", json={"offset": offset}
        )
        assert resp.status_code == 400

    def test_split_invalid_segment_returns_404(
        self, auth_client: TestClient, seeded_session: tuple[str, list[int]]
    ) -> None:
        sid, _ = seeded_session
        resp = auth_client.post(f"/api/session/{sid}/segments/999999/split", json={"offset": 1})
        assert resp.status_code == 404


class TestAccessControl:
    @pytest.mark.parametrize(
        ("method", "suffix", "body"),
        [
            ("PATCH", "", {"text": "x"}),
            ("DELETE", "", None),
            ("POST", "/merge-next", None),
            ("POST", "/split", {"offset": 1}),
        ],
    )
    def test_non_transcribed_session_returns_409(
        self,
        auth_client: TestClient,
        session_id: str,
        method: str,
        suffix: str,
        body: dict | None,
    ) -> None:
        resp = auth_client.request(
            method, f"/api/session/{session_id}/segments/1{suffix}", json=body
        )
        assert resp.status_code == 409

    def test_non_creator_returns_404(
        self,
        auth_client: TestClient,
        seeded_session: tuple[str, list[int]],
        web_auth_service: AuthService,
    ) -> None:
        sid, ids = seeded_session
        _, other_token = web_auth_service.register("other", "test-pass-000", "default")
        my_token = auth_client.cookies["meetscribe_session"]
        auth_client.cookies.set("meetscribe_session", other_token)
        resp = auth_client.patch(f"/api/session/{sid}/segments/{ids[0]}", json={"text": "x"})
        auth_client.cookies.set("meetscribe_session", my_token)
        assert resp.status_code == 404
        assert _segments(sid)[0]["text"] == "hello there"

    def test_admin_can_edit_others_session(
        self,
        auth_client: TestClient,
        seeded_session: tuple[str, list[int]],
        admin_user: tuple[AuthUser, str],
    ) -> None:
        sid, ids = seeded_session
        _, admin_token = admin_user
        auth_client.cookies.set("meetscribe_session", admin_token)
        resp = auth_client.patch(f"/api/session/{sid}/segments/{ids[0]}", json={"text": "edited"})
        assert resp.status_code == 200
        assert _segments(sid)[0]["text"] == "edited"
