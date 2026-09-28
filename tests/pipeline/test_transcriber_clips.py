"""Tests for pipeline/transcriber.py — track + clips transcription mode."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from meetscribe.errors import ClipsEndpointUnavailable, SpeachesAPIError
from meetscribe.pipeline.models import SpeechSegment
from tests.conftest import make_wav_file
from tests.pipeline.test_transcriber_chunks import make_transcriber

CLIPS_URL = "http://a:8000/v1/audio/transcriptions/clips"
CHUNKS_URL = "http://a:8000/v1/audio/transcriptions"
OPENAPI_WITH_CLIPS = {"paths": {"/v1/audio/transcriptions/clips": {}}}

DIARIZED = [
    SpeechSegment(1000, 3000, "Alice"),
    SpeechSegment(4000, 6000, "Bob"),
    SpeechSegment(7000, 9000, "Alice"),
]


def _response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(status_code=status_code, text=str(payload.get("detail", "")))
    resp.json.return_value = payload
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            str(status_code), request=MagicMock(), response=resp
        )
    return resp


def _seg(clip_index: int, start: float, end: float, text: str) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "clip_index": clip_index,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
    }


def _run(tmp_path: Path, posts: list, openapi: dict = OPENAPI_WITH_CLIPS, **overrides):
    audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
    t = make_transcriber(segment_mode="clips", max_inflight=1, **overrides)
    progress = []
    with (
        patch("meetscribe.pipeline.transcriber.httpx.get", return_value=_response(openapi)),
        patch("meetscribe.pipeline.transcriber.httpx.post", side_effect=posts) as post,
        patch("tenacity.nap.time.sleep"),
    ):
        result = t.transcribe_segments(
            audio, DIARIZED, progress_callback=lambda done, total: progress.append((done, total))
        )
    return result, post, progress


class TestClipsRequest:
    def test_sends_track_once_with_clips(self, tmp_path: Path):
        payload = {
            "segments": [
                _seg(0, 1.2, 2.8, "Привет"),
                _seg(2, 7.1, 8.9, "Пока"),
                _seg(1, 4.1, 5.0, "Угу"),
            ],
            "failed_clips": [],
        }
        result, post, progress = _run(tmp_path, [_response(payload)], chunk_padding_ms=200)

        assert post.call_count == 1
        kwargs = post.call_args[1]
        assert post.call_args[0][0] == CLIPS_URL
        assert kwargs["files"]["file"][0] == "test.wav"
        assert json.loads(kwargs["data"]["clips"]) == [
            {"start": 1.0, "end": 3.0, "speaker": "Alice"},
            {"start": 4.0, "end": 6.0, "speaker": "Bob"},
            {"start": 7.0, "end": 9.0, "speaker": "Alice"},
        ]
        assert kwargs["data"]["model"] == "m"
        assert kwargs["data"]["language"] == "ru"
        # The server adds the padding itself
        assert kwargs["data"]["pad_ms"] == "200"
        assert kwargs["timeout"] == 600.0

        assert [(s.start_ms, s.end_ms, s.speaker, s.text) for s in result.segments] == [
            (1200, 2800, "Alice", "Привет"),
            (4100, 5000, "Bob", "Угу"),
            (7100, 8900, "Alice", "Пока"),
        ]
        assert result.failed_chunks == []
        assert progress == [(6000, 6000)]

    def test_filter_and_clamp_apply(self, tmp_path: Path):
        payload = {
            "segments": [
                _seg(0, 0.9, 3.1, "Привет"),
                _seg(1, 4.0, 6.0, "Продолжение следует..."),
            ],
        }
        result, _, _ = _run(tmp_path, [_response(payload)])

        assert [(s.start_ms, s.end_ms, s.speaker) for s in result.segments] == [
            (1000, 3000, "Alice")
        ]
        assert [(s.text, s.drop_reason, s.speaker) for s in result.dropped] == [
            ("Продолжение следует...", "blocklist", "Bob")
        ]

    def test_speaker_omitted_when_unknown(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        t = make_transcriber(segment_mode="clips")
        with (
            patch(
                "meetscribe.pipeline.transcriber.httpx.get",
                return_value=_response(OPENAPI_WITH_CLIPS),
            ),
            patch(
                "meetscribe.pipeline.transcriber.httpx.post",
                return_value=_response({"segments": [_seg(0, 1.0, 2.0, "Привет")]}),
            ) as post,
        ):
            t.transcribe_segments(audio, [SpeechSegment(1000, 3000, None)])

        assert json.loads(post.call_args[1]["data"]["clips"]) == [{"start": 1.0, "end": 3.0}]

    def test_clips_never_exceed_server_limit(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=100.0)
        t = make_transcriber(
            segment_mode="clips", max_chunk_ms=60000, min_chunk_ms=1000, max_gap_ms=2000
        )
        diarized = [
            SpeechSegment(0, 70000, "Alice"),
            SpeechSegment(70500, 71000, "Alice"),
            SpeechSegment(72000, 99000, "Bob"),
        ]
        with (
            patch(
                "meetscribe.pipeline.transcriber.httpx.get",
                return_value=_response(OPENAPI_WITH_CLIPS),
            ),
            patch(
                "meetscribe.pipeline.transcriber.httpx.post",
                return_value=_response({"segments": [_seg(0, 1.0, 2.0, "Привет")]}),
            ) as post,
        ):
            t.transcribe_segments(audio, diarized)

        clips = json.loads(post.call_args[1]["data"]["clips"])
        assert all(clip["end"] - clip["start"] <= 30.0 for clip in clips)
        assert clips[0]["start"] == 0.0
        assert clips[-1]["end"] == 99.0


class TestClipsFailures:
    def test_failed_clips_are_retried_once(self, tmp_path: Path):
        first = {
            "segments": [_seg(0, 1.0, 2.0, "Привет")],
            "failed_clips": [
                {"clip_index": 1, "reason": "error: RuntimeError"},
                {"clip_index": 2, "reason": "error: RuntimeError"},
            ],
        }
        second = {
            "segments": [_seg(1, 7.0, 8.0, "Пока")],
            "failed_clips": [{"clip_index": 0, "reason": "error: RuntimeError"}],
        }
        result, post, _ = _run(tmp_path, [_response(first), _response(second)])

        assert post.call_count == 2
        assert json.loads(post.call_args[1]["data"]["clips"]) == [
            {"start": 4.0, "end": 6.0, "speaker": "Bob"},
            {"start": 7.0, "end": 9.0, "speaker": "Alice"},
        ]
        assert [s.text for s in result.segments] == ["Привет", "Пока"]
        assert [(c.start_ms, c.speaker) for c in result.failed_chunks] == [(4000, "Bob")]

    def test_clip_empty_on_retry_is_not_a_failure(self, tmp_path: Path):
        first = {
            "segments": [_seg(0, 1.0, 2.0, "Привет")],
            "failed_clips": [{"clip_index": 1, "reason": "error: RuntimeError"}],
        }
        result, _, _ = _run(tmp_path, [_response(first), _response({"segments": []})])

        assert [s.text for s in result.segments] == ["Привет"]
        assert result.failed_chunks == []

    def test_failed_retry_request_keeps_first_result(self, tmp_path: Path):
        first = {
            "segments": [_seg(0, 1.0, 2.0, "Привет")],
            "failed_clips": [{"clip_index": 1, "reason": "error: RuntimeError"}],
        }
        result, _, _ = _run(
            tmp_path, [_response(first), _response({"detail": "bad"}, status_code=422)]
        )

        assert [s.text for s in result.segments] == ["Привет"]
        assert [c.start_ms for c in result.failed_chunks] == [4000]

    def test_clip_without_speech_is_not_a_failure(self, tmp_path: Path):
        payload = {
            "segments": [_seg(0, 1.0, 2.0, "Привет")],
            "failed_clips": [
                {"clip_index": 1, "reason": "no_speech"},
                {"clip_index": 2, "reason": "empty"},
            ],
        }
        result, post, _ = _run(tmp_path, [_response(payload)])

        assert post.call_count == 1
        assert result.failed_chunks == []

    def test_out_of_range_clip_is_reported_without_retry(self, tmp_path: Path):
        payload = {
            "segments": [_seg(0, 1.0, 2.0, "Привет")],
            "failed_clips": [{"clip_index": 2, "reason": "out_of_range"}],
        }
        result, post, _ = _run(tmp_path, [_response(payload)])

        assert post.call_count == 1
        assert [c.start_ms for c in result.failed_chunks] == [7000]

    def test_nothing_transcribed_raises(self, tmp_path: Path):
        payload = {
            "segments": [],
            "failed_clips": [{"clip_index": i, "reason": "error: RuntimeError"} for i in range(3)],
        }
        with pytest.raises(SpeachesAPIError, match="No clip could be transcribed"):
            _run(tmp_path, [_response(payload), _response(payload)])

    def test_transient_error_retries_whole_track(self, tmp_path: Path):
        payload = {"segments": [_seg(0, 1.0, 2.0, "Привет")]}
        result, post, _ = _run(
            tmp_path, [_response({"detail": "closed"}, status_code=499), _response(payload)]
        )

        assert post.call_count == 2
        assert [s.text for s in result.segments] == ["Привет"]

    def test_request_error_fails_the_track(self, tmp_path: Path):
        with pytest.raises(SpeachesAPIError, match="422"):
            _run(tmp_path, [_response({"detail": "clip too long"}, status_code=422)])


class TestClipsFallback:
    CHUNK = {"segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}]}

    def test_old_server_falls_back_to_chunks(self, tmp_path: Path):
        result, post, progress = _run(
            tmp_path, [_response(self.CHUNK) for _ in range(3)], openapi={"paths": {}}
        )

        assert [call[0][0] for call in post.call_args_list] == [CHUNKS_URL] * 3
        assert [s.start_ms for s in result.segments] == [1000, 4000, 7000]
        assert len(progress) == 3

    def test_404_on_endpoint_falls_back_to_chunks(self, tmp_path: Path):
        posts = [_response({"detail": "Not Found"}, status_code=404)]
        posts += [_response(self.CHUNK) for _ in range(3)]
        result, post, _ = _run(tmp_path, posts)

        assert [call[0][0] for call in post.call_args_list] == [CLIPS_URL] + [CHUNKS_URL] * 3
        assert len(result.segments) == 3

    def test_model_not_enabled_is_an_error_not_a_fallback(self, tmp_path: Path):
        detail = "Model 'turbo' is not enabled on this server"
        with pytest.raises(SpeachesAPIError, match="404") as exc:
            _run(tmp_path, [_response({"detail": detail}, status_code=404)])
        assert not isinstance(exc.value, ClipsEndpointUnavailable)

    def test_unreachable_openapi_falls_back_without_caching(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        t = make_transcriber(segment_mode="clips", max_inflight=1)
        clips = {"segments": [_seg(0, 1.0, 2.0, "Привет")]}
        with (
            patch(
                "meetscribe.pipeline.transcriber.httpx.get",
                side_effect=[httpx.ConnectError("down"), _response(OPENAPI_WITH_CLIPS)],
            ),
            patch(
                "meetscribe.pipeline.transcriber.httpx.post",
                side_effect=[_response(self.CHUNK) for _ in range(3)] + [_response(clips)],
            ) as post,
        ):
            t.transcribe_segments(audio, DIARIZED)
            t.transcribe_segments(audio, DIARIZED)

        urls = [call[0][0] for call in post.call_args_list]
        assert urls == [CHUNKS_URL] * 3 + [CLIPS_URL]

    def test_endpoint_checked_once(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        payload = {"segments": [_seg(0, 1.0, 2.0, "Привет")]}
        t = make_transcriber(segment_mode="clips")
        with (
            patch(
                "meetscribe.pipeline.transcriber.httpx.get",
                return_value=_response(OPENAPI_WITH_CLIPS),
            ) as get,
            patch(
                "meetscribe.pipeline.transcriber.httpx.post",
                side_effect=[_response(payload), _response(payload)],
            ),
        ):
            t.transcribe_segments(audio, DIARIZED)
            t.transcribe_segments(audio, DIARIZED)

        assert get.call_count == 1

    def test_chunks_mode_never_touches_clips(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        t = make_transcriber(max_inflight=1)
        with (
            patch("meetscribe.pipeline.transcriber.httpx.get") as get,
            patch(
                "meetscribe.pipeline.transcriber.httpx.post",
                side_effect=[_response(self.CHUNK) for _ in range(3)],
            ) as post,
        ):
            t.transcribe_segments(audio, DIARIZED)

        get.assert_not_called()
        assert [call[0][0] for call in post.call_args_list] == [CHUNKS_URL] * 3
