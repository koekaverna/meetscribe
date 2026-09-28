"""Tests for pipeline/transcriber.py — retries, chunk failures, boundaries, duplicates."""

import io
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from meetscribe.config import TranscriptionConfig
from meetscribe.errors import ConfigurationError, SpeachesAPIError
from meetscribe.pipeline.models import SpeechSegment, TranscriptionResult, TranscriptSegment
from meetscribe.pipeline.transcriber import (
    RemoteTranscriber,
    Transcriber,
    failed_chunk_placeholders,
    find_longest_pause,
)
from tests.conftest import make_wav_file


def make_transcriber(**overrides) -> Transcriber:
    params = {
        "server_urls": ["http://a:8000"],
        "language": "ru",
        "timeout": 10.0,
        "model": "m",
        "max_gap_ms": 500,
        "max_chunk_ms": 30000,
        "no_speech_prob_threshold": 0.5,
        "avg_logprob_threshold": -0.25,
    }
    return Transcriber(**{**params, **overrides})


def _wav_ms(audio_bytes: bytes) -> int:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        return wf.getnframes() * 1000 // wf.getframerate()


def _api_error(status_code: int | None = 500) -> SpeachesAPIError:
    return SpeachesAPIError("boom", status_code=status_code, endpoint="http://a:8000")


class TestRetryPolicy:
    @pytest.mark.parametrize("status_code", [499, 429, 500, 503, None])
    def test_transient(self, status_code):
        assert _api_error(status_code).is_transient

    @pytest.mark.parametrize("status_code", [400, 404, 422])
    def test_not_transient(self, status_code):
        assert not _api_error(status_code).is_transient

    def test_499_is_retried(self):
        rt = RemoteTranscriber(
            "http://host:8000",
            timeout=10.0,
            model="m",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        dropped = MagicMock(status_code=499, text="client closed request")
        dropped.raise_for_status.side_effect = httpx.HTTPStatusError(
            "499", request=MagicMock(), response=dropped
        )
        ok = MagicMock()
        ok.json.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}]}

        with (
            patch("meetscribe.pipeline.transcriber.httpx.post", side_effect=[dropped, ok]) as m,
            patch("tenacity.nap.time.sleep"),
        ):
            result = rt.transcribe_bytes(b"data", "ru")

        assert m.call_count == 2
        assert [s.text for s in result] == ["Привет"]


class TestChunkFailures:
    def test_failed_chunk_does_not_abort(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        diarized = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(3000, 5000, "Bob"),
            SpeechSegment(6000, 8000, "Alice"),
        ]
        client = MagicMock()
        client.transcribe_bytes.side_effect = [
            [TranscriptSegment(start_ms=0, end_ms=500, text="Hi")],
            _api_error(),
            [TranscriptSegment(start_ms=0, end_ms=500, text="Hi")],
        ]
        t = make_transcriber(max_inflight=1)
        t.clients = [client]
        progress = []

        result = t.transcribe_segments(
            audio, diarized, progress_callback=lambda done, total: progress.append((done, total))
        )

        assert [(s.start_ms, s.speaker) for s in result.segments] == [(0, "Alice"), (6000, "Alice")]
        assert [(c.start_ms, c.end_ms, c.speaker) for c in result.failed_chunks] == [
            (3000, 5000, "Bob")
        ]
        # The failed chunk still counts towards progress, so the bar reaches 100%
        assert progress[-1] == (6000, 6000)

    def test_all_chunks_failed_raises(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        diarized = [SpeechSegment(0, 2000, "Alice"), SpeechSegment(3000, 5000, "Bob")]
        client = MagicMock()
        client.transcribe_bytes.side_effect = _api_error()
        t = make_transcriber()
        t.clients = [client]

        with pytest.raises(SpeachesAPIError):
            t.transcribe_segments(audio, diarized)

    def test_dead_server_aborts_early(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=30.0)
        diarized = [SpeechSegment(i * 2000, i * 2000 + 1000, f"S{i}") for i in range(10)]

        def unreachable(*_args, **_kwargs):
            # A real request takes time, which is what lets queued chunks be cancelled
            time.sleep(0.05)
            raise _api_error(None)

        client = MagicMock()
        client.transcribe_bytes.side_effect = unreachable
        t = make_transcriber(max_inflight=1)
        t.clients = [client]

        with pytest.raises(SpeachesAPIError):
            t.transcribe_segments(audio, diarized)

        assert client.transcribe_bytes.call_count < len(diarized)

    def test_unexpected_error_still_aborts(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        diarized = [SpeechSegment(0, 2000, "Alice"), SpeechSegment(3000, 5000, "Bob")]
        client = MagicMock()
        client.transcribe_bytes.side_effect = [
            [TranscriptSegment(start_ms=0, end_ms=500, text="Hi")],
            KeyError("start"),
        ]
        t = make_transcriber(max_inflight=1)
        t.clients = [client]

        with pytest.raises(KeyError):
            t.transcribe_segments(audio, diarized)

    def test_failed_chunk_placeholders(self):
        result = TranscriptionResult(failed_chunks=[SpeechSegment(3000, 5000, "Bob")])
        placeholders = failed_chunk_placeholders(result)
        assert [(p.start_ms, p.end_ms, p.speaker, p.text, p.drop_reason) for p in placeholders] == [
            (3000, 5000, "Bob", "", "failed")
        ]


class TestChunkBoundaries:
    def _run(self, tmp_path: Path, diarized, returned, **overrides):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=10.0)
        sent = []

        def transcribe_bytes(audio_bytes, language, filename="chunk.wav"):
            sent.append(_wav_ms(audio_bytes))
            return [TranscriptSegment(start_ms=s, end_ms=e, text=txt) for s, e, txt in returned]

        client = MagicMock()
        client.transcribe_bytes = transcribe_bytes
        t = make_transcriber(max_inflight=1, **overrides)
        t.clients = [client]
        return t.transcribe_segments(audio, diarized), sent

    def test_audio_is_padded(self, tmp_path: Path):
        _, sent = self._run(
            tmp_path, [SpeechSegment(1000, 3000, "Alice")], [], chunk_padding_ms=200
        )
        assert sent == [2400]

    def test_timestamps_account_for_padding(self, tmp_path: Path):
        result, _ = self._run(
            tmp_path,
            [SpeechSegment(1000, 3000, "Alice")],
            [(700, 1200, "Привет")],
            chunk_padding_ms=200,
        )
        # Audio starts at 800 ms, so 700 ms inside it is 1500 ms of the track
        assert [(s.start_ms, s.end_ms) for s in result.segments] == [(1500, 2000)]

    def test_text_from_padding_stays_with_chunk_speaker(self, tmp_path: Path):
        """A word heard in the padding must not go to the neighbouring speaker."""
        result, _ = self._run(
            tmp_path,
            [SpeechSegment(1000, 3000, "Alice"), SpeechSegment(3300, 6000, "Bob")],
            [(0, 100, "Да")],
            chunk_padding_ms=200,
        )
        assert [(s.start_ms, s.end_ms, s.speaker) for s in result.segments] == [
            (1000, 1000, "Alice"),
            (3300, 3300, "Bob"),
        ]

    def test_timestamps_clamped_to_chunk(self, tmp_path: Path):
        result, _ = self._run(
            tmp_path, [SpeechSegment(1000, 3000, "Alice")], [(1500, 9000, "Привет")]
        )
        assert [(s.start_ms, s.end_ms) for s in result.segments] == [(2500, 3000)]

    def test_short_chunk_is_absorbed(self, tmp_path: Path):
        _, sent = self._run(
            tmp_path,
            [SpeechSegment(0, 2900, "Alice"), SpeechSegment(3000, 3400, "Alice")],
            [],
            max_chunk_ms=3000,
            min_chunk_ms=1000,
        )
        assert sent == [3400]

    def test_long_segment_is_split(self, tmp_path: Path):
        _, sent = self._run(tmp_path, [SpeechSegment(0, 8000, "Alice")], [], max_chunk_ms=5000)
        assert len(sent) == 2
        assert sum(sent) == 8000
        assert max(sent) <= 5000

    def test_overlap_is_not_sent_twice(self, tmp_path: Path):
        _, sent = self._run(
            tmp_path, [SpeechSegment(0, 5000, "Alice"), SpeechSegment(4000, 7000, "Bob")], []
        )
        assert sent == [5000, 2000]


class TestFindLongestPause:
    def _pcm(self, spans: list[tuple[int, int]], rate: int = 16000) -> bytes:
        """PCM made of (duration_ms, amplitude) spans."""
        out = bytearray()
        for duration_ms, amplitude in spans:
            sample = int(amplitude).to_bytes(2, "little", signed=True)
            out += sample * (rate * duration_ms // 1000)
        return bytes(out)

    def test_finds_middle_of_longest_pause(self):
        raw = self._pcm([(1000, 8000), (200, 0), (1000, 8000), (600, 0), (1200, 8000)])
        # Pauses: 1000–1200 ms and 2200–2800 ms
        assert find_longest_pause(raw, 16000, 2, 1, 0, 4000) == 2500

    def test_searches_only_inside_window(self):
        raw = self._pcm([(1000, 8000), (200, 0), (1000, 8000), (600, 0), (1200, 8000)])
        assert find_longest_pause(raw, 16000, 2, 1, 500, 2000) == 1100

    def test_quiet_noise_counts_as_pause(self):
        raw = self._pcm([(1000, 8000), (400, 100), (1000, 8000)])
        assert find_longest_pause(raw, 16000, 2, 1, 0, 2400) == 1200

    def test_no_pause_picks_quietest_frame(self):
        raw = self._pcm([(1000, 8000), (20, 4000), (1000, 8000)])
        assert find_longest_pause(raw, 16000, 2, 1, 0, 2020) == 1010

    def test_silence_returns_window_middle(self):
        raw = self._pcm([(4000, 0)])
        assert find_longest_pause(raw, 16000, 2, 1, 1000, 3000) == 2000

    def test_unsupported_sample_width_returns_midpoint(self):
        assert find_longest_pause(b"\x00" * 64000, 16000, 1, 1, 1000, 3000) == 2000


class TestJointDuplicates:
    PHRASE = "давайте перейдём к следующему вопросу"

    def _run(self, tmp_path: Path, diarized, texts, **overrides):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=20.0)
        client = MagicMock()
        client.transcribe_bytes.side_effect = [
            [TranscriptSegment(start_ms=0, end_ms=1000, text=text)] for text in texts
        ]
        t = make_transcriber(**{"max_inflight": 1, "dedup_similarity": 0.9, **overrides})
        t.clients = [client]
        return t.transcribe_segments(audio, diarized)

    def test_repeat_at_joint_is_dropped(self, tmp_path: Path):
        result = self._run(
            tmp_path,
            [SpeechSegment(0, 5000, "Alice"), SpeechSegment(5200, 9000, "Bob")],
            [self.PHRASE, "Давайте перейдём к следующему вопросу."],
        )
        assert [(s.text, s.speaker) for s in result.segments] == [(self.PHRASE, "Alice")]
        assert [(s.drop_reason, s.speaker) for s in result.dropped] == [("duplicate", "Bob")]

    def test_short_phrase_is_kept(self, tmp_path: Path):
        result = self._run(
            tmp_path,
            [SpeechSegment(0, 5000, "Alice"), SpeechSegment(5200, 9000, "Bob")],
            ["да да да", "да да да"],
        )
        assert len(result.segments) == 2

    def test_different_phrase_is_kept(self, tmp_path: Path):
        result = self._run(
            tmp_path,
            [SpeechSegment(0, 5000, "Alice"), SpeechSegment(5200, 9000, "Bob")],
            [self.PHRASE, "давайте вернёмся к прошлому вопросу"],
        )
        assert len(result.segments) == 2

    def test_distant_chunks_are_not_compared(self, tmp_path: Path):
        result = self._run(
            tmp_path,
            [SpeechSegment(0, 5000, "Alice"), SpeechSegment(9000, 12000, "Bob")],
            [self.PHRASE, self.PHRASE],
        )
        assert len(result.segments) == 2

    def test_disabled_with_zero_similarity(self, tmp_path: Path):
        result = self._run(
            tmp_path,
            [SpeechSegment(0, 5000, "Alice"), SpeechSegment(5200, 9000, "Bob")],
            [self.PHRASE, self.PHRASE],
            dedup_similarity=0.0,
        )
        assert len(result.segments) == 2


class TestFromConfig:
    def test_passes_all_settings(self):
        cfg = TranscriptionConfig(
            servers=["gpu1"],
            model="turbo",
            timeout=33.0,
            max_gap_ms=700,
            max_chunk_ms=20000,
            no_speech_prob_threshold=0.6,
            avg_logprob_threshold=-0.3,
            legacy_filter_max_words=4,
            hallucination_logprob_threshold=-1.2,
            hallucination_phrases=["Тест"],
            hallucination_phrase_max_extra_words=2,
            chunk_padding_ms=150,
            min_chunk_ms=900,
            dedup_similarity=0.8,
            dedup_min_words=5,
            segment_mode="clips",
            max_inflight=7,
        )
        t = Transcriber.from_config(cfg, ["http://a:8000", "http://b:8000"], "en")

        assert t.language == "en"
        assert (t.max_gap_ms, t.max_chunk_ms) == (700, 20000)
        assert (t.chunk_padding_ms, t.min_chunk_ms) == (150, 900)
        assert (t.dedup_similarity, t.dedup_min_words) == (0.8, 5)
        assert t.segment_mode == "clips"
        assert t.max_inflight == 7
        assert [c.server_url for c in t.clients] == ["http://a:8000", "http://b:8000"]
        for client in t.clients:
            assert (client.timeout, client.model) == (33.0, "turbo")
            f = client.hallucination_filter
            assert (f.no_speech_prob_threshold, f.avg_logprob_threshold) == (0.6, -0.3)
            assert (f.legacy_max_words, f.logprob_floor) == (4, -1.2)
            assert (f.phrases, f.phrase_max_extra_words) == (["Тест"], 2)

    def test_zero_max_inflight_means_auto(self):
        cfg = TranscriptionConfig(servers=["gpu1"], max_inflight=0)
        assert Transcriber.from_config(cfg, ["http://a:8000"], "ru").max_inflight == 3

    def test_unknown_segment_mode_raises(self):
        with pytest.raises(ConfigurationError, match="segment_mode"):
            make_transcriber(segment_mode="tracks")
        with pytest.raises(ConfigurationError, match="segment_mode"):
            TranscriptionConfig(segment_mode="tracks")
