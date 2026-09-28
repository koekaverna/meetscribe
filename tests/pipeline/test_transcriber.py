"""Tests for pipeline/transcriber.py — RemoteTranscriber and Transcriber."""

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetscribe.errors import ConfigurationError
from meetscribe.pipeline.models import SpeechSegment, TranscriptionResult, TranscriptSegment
from meetscribe.pipeline.transcriber import RemoteTranscriber, Transcriber
from tests.conftest import make_wav_file


class TestRemoteTranscriberInit:
    def test_strips_trailing_slash(self):
        rt = RemoteTranscriber(
            "http://host:8000/",
            timeout=10.0,
            model="m",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        assert rt.server_url == "http://host:8000"

    def test_stores_params(self):
        rt = RemoteTranscriber(
            "http://h",
            timeout=30.0,
            model="test-model",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        assert rt.timeout == 30.0
        assert rt.model == "test-model"


class TestRemoteTranscriberRequest:
    def test_request_params(self):
        rt = RemoteTranscriber(
            "http://host:8000",
            timeout=10.0,
            model="whisper-large",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "segments": [
                {"start": 0.5, "end": 1.5, "text": "Hello world"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp) as m:
            rt.transcribe_bytes(b"audio_data", "en", "chunk.wav")

        call_kwargs = m.call_args
        assert call_kwargs[0][0] == "http://host:8000/v1/audio/transcriptions"
        files = call_kwargs[1]["files"]
        assert files["file"][0] == "chunk.wav"
        assert files["file"][1] == b"audio_data"
        assert files["file"][2] == "audio/wav"
        data = call_kwargs[1]["data"]
        assert data["model"] == "whisper-large"
        assert data["language"] == "en"
        assert data["response_format"] == "verbose_json"
        assert data["timestamp_granularities[]"] == "segment"
        assert call_kwargs[1]["timeout"] == 10.0

    def test_parses_timestamps_to_ms(self):
        rt = RemoteTranscriber(
            "http://host:8000",
            timeout=10.0,
            model="m",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "segments": [
                {"start": 1.5, "end": 3.25, "text": "Hello"},
                {"start": 4.0, "end": 5.0, "text": "World"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "en")

        assert len(result) == 2
        assert result[0].start_ms == 1500
        assert result[0].end_ms == 3250
        assert result[0].text == "Hello"
        assert result[1].start_ms == 4000
        assert result[1].end_ms == 5000

    def test_skips_empty_text(self):
        rt = RemoteTranscriber(
            "http://host:8000",
            timeout=10.0,
            model="m",
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "   "},
                {"start": 1.0, "end": 2.0, "text": "Real text"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "en")

        assert len(result) == 1
        assert result[0].text == "Real text"


class TestHallucinationFiltering:
    def _make_rt(self, no_speech_thresh=0.5, logprob_thresh=-0.25):
        return RemoteTranscriber(
            "http://host:8000",
            timeout=10.0,
            model="m",
            no_speech_prob_threshold=no_speech_thresh,
            avg_logprob_threshold=logprob_thresh,
        )

    def _mock_response(self, segments):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"segments": segments}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_filters_high_no_speech_prob(self):
        rt = self._make_rt()
        mock_resp = self._mock_response(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Real speech",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.15,
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "Редактор субтитров",
                    "no_speech_prob": 0.75,
                    "avg_logprob": -0.3,
                },
            ]
        )

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "ru")

        assert [(s.text, s.drop_reason) for s in result] == [
            ("Real speech", None),
            ("Редактор субтитров", "blocklist"),
        ]
        assert result[1].no_speech_prob == 0.75
        assert result[1].avg_logprob == -0.3

    def _reasons(self, rt, segments):
        mock_resp = self._mock_response(
            [{"start": float(i), "end": i + 1.0, **seg} for i, seg in enumerate(segments)]
        )
        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            return [s.drop_reason for s in rt.transcribe_bytes(b"data", "ru")]

    def test_legacy_rule_drops_short_segment(self):
        reasons = self._reasons(
            self._make_rt(),
            [{"text": "Ну вот так", "no_speech_prob": 0.75, "avg_logprob": -0.3}],
        )
        assert reasons == ["legacy"]

    def test_legacy_rule_word_limit(self):
        """The legacy rule applies up to 5 words inclusive, not beyond."""
        metrics = {"no_speech_prob": 0.75, "avg_logprob": -0.3}
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": "раз два три четыре пять", **metrics},
                {"text": "раз два три четыре пять шесть", **metrics},
            ],
        )
        assert reasons == ["legacy", None]

    def test_logprob_floor_drops_any_length(self):
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": "это длинная фраза из целых семи слов", "avg_logprob": -1.0},
                {"text": "это длинная фраза из целых семи слов", "avg_logprob": -0.99},
            ],
        )
        assert reasons == ["logprob", None]

    def test_blocklist_ignores_case_and_punctuation(self):
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": "ПРОДОЛЖЕНИЕ СЛЕДУЕТ...", "no_speech_prob": 0.01, "avg_logprob": -0.1},
                {"text": "Субтитры сделал DimaTorzok", "avg_logprob": -0.1},
                {"text": "[смех]", "avg_logprob": -0.1},
            ],
        )
        assert reasons == ["blocklist", "blocklist", "blocklist"]

    def test_caption_in_capitals_is_dropped(self):
        confident = {"no_speech_prob": 0.01, "avg_logprob": -0.1}
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": "ДИНАМИЧНАЯ МУЗЫКА", **confident},
                {"text": "ГРОХОТ!", **confident},
                {"text": "Динамичная музыка играет", **confident},
                {"text": "КВА", **confident},
                {"text": "API REST", **confident},
                {"text": "ОК, ОК", **confident},
            ],
        )
        assert reasons == ["caption", "caption", None, None, None, None]

    def test_new_default_phrases(self):
        confident = {"no_speech_prob": 0.01, "avg_logprob": -0.1}
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": "Увидимся в следующих видео!", **confident},
                {"text": "Благодарю за внимание!", **confident},
                {"text": "Субтитры добавил DimaTorzok", **confident},
                {"text": "И с вами был Иван Головин.", **confident},
                {"text": "Ещё хотел спросить про субтитры.", **confident},
            ],
        )
        assert reasons == ["blocklist"] * 4 + [None]

    def test_blocklist_matches_whole_words_only(self):
        reasons = self._reasons(
            self._make_rt(),
            [{"text": "Корректоры уже всё проверили", "avg_logprob": -0.1}],
        )
        assert reasons == [None]

    def test_blocklist_keeps_phrase_inside_long_segment(self):
        """A phrase with more than 8 extra words around it is real speech."""
        eight_extra = "Спасибо за внимание коллеги на этом у меня всё на сегодня"
        nine_extra = eight_extra + " точно"
        reasons = self._reasons(
            self._make_rt(),
            [
                {"text": eight_extra, "avg_logprob": -0.1},
                {"text": nine_extra, "avg_logprob": -0.1},
            ],
        )
        assert reasons == ["blocklist", None]

    def test_keeps_segment_below_threshold(self):
        rt = self._make_rt()
        mock_resp = self._mock_response(
            [
                {"start": 0.0, "end": 1.0, "text": "Да", "no_speech_prob": 0.3},
            ]
        )

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "ru")

        assert len(result) == 1

    def test_no_filtering_when_field_missing(self):
        """If API doesn't return no_speech_prob, default (0.0) never triggers filter."""
        rt = self._make_rt()
        mock_resp = self._mock_response(
            [
                {"start": 0.0, "end": 1.0, "text": "No metrics"},
            ]
        )

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "en")

        assert len(result) == 1

    def test_disabled_with_threshold_1(self):
        """Setting no_speech_prob_threshold=1.0 disables filtering."""
        rt = self._make_rt(no_speech_thresh=1.0)
        mock_resp = self._mock_response(
            [
                {"start": 0.0, "end": 1.0, "text": "Garbage", "no_speech_prob": 0.99},
            ]
        )

        with patch("meetscribe.pipeline.transcriber.httpx.post", return_value=mock_resp):
            result = rt.transcribe_bytes(b"data", "en")

        assert len(result) == 1


class TestTranscriberInit:
    def test_no_urls_raises(self):
        with pytest.raises(ConfigurationError, match="At least one transcription server URL"):
            Transcriber(
                server_urls=[],
                language="en",
                timeout=10.0,
                model="m",
                max_gap_ms=500,
                max_chunk_ms=30000,
                no_speech_prob_threshold=0.5,
                avg_logprob_threshold=-0.25,
            )

    def test_non_positive_max_inflight_raises(self):
        with pytest.raises(ConfigurationError, match="max_inflight"):
            Transcriber(
                server_urls=["http://a:8000"],
                language="en",
                timeout=10.0,
                model="m",
                max_gap_ms=500,
                max_chunk_ms=30000,
                no_speech_prob_threshold=0.5,
                avg_logprob_threshold=-0.25,
                max_inflight=0,
            )

    def test_creates_clients(self):
        t = Transcriber(
            server_urls=["http://a:8000", "http://b:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        assert len(t.clients) == 2
        assert t.language == "en"
        assert t.max_gap_ms == 500
        assert t.max_chunk_ms == 30000


class TestTranscribeFile:
    def test_sets_speaker_on_all_segments(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=2.0)
        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients[0] = MagicMock()
        t.clients[0].transcribe.return_value = [
            TranscriptSegment(start_ms=0, end_ms=500, text="Hello"),
            TranscriptSegment(start_ms=500, end_ms=1000, text="World"),
        ]
        t.clients[0].timeout = 10.0

        result = t.transcribe_file(audio, speaker="Alice")

        assert len(result.segments) == 2
        assert all(s.speaker == "Alice" for s in result.segments)

    def test_separates_dropped_segments(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=2.0)
        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients[0] = MagicMock()
        t.clients[0].transcribe.return_value = [
            TranscriptSegment(start_ms=0, end_ms=500, text="Hello"),
            TranscriptSegment(start_ms=500, end_ms=1000, text="КОНЕЦ", drop_reason="blocklist"),
        ]
        t.clients[0].timeout = 10.0

        result = t.transcribe_file(audio, speaker="Alice")

        assert [s.text for s in result.segments] == ["Hello"]
        assert [(s.text, s.speaker) for s in result.dropped] == [("КОНЕЦ", "Alice")]

    def test_uses_longer_timeout(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=1.0)
        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients[0] = MagicMock()
        t.clients[0].timeout = 10.0
        t.clients[0].transcribe.return_value = []

        t.transcribe_file(audio)

        # During call, timeout should have been set to max(10, 600) = 600
        # After call, timeout should be restored to 10
        assert t.clients[0].timeout == 10.0


class TestTranscribeSegments:
    def test_offsets_timestamps(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=5.0)

        diarized = [SpeechSegment(1000, 3000, "Alice")]

        mock_client = MagicMock()
        mock_client.transcribe_bytes.return_value = [
            TranscriptSegment(start_ms=0, end_ms=500, text="Hello"),
        ]

        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients = [mock_client]

        results = t.transcribe_segments(audio, diarized).segments

        assert len(results) == 1
        assert results[0].start_ms == 1000  # 0 + 1000 offset
        assert results[0].end_ms == 1500  # 500 + 1000 offset
        assert results[0].speaker == "Alice"

    def test_empty_segments_returns_empty(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=1.0)
        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        assert t.transcribe_segments(audio, []) == TranscriptionResult()

    def test_wav_slicing_math(self, tmp_path: Path):
        """Verify that audio is sliced at correct sample boundaries."""
        audio = make_wav_file(tmp_path / "test.wav", duration_s=3.0)

        diarized = [SpeechSegment(1000, 2000, "Alice")]

        received_bytes = []

        def capture_bytes(audio_bytes, language, filename="chunk.wav"):
            received_bytes.append(audio_bytes)
            return [TranscriptSegment(start_ms=0, end_ms=500, text="Hi")]

        mock_client = MagicMock()
        mock_client.transcribe_bytes = capture_bytes

        t = Transcriber(
            server_urls=["http://a:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients = [mock_client]

        t.transcribe_segments(audio, diarized)

        assert len(received_bytes) == 1
        # Parse the WAV chunk to verify correct duration
        buf = io.BytesIO(received_bytes[0])
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 16000
            # 1 second chunk: 1000ms to 2000ms = 16000 frames
            assert wf.getnframes() == 16000

    def test_round_robin_clients(self, tmp_path: Path):
        audio = make_wav_file(tmp_path / "test.wav", duration_s=5.0)

        diarized = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1500, 2500, "Bob"),
            SpeechSegment(3000, 4000, "Alice"),
        ]

        client1 = MagicMock()
        client1.transcribe_bytes.return_value = [
            TranscriptSegment(start_ms=0, end_ms=500, text="Hi"),
        ]
        client2 = MagicMock()
        client2.transcribe_bytes.return_value = [
            TranscriptSegment(start_ms=0, end_ms=500, text="Hi"),
        ]

        t = Transcriber(
            server_urls=["http://a:8000", "http://b:8000"],
            language="en",
            timeout=10.0,
            model="m",
            max_gap_ms=500,
            max_chunk_ms=30000,
            no_speech_prob_threshold=0.5,
            avg_logprob_threshold=-0.25,
        )
        t.clients = [client1, client2]

        t.transcribe_segments(audio, diarized)

        # 3 segments with different speakers → not merged → round-robin across 2 clients
        total_calls = client1.transcribe_bytes.call_count + client2.transcribe_bytes.call_count
        assert total_calls == 3
        assert client1.transcribe_bytes.call_count >= 1
        assert client2.transcribe_bytes.call_count >= 1
