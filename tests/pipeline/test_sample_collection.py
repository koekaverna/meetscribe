"""Tests for collect_sample_segments — slicing long segments for sample extraction."""

from meetscribe.pipeline.models import SpeechSegment, collect_sample_segments

# Default config values (match EmbeddingsConfig defaults)
MIN_MS = 3000
MAX_MS = 12000
IDEAL_MS = 7000


def _seg(start_ms: int, end_ms: int, speaker: str | None = "Alice") -> SpeechSegment:
    return SpeechSegment(start_ms=start_ms, end_ms=end_ms, speaker=speaker)


def _collect(segs: list[SpeechSegment], **kwargs) -> dict[str, list[SpeechSegment]]:
    defaults = {"min_duration_ms": MIN_MS, "max_duration_ms": MAX_MS, "ideal_ms": IDEAL_MS}
    defaults.update(kwargs)
    return collect_sample_segments(segs, **defaults)


class TestCollectSampleSegments:
    def test_empty_input(self):
        assert _collect([]) == {}

    def test_segments_in_range_kept(self):
        segs = [_seg(0, 5000), _seg(10000, 18000)]
        result = _collect(segs)
        assert len(result["Alice"]) == 2

    def test_short_segments_skipped(self):
        segs = [_seg(0, 2000), _seg(5000, 6000)]
        result = _collect(segs)
        assert result == {}

    def test_no_speaker_skipped(self):
        segs = [_seg(0, 5000, speaker=None)]
        result = _collect(segs)
        assert result == {}

    def test_long_segment_sliced_into_chunks(self):
        # 21s segment → 3 chunks of 7s each
        segs = [_seg(0, 21000)]
        result = _collect(segs)
        chunks = result["Alice"]
        assert len(chunks) == 3
        assert chunks[0].start_ms == 0 and chunks[0].end_ms == 7000
        assert chunks[1].start_ms == 7000 and chunks[1].end_ms == 14000
        assert chunks[2].start_ms == 14000 and chunks[2].end_ms == 21000

    def test_long_segment_short_remainder_dropped(self):
        # 15s segment → 2 chunks: 7s + 7s, remainder 1s dropped
        segs = [_seg(0, 15000)]
        result = _collect(segs)
        chunks = result["Alice"]
        assert len(chunks) == 2
        assert chunks[1].end_ms == 14000

    def test_long_segment_valid_remainder_kept(self):
        # 13s segment → 7s + 6s (6s >= 3s min)
        segs = [_seg(0, 13000)]
        result = _collect(segs)
        chunks = result["Alice"]
        assert len(chunks) == 2
        assert chunks[1].start_ms == 7000 and chunks[1].end_ms == 13000

    def test_chunks_preserve_speaker(self):
        segs = [_seg(0, 21000, "Bob")]
        result = _collect(segs)
        assert all(c.speaker == "Bob" for c in result["Bob"])

    def test_multiple_speakers_grouped(self):
        segs = [
            _seg(0, 5000, "Alice"),
            _seg(10000, 25000, "Bob"),  # long, gets sliced
            _seg(30000, 35000, "Alice"),
        ]
        result = _collect(segs)
        assert len(result["Alice"]) == 2
        assert len(result["Bob"]) >= 2  # 15s → 2 chunks

    def test_exact_max_duration_kept_as_is(self):
        segs = [_seg(0, 12000)]
        result = _collect(segs)
        assert len(result["Alice"]) == 1
        assert result["Alice"][0].end_ms == 12000

    def test_just_over_max_gets_sliced(self):
        segs = [_seg(0, 12001)]
        result = _collect(segs)
        chunks = result["Alice"]
        assert len(chunks) >= 1
        assert chunks[0].end_ms == 7000

    def test_custom_params(self):
        # 10s segment, with min=2s, max=5s, ideal=4s → 4+4+2
        segs = [_seg(0, 10000)]
        result = _collect(segs, min_duration_ms=2000, max_duration_ms=5000, ideal_ms=4000)
        chunks = result["Alice"]
        assert len(chunks) == 3
        assert chunks[0].duration_ms == 4000
        assert chunks[1].duration_ms == 4000
        assert chunks[2].duration_ms == 2000
