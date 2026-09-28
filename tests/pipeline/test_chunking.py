"""Tests for pipeline/models.py — chunk boundaries for transcription."""

from meetscribe.pipeline.models import (
    SpeechSegment,
    absorb_short_chunks,
    pad_chunks,
    resolve_speaker_overlaps,
    split_long_segments,
)


def _spans(segments: list[SpeechSegment]) -> list[tuple[int, int, str | None]]:
    return [(s.start_ms, s.end_ms, s.speaker) for s in segments]


def _midpoint(lo: int, hi: int) -> int:
    return (lo + hi) // 2


class TestResolveSpeakerOverlaps:
    def test_no_overlap_unchanged(self):
        segs = [SpeechSegment(0, 1000, "A"), SpeechSegment(1000, 2000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 1000, "A"), (1000, 2000, "B")]

    def test_does_not_modify_input(self):
        segs = [SpeechSegment(0, 10000, "A"), SpeechSegment(8000, 12000, "B")]
        resolve_speaker_overlaps(segs)
        assert _spans(segs) == [(0, 10000, "A"), (8000, 12000, "B")]

    def test_shared_span_goes_to_longer_earlier_segment(self):
        segs = [SpeechSegment(0, 10000, "A"), SpeechSegment(8000, 12000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 10000, "A"), (10000, 12000, "B")]

    def test_shared_span_goes_to_longer_later_segment(self):
        segs = [SpeechSegment(0, 4000, "A"), SpeechSegment(3000, 20000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 3000, "A"), (3000, 20000, "B")]

    def test_tie_goes_to_earlier_segment(self):
        segs = [SpeechSegment(0, 4000, "A"), SpeechSegment(3000, 7000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 4000, "A"), (4000, 7000, "B")]

    def test_segment_inside_longer_one_is_dropped(self):
        segs = [SpeechSegment(0, 10000, "A"), SpeechSegment(4000, 5000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 10000, "A")]

    def test_same_start_shorter_is_dropped(self):
        segs = [SpeechSegment(0, 2000, "A"), SpeechSegment(0, 9000, "B")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 9000, "B")]

    def test_same_speaker_overlap_left_alone(self):
        segs = [SpeechSegment(0, 5000, "A"), SpeechSegment(4000, 8000, "A")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 5000, "A"), (4000, 8000, "A")]

    def test_overlap_with_non_adjacent_segment(self):
        """A long segment overlaps one that is not next to it in the list."""
        segs = [
            SpeechSegment(0, 20000, "A"),
            SpeechSegment(1000, 2000, "A"),
            SpeechSegment(19000, 23000, "B"),
        ]
        assert _spans(resolve_speaker_overlaps(segs)) == [
            (0, 20000, "A"),
            (1000, 2000, "A"),
            (20000, 23000, "B"),
        ]

    def test_precedence_uses_original_duration(self):
        """B was trimmed by A, but still outweighs C by its original length."""
        segs = [
            SpeechSegment(0, 10000, "A"),
            SpeechSegment(9000, 14000, "B"),
            SpeechSegment(13000, 17500, "C"),
        ]
        assert _spans(resolve_speaker_overlaps(segs)) == [
            (0, 10000, "A"),
            (10000, 14000, "B"),
            (14000, 17500, "C"),
        ]

    def test_sorts_by_start(self):
        segs = [SpeechSegment(5000, 6000, "B"), SpeechSegment(0, 1000, "A")]
        assert _spans(resolve_speaker_overlaps(segs)) == [(0, 1000, "A"), (5000, 6000, "B")]


class TestSplitLongSegments:
    def test_short_segment_unchanged(self):
        segs = [SpeechSegment(0, 30000, "A")]
        assert _spans(split_long_segments(segs, 30000, _midpoint)) == [(0, 30000, "A")]

    def test_window_keeps_both_parts_within_limit(self):
        windows = []

        def find_split(lo: int, hi: int) -> int:
            windows.append((lo, hi))
            return lo

        result = split_long_segments([SpeechSegment(10000, 50000, "A")], 30000, find_split)

        # 40 s: the cut must leave at most 30 s on either side
        assert windows == [(20000, 40000)]
        assert _spans(result) == [(10000, 20000, "A"), (20000, 50000, "A")]

    def test_very_long_segment_split_repeatedly(self):
        result = split_long_segments([SpeechSegment(0, 100000, "A")], 30000, lambda lo, hi: hi)
        assert _spans(result) == [
            (0, 30000, "A"),
            (30000, 60000, "A"),
            (60000, 90000, "A"),
            (90000, 100000, "A"),
        ]

    def test_split_point_clamped_to_window(self):
        result = split_long_segments([SpeechSegment(0, 40000, "A")], 30000, lambda lo, hi: 0)
        assert _spans(result) == [(0, 10000, "A"), (10000, 40000, "A")]

    def test_parts_cover_segment_within_limit(self):
        result = split_long_segments([SpeechSegment(500, 95500, "A")], 30000, _midpoint)
        assert result[0].start_ms == 500
        assert result[-1].end_ms == 95500
        assert all(a.end_ms == b.start_ms for a, b in zip(result, result[1:]))
        assert all(s.duration_ms <= 30000 for s in result)


class TestAbsorbShortChunks:
    def test_short_chunk_joins_previous(self):
        chunks = [SpeechSegment(0, 29500, "A"), SpeechSegment(29800, 30400, "A")]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == [(0, 30400, "A")]

    def test_short_chunk_joins_next(self):
        chunks = [SpeechSegment(0, 600, "A"), SpeechSegment(900, 30500, "A")]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == [(0, 30500, "A")]

    def test_other_speaker_not_joined(self):
        chunks = [
            SpeechSegment(0, 5000, "A"),
            SpeechSegment(5100, 5600, "B"),
            SpeechSegment(5700, 9000, "A"),
        ]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == _spans(chunks)

    def test_long_pause_not_joined(self):
        chunks = [SpeechSegment(0, 5000, "A"), SpeechSegment(5600, 6000, "A")]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == _spans(chunks)

    def test_chunk_at_threshold_is_kept(self):
        chunks = [SpeechSegment(0, 5000, "A"), SpeechSegment(5100, 6100, "A")]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == _spans(chunks)

    def test_run_of_short_chunks_collapses(self):
        chunks = [
            SpeechSegment(0, 400, "A"),
            SpeechSegment(500, 900, "A"),
            SpeechSegment(1000, 1300, "A"),
        ]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == [(0, 1300, "A")]

    def test_lone_short_chunk_is_kept(self):
        chunks = [SpeechSegment(0, 400, "A")]
        assert _spans(absorb_short_chunks(chunks, 1000, 500)) == [(0, 400, "A")]

    def test_does_not_modify_input(self):
        chunks = [SpeechSegment(0, 5000, "A"), SpeechSegment(5100, 5400, "A")]
        absorb_short_chunks(chunks, 1000, 500)
        assert _spans(chunks) == [(0, 5000, "A"), (5100, 5400, "A")]


class TestPadChunks:
    def test_padding_with_room(self):
        chunks = [SpeechSegment(1000, 2000, "A"), SpeechSegment(5000, 6000, "B")]
        assert pad_chunks(chunks, 200, 10000) == [(800, 2200), (4800, 6200)]

    def test_clipped_at_track_edges(self):
        chunks = [SpeechSegment(100, 2000, "A"), SpeechSegment(5000, 9900, "B")]
        assert pad_chunks(chunks, 200, 10000) == [(0, 2200), (4800, 10000)]

    def test_narrow_pause_is_shared(self):
        chunks = [SpeechSegment(1000, 2000, "A"), SpeechSegment(2100, 3000, "B")]
        assert pad_chunks(chunks, 200, 10000) == [(800, 2050), (2050, 3200)]

    def test_touching_chunks_get_no_padding_between(self):
        chunks = [SpeechSegment(1000, 2000, "A"), SpeechSegment(2000, 3000, "B")]
        assert pad_chunks(chunks, 200, 10000) == [(800, 2000), (2000, 3200)]

    def test_overlapping_chunks_get_no_padding_between(self):
        chunks = [SpeechSegment(1000, 2500, "A"), SpeechSegment(2000, 3000, "A")]
        assert pad_chunks(chunks, 200, 10000) == [(800, 2500), (2000, 3200)]

    def test_zero_padding(self):
        chunks = [SpeechSegment(1000, 2000, "A")]
        assert pad_chunks(chunks, 0, 10000) == [(1000, 2000)]

    def test_chunk_beyond_track_end(self):
        chunks = [SpeechSegment(1000, 10500, "A")]
        assert pad_chunks(chunks, 200, 10000) == [(800, 10500)]
