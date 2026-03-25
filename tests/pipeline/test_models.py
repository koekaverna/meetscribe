"""Tests for pipeline/models.py — segment merging logic."""

from meetscribe.pipeline.models import (
    SpeechSegment,
    TranscriptSegment,
    merge_by_proximity,
    merge_close_segments,
)


class TestMergeCloseSegments:
    def test_empty_returns_empty(self):
        assert merge_close_segments([], max_gap_ms=500, max_chunk_ms=30000) == []

    def test_single_segment_unchanged(self):
        segs = [SpeechSegment(0, 1000, "Alice")]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000
        assert result[0].speaker == "Alice"

    def test_same_speaker_small_gap_merges(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1200, 2000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[0].speaker == "Alice"

    def test_different_speakers_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1200, 2000, "Bob"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000
        assert result[0].speaker == "Alice"
        assert result[1].start_ms == 1200
        assert result[1].end_ms == 2000
        assert result[1].speaker == "Bob"

    def test_large_gap_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(2000, 3000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000
        assert result[1].start_ms == 2000
        assert result[1].end_ms == 3000

    def test_max_chunk_ms_prevents_merge(self):
        segs = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(2100, 4000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=3000)
        assert len(result) == 2
        assert result[0].end_ms == 2000
        assert result[1].start_ms == 2100

    def test_chain_merge_three_segments(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1200, 2000, "Alice"),
            SpeechSegment(2300, 3000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 3000


class TestMergeByProximity:
    def test_empty_returns_empty(self):
        assert merge_by_proximity([], max_gap_ms=500, max_chunk_ms=30000) == []

    def test_different_speakers_still_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1200, 2000, "Bob"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[0].speaker == "Alice"

    def test_large_gap_keeps_separate_segments_with_correct_values(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(3000, 4000, "Bob"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000
        assert result[0].speaker == "Alice"
        assert result[1].start_ms == 3000
        assert result[1].end_ms == 4000
        assert result[1].speaker == "Bob"

    def test_max_chunk_prevents_merge_preserves_content(self):
        segs = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(2100, 4000, "Bob"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=3000)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[0].speaker == "Alice"
        assert result[1].start_ms == 2100
        assert result[1].end_ms == 4000
        assert result[1].speaker == "Bob"

    def test_exact_boundary_gap_merges(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1500, 2000, "Alice"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].end_ms == 2000

    def test_one_over_boundary_gap_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1501, 2000, "Alice"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].end_ms == 1000
        assert result[1].start_ms == 1501

    def test_three_segments_partial_merge(self):
        """First two merge, third doesn't due to gap."""
        segs = [
            SpeechSegment(0, 1000, "A"),
            SpeechSegment(1200, 2000, "B"),
            SpeechSegment(5000, 6000, "C"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[0].speaker == "A"
        assert result[1].start_ms == 5000
        assert result[1].end_ms == 6000
        assert result[1].speaker == "C"

    def test_duration_calculated_from_first_segment_start(self):
        """Verify duration = seg.end_ms - cur.start_ms (not + cur.start_ms)."""
        segs = [
            SpeechSegment(1000, 2000, "A"),
            SpeechSegment(2100, 3500, "B"),
        ]
        # duration = 3500 - 1000 = 2500, max_chunk=2500 → should merge
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=2500)
        assert len(result) == 1
        assert result[0].end_ms == 3500

        # duration = 3500 - 1000 = 2500, max_chunk=2499 → should NOT merge
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=2499)
        assert len(result) == 2


class TestMergeCloseSegmentsBoundary:
    def test_exact_gap_boundary_merges(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1500, 2000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 1
        assert result[0].end_ms == 2000

    def test_one_over_gap_boundary_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1501, 2000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2

    def test_exact_chunk_boundary_merges(self):
        segs = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(2100, 3000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=3000)
        assert len(result) == 1

    def test_one_over_chunk_boundary_not_merged(self):
        segs = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(2100, 3001, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=3000)
        assert len(result) == 2


class TestDataclassDefaults:
    def test_speech_segment_default_speaker_is_none(self):
        seg = SpeechSegment(0, 1000)
        assert seg.speaker is None

    def test_transcript_segment_default_speaker_is_none(self):
        seg = TranscriptSegment(0, 1000, "Hello")
        assert seg.speaker is None

    def test_speech_segment_duration(self):
        seg = SpeechSegment(500, 1500)
        assert seg.duration_ms == 1000
