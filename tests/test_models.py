"""Tests for pipeline/models.py — segment merging logic."""

from meetscribe.pipeline.models import SpeechSegment, merge_by_proximity, merge_close_segments


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
        assert result[0].speaker == "Alice"
        assert result[1].speaker == "Bob"

    def test_large_gap_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(2000, 3000, "Alice"),
        ]
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
        assert result[0].end_ms == 1000
        assert result[1].start_ms == 2000

    def test_max_chunk_ms_prevents_merge(self):
        segs = [
            SpeechSegment(0, 2000, "Alice"),
            SpeechSegment(2100, 4000, "Alice"),
        ]
        # Total would be 4000ms, but max_chunk_ms=3000 prevents merge
        result = merge_close_segments(segs, max_gap_ms=500, max_chunk_ms=3000)
        assert len(result) == 2

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

    def test_speaker_preserved_from_first(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(1200, 2000, "Bob"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert result[0].speaker == "Alice"

    def test_large_gap_not_merged(self):
        segs = [
            SpeechSegment(0, 1000, "Alice"),
            SpeechSegment(3000, 4000, "Alice"),
        ]
        result = merge_by_proximity(segs, max_gap_ms=500, max_chunk_ms=30000)
        assert len(result) == 2
