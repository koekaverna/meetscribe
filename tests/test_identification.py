"""Tests for pipeline/embeddings.py — speaker identification and enrollment."""

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meetscribe.pipeline.embeddings import SpeakerIdentifier, enroll_samples
from meetscribe.pipeline.models import SpeechSegment
from tests.conftest import make_wav_file


def _norm(raw: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in raw))
    return [x / n for x in raw] if n > 0 else raw


# Two clearly distinct speakers
ALICE_EMB = _norm([1.0, 0.0, 0.0, 0.0])
BOB_EMB = _norm([0.0, 1.0, 0.0, 0.0])
# Similar to Alice but not identical
ALICE_LIKE = _norm([0.95, 0.05, 0.0, 0.0])
# Ambiguous — between Alice and Bob
AMBIGUOUS = _norm([0.5, 0.5, 0.0, 0.0])


class TestSpeakerIdentifierIdentify:
    def _make_identifier(
        self,
        voiceprints=None,
        threshold=0.8,
        confident_gap=0.2,
        min_threshold=0.4,
    ):
        vp = voiceprints or {}
        return SpeakerIdentifier(
            vp,
            threshold,
            confident_gap,
            min_threshold,
            unknown_cluster_threshold=0.25,
        )

    def test_no_voiceprints_returns_none(self):
        ident = self._make_identifier()
        name, sim = ident.identify(ALICE_EMB)
        assert name is None
        assert sim == -1.0

    def test_direct_match_above_threshold(self):
        ident = self._make_identifier({"Alice": ALICE_EMB})
        name, sim = ident.identify(ALICE_EMB)
        assert name == "Alice"
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_below_threshold_no_match(self):
        # Need two voiceprints so confident_gap doesn't trigger
        ident = self._make_identifier(
            {"Alice": ALICE_EMB, "Bob": BOB_EMB},
            threshold=0.999,
            confident_gap=2.0,  # impossible gap
            min_threshold=0.999,  # floor above any real similarity
        )
        name, sim = ident.identify(ALICE_LIKE)
        assert name is None

    def test_confident_gap_accepts(self):
        # Alice is the only voiceprint, so gap to second = sim - (-1.0) which is large
        ident = self._make_identifier(
            {"Alice": ALICE_EMB},
            threshold=0.99,  # direct match won't work
            confident_gap=0.2,
            min_threshold=0.4,
        )
        name, sim = ident.identify(ALICE_LIKE)
        assert name == "Alice"

    def test_below_min_threshold_rejected(self):
        ident = self._make_identifier(
            {"Alice": ALICE_EMB},
            threshold=0.99,
            confident_gap=0.01,
            min_threshold=0.99,  # floor so high nothing passes
        )
        name, _ = ident.identify(AMBIGUOUS)
        assert name is None

    def test_best_match_selected_from_multiple(self):
        ident = self._make_identifier({"Alice": ALICE_EMB, "Bob": BOB_EMB})
        name, _ = ident.identify(ALICE_LIKE)
        assert name == "Alice"


class TestFindNearestLabeled:
    def test_left_neighbor(self):
        segs = [
            SpeechSegment(0, 100, "Alice"),
            SpeechSegment(100, 200, None),
        ]
        result = SpeakerIdentifier._find_nearest_labeled(1, segs)
        assert result == "Alice"

    def test_right_neighbor(self):
        segs = [
            SpeechSegment(0, 100, None),
            SpeechSegment(100, 200, "Bob"),
        ]
        result = SpeakerIdentifier._find_nearest_labeled(0, segs)
        assert result == "Bob"

    def test_left_preferred_when_equidistant(self):
        segs = [
            SpeechSegment(0, 100, "Alice"),
            SpeechSegment(100, 200, None),
            SpeechSegment(200, 300, "Bob"),
        ]
        result = SpeakerIdentifier._find_nearest_labeled(1, segs)
        assert result == "Alice"

    def test_no_labeled_neighbors(self):
        segs = [
            SpeechSegment(0, 100, None),
            SpeechSegment(100, 200, None),
        ]
        result = SpeakerIdentifier._find_nearest_labeled(0, segs)
        assert result is None


class TestIdentifySegments:
    def _make_identifier(self, voiceprints):
        return SpeakerIdentifier(
            voiceprints,
            threshold=0.8,
            confident_gap=0.2,
            min_threshold=0.4,
            unknown_cluster_threshold=0.25,
        )

    def test_all_known_speakers(self):
        ident = self._make_identifier({"Alice": ALICE_EMB, "Bob": BOB_EMB})
        segments_with_emb = [
            (SpeechSegment(0, 1000), ALICE_EMB),
            (SpeechSegment(1000, 2000), BOB_EMB),
        ]
        result = ident.identify_segments(segments_with_emb)
        assert result[0].speaker == "Alice"
        assert result[1].speaker == "Bob"

    def test_all_unknown_get_cluster_labels(self):
        ident = self._make_identifier({})
        segments_with_emb = [
            (SpeechSegment(0, 1000), ALICE_EMB),
            (SpeechSegment(1000, 2000), ALICE_EMB),
            (SpeechSegment(2000, 3000), BOB_EMB),
        ]
        result = ident.identify_segments(segments_with_emb)
        # All should have "Unknown-N" labels
        assert all(s.speaker.startswith("Unknown-") for s in result)
        # Same voice → same label
        assert result[0].speaker == result[1].speaker
        # Different voice → different label
        assert result[0].speaker != result[2].speaker

    def test_short_segments_inherit_from_neighbor(self):
        ident = self._make_identifier({"Alice": ALICE_EMB})
        segments_with_emb = [
            (SpeechSegment(0, 1000), ALICE_EMB),
            (SpeechSegment(1000, 1500), None),  # too short, no embedding
        ]
        result = ident.identify_segments(segments_with_emb)
        assert result[0].speaker == "Alice"
        assert result[1].speaker == "Alice"  # inherited

    def test_mixed_known_and_unknown(self):
        ident = self._make_identifier({"Alice": ALICE_EMB})
        segments_with_emb = [
            (SpeechSegment(0, 1000), ALICE_EMB),
            (SpeechSegment(1000, 2000), BOB_EMB),  # unknown
        ]
        result = ident.identify_segments(segments_with_emb)
        assert result[0].speaker == "Alice"
        assert result[1].speaker.startswith("Unknown-")


class TestEnrollSamples:
    def test_copies_files_and_counts(self, tmp_path: Path, mock_extractor: MagicMock):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        enrolled_dir = tmp_path / "enrolled"

        # Create 2 sample WAV files in source
        make_wav_file(source_dir / "s1.wav")
        make_wav_file(source_dir / "s2.wav")

        sample_paths = list(source_dir.glob("*.wav"))
        embedding, total, new = enroll_samples(mock_extractor, sample_paths, enrolled_dir)

        assert new == 2
        assert total == 2
        assert len(embedding) > 0
        # Files actually copied to enrolled_dir
        assert len(list(enrolled_dir.glob("*.wav"))) == 2

    def test_skips_files_already_in_enrolled_dir(self, tmp_path: Path, mock_extractor: MagicMock):
        enrolled_dir = tmp_path / "enrolled"
        enrolled_dir.mkdir(parents=True)

        # File already in enrolled_dir
        make_wav_file(enrolled_dir / "existing.wav")

        sample_paths = list(enrolled_dir.glob("*.wav"))
        _, total, new = enroll_samples(mock_extractor, sample_paths, enrolled_dir)

        assert new == 0
        assert total == 1  # existing file still counted

    def test_includes_previously_enrolled(self, tmp_path: Path, mock_extractor: MagicMock):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        enrolled_dir = tmp_path / "enrolled"
        enrolled_dir.mkdir(parents=True)

        # Pre-existing enrolled sample
        make_wav_file(enrolled_dir / "old.wav")
        # New sample
        make_wav_file(source_dir / "new.wav")

        sample_paths = [source_dir / "new.wav"]
        _, total, new = enroll_samples(mock_extractor, sample_paths, enrolled_dir)

        assert new == 1
        assert total == 2  # old + new
        # extract_from_file called for both files (voiceprint includes all)
        assert mock_extractor.extract_from_file.call_count == 2
