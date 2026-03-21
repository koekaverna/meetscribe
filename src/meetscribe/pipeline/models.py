"""Shared data classes for the pipeline."""

from dataclasses import dataclass


@dataclass
class SpeechSegment:
    """A segment of detected speech with optional speaker label."""

    start_ms: int
    end_ms: int
    speaker: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class TranscriptSegment:
    """Transcribed segment with timestamp and speaker."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


def merge_close_segments(
    segments: list[SpeechSegment],
    max_gap_ms: int = 500,
    max_chunk_ms: int = 30000,
) -> list[SpeechSegment]:
    """Merge segments with small gaps into larger chunks for transcription.

    Preserves the speaker from the first segment in each merged group.
    Only merges segments with the same speaker.
    """
    if not segments:
        return []

    merged: list[SpeechSegment] = []
    cur = SpeechSegment(segments[0].start_ms, segments[0].end_ms, segments[0].speaker)

    for seg in segments[1:]:
        gap = seg.start_ms - cur.end_ms
        duration = seg.end_ms - cur.start_ms

        if gap <= max_gap_ms and duration <= max_chunk_ms and seg.speaker == cur.speaker:
            cur.end_ms = seg.end_ms
        else:
            merged.append(cur)
            cur = SpeechSegment(seg.start_ms, seg.end_ms, seg.speaker)

    merged.append(cur)
    return merged


def merge_by_proximity(
    segments: list[SpeechSegment],
    max_gap_ms: int = 500,
    max_chunk_ms: int = 30000,
) -> list[SpeechSegment]:
    """Merge close segments ignoring speaker, purely by time proximity.

    Used before speaker identification to create reasonable chunks for STT.
    Speaker field is preserved from the first segment in each group.
    """
    if not segments:
        return []

    merged: list[SpeechSegment] = []
    cur = SpeechSegment(segments[0].start_ms, segments[0].end_ms, segments[0].speaker)

    for seg in segments[1:]:
        gap = seg.start_ms - cur.end_ms
        duration = seg.end_ms - cur.start_ms

        if gap <= max_gap_ms and duration <= max_chunk_ms:
            cur.end_ms = seg.end_ms
        else:
            merged.append(cur)
            cur = SpeechSegment(seg.start_ms, seg.end_ms, seg.speaker)

    merged.append(cur)
    return merged
