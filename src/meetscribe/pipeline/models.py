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
    max_gap_ms: int,
    max_chunk_ms: int,
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


def collect_sample_segments(
    segments: list[SpeechSegment],
    min_duration_ms: int,
    max_duration_ms: int,
    ideal_ms: int,
) -> dict[str, list[SpeechSegment]]:
    """Group labeled segments by speaker for sample extraction.

    Segments in [min_duration_ms, max_duration_ms] are kept as-is.
    Segments longer than max_duration_ms are sliced into ~ideal_ms chunks.
    Segments shorter than min_duration_ms or without a speaker are skipped.
    """
    speaker_segments: dict[str, list[SpeechSegment]] = {}
    for seg in segments:
        if not seg.speaker:
            continue
        if min_duration_ms <= seg.duration_ms <= max_duration_ms:
            speaker_segments.setdefault(seg.speaker, []).append(seg)
        elif seg.duration_ms > max_duration_ms:
            pos = seg.start_ms
            while pos < seg.end_ms:
                chunk_end = min(pos + ideal_ms, seg.end_ms)
                if chunk_end - pos >= min_duration_ms:
                    chunk = SpeechSegment(
                        start_ms=pos, end_ms=chunk_end, speaker=seg.speaker
                    )
                    speaker_segments.setdefault(seg.speaker, []).append(chunk)
                pos = chunk_end
    return speaker_segments



def merge_by_proximity(
    segments: list[SpeechSegment],
    max_gap_ms: int,
    max_chunk_ms: int,
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
