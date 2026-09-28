"""Shared data classes for the pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, field


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
    track_num: int | None = None
    # Set on segments removed from the transcript: why (see pipeline.hallucination)
    drop_reason: str | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass
class TranscriptionResult:
    """Outcome of transcribing one track.

    ``dropped`` holds segments filtered out of the transcript, each with its
    ``drop_reason``, so they can be reviewed and restored. ``failed_chunks``
    are audio spans the server could not transcribe.
    """

    segments: list[TranscriptSegment] = field(default_factory=list)
    dropped: list[TranscriptSegment] = field(default_factory=list)
    failed_chunks: list[SpeechSegment] = field(default_factory=list)


def format_transcript_markdown(segments: list[TranscriptSegment]) -> str:
    """Render segments as a markdown dialogue: `**[MM:SS] Speaker:** text`."""
    return "\n\n".join(
        f"**[{s.start_ms // 60000:02d}:{(s.start_ms // 1000) % 60:02d}] "
        f"{s.speaker or 'Unknown'}:** {s.text}"
        for s in segments
    )


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


def resolve_speaker_overlaps(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    """Remove time overlaps between segments of different speakers.

    The shared span goes to the longer segment (the earlier one on a tie); the
    other is trimmed, or dropped when nothing of it is left. Overlaps within
    one speaker are left for the merge step. Returns new segments sorted by
    start time.
    """
    ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
    resolved: list[SpeechSegment] = []
    durations: list[int] = []  # original durations: trimming must not change precedence
    max_end_before: list[int] = []  # max end over resolved[:i + 1], upper bound

    for seg in ordered:
        cur = SpeechSegment(seg.start_ms, seg.end_ms, seg.speaker)
        duration = seg.duration_ms
        for i in range(len(resolved) - 1, -1, -1):
            if max_end_before[i] <= cur.start_ms:
                break
            other = resolved[i]
            if other.speaker == cur.speaker or other.end_ms <= cur.start_ms:
                continue
            if durations[i] >= duration:
                cur.start_ms = max(cur.start_ms, other.end_ms)
            else:
                other.end_ms = min(other.end_ms, cur.start_ms)
        if cur.duration_ms <= 0:
            continue
        resolved.append(cur)
        durations.append(duration)
        max_end_before.append(max(cur.end_ms, max_end_before[-1] if max_end_before else 0))

    return [s for s in resolved if s.duration_ms > 0]


def split_long_segments(
    segments: list[SpeechSegment],
    max_chunk_ms: int,
    find_split: Callable[[int, int], int],
    min_part_ms: int = 1000,
) -> list[SpeechSegment]:
    """Cut segments longer than max_chunk_ms into parts of at most max_chunk_ms.

    ``find_split(lo_ms, hi_ms)`` picks the cut point inside the allowed window —
    normally the longest pause, so words are not cut in half.
    """
    result: list[SpeechSegment] = []
    for seg in segments:
        start = seg.start_ms
        while seg.end_ms - start > max_chunk_ms:
            lo = start + min_part_ms
            hi = min(start + max_chunk_ms, seg.end_ms - min_part_ms)
            if seg.end_ms - start <= 2 * max_chunk_ms:
                # One cut is enough if the tail also fits into a chunk
                lo = max(lo, seg.end_ms - max_chunk_ms)
            if lo >= hi:
                break
            cut = min(max(find_split(lo, hi), lo), hi)
            result.append(SpeechSegment(start, cut, seg.speaker))
            start = cut
        result.append(SpeechSegment(start, seg.end_ms, seg.speaker))
    return result


def absorb_short_chunks(
    chunks: list[SpeechSegment],
    min_chunk_ms: int,
    max_gap_ms: int,
) -> list[SpeechSegment]:
    """Attach chunks shorter than min_chunk_ms to an adjacent chunk of the same speaker.

    Whisper is unreliable on sub-second clips. A short chunk joins the previous
    or the next chunk when it has the same speaker and the pause between them
    is at most max_gap_ms; otherwise it is kept as is.
    """
    result: list[SpeechSegment] = []
    pending: SpeechSegment | None = None  # short chunk waiting for the next one

    for chunk in chunks:
        cur = SpeechSegment(chunk.start_ms, chunk.end_ms, chunk.speaker)
        if pending is not None:
            if cur.speaker == pending.speaker and cur.start_ms - pending.end_ms <= max_gap_ms:
                cur.start_ms = min(cur.start_ms, pending.start_ms)
                cur.end_ms = max(cur.end_ms, pending.end_ms)
            else:
                result.append(pending)
            pending = None
        if cur.duration_ms < min_chunk_ms:
            prev = result[-1] if result else None
            if (
                prev is not None
                and prev.speaker == cur.speaker
                and cur.start_ms - prev.end_ms <= max_gap_ms
            ):
                prev.end_ms = max(prev.end_ms, cur.end_ms)
            else:
                pending = cur
            continue
        result.append(cur)

    if pending is not None:
        result.append(pending)
    return result


def pad_chunks(
    chunks: list[SpeechSegment],
    padding_ms: int,
    total_ms: int,
) -> list[tuple[int, int]]:
    """Audio bounds ``(start_ms, end_ms)`` for each chunk, widened by padding_ms.

    Padding never reaches into a neighbouring chunk: a pause between two chunks
    is shared between them half and half.
    """
    bounds: list[tuple[int, int]] = []
    max_end = 0
    for i, chunk in enumerate(chunks):
        room_before = chunk.start_ms if i == 0 else max(0, chunk.start_ms - max_end) // 2
        if i + 1 < len(chunks):
            room_after = max(0, chunks[i + 1].start_ms - chunk.end_ms) // 2
        else:
            room_after = max(0, total_ms - chunk.end_ms)
        bounds.append(
            (
                chunk.start_ms - min(padding_ms, room_before),
                chunk.end_ms + min(padding_ms, room_after),
            )
        )
        max_end = max(max_end, chunk.end_ms)
    return bounds


def filter_segments_by_speaker(
    segments: list[SpeechSegment], target_speaker: str
) -> list[SpeechSegment]:
    """Keep only segments attributed to the target speaker.

    Used for open-space recordings to drop other people's speech, leaving
    only the chosen speaker. Segments are filtered before transcription so
    foreign speech is never sent to the STT server.
    """
    return [s for s in segments if s.speaker == target_speaker]


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
                    chunk = SpeechSegment(start_ms=pos, end_ms=chunk_end, speaker=seg.speaker)
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
