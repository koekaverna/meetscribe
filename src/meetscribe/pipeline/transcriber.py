"""Remote transcription via speaches API (OpenAI-compatible)."""

import io
import json
import logging
import sys
import time
import wave
from array import array
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from tqdm import tqdm

from meetscribe.errors import (
    ClipsEndpointUnavailable,
    ConfigurationError,
    SpeachesAPIError,
    speaches_retry,
)

from .hallucination import (
    REASON_DUPLICATE,
    REASON_FAILED,
    HallucinationFilter,
    normalize_words,
)
from .models import (
    SpeechSegment,
    TranscriptionResult,
    TranscriptSegment,
    absorb_short_chunks,
    merge_close_segments,
    pad_chunks,
    resolve_speaker_overlaps,
    split_long_segments,
)

if TYPE_CHECKING:
    from meetscribe.config import TranscriptionConfig

logger = logging.getLogger(__name__)

# With this many chunks failed and none transcribed the server is considered
# down: the run is aborted instead of waiting out retries for every chunk.
_MAX_FAILURES_WITHOUT_SUCCESS = 3

SEGMENT_MODE_CHUNKS = "chunks"
SEGMENT_MODE_CLIPS = "clips"
SEGMENT_MODES = (SEGMENT_MODE_CHUNKS, SEGMENT_MODE_CLIPS)

_CLIPS_PATH = "/v1/audio/transcriptions/clips"
# The clips endpoint rejects the whole request if any clip is longer
_CLIPS_MAX_MS = 30000
# One request carries a whole track and may wait for a free slot on the server
_TRACK_TIMEOUT_S = 600.0

_CLIPS_MAX_PAD_MS = 2000
# failed_clips reasons that mean "no text in this clip" rather than a failure
_CLIP_NO_TEXT_REASONS = ("empty", "no_speech")

_PAUSE_FRAME_MS = 20
# A frame is silent when its energy is below 1% of the loud frames' energy
# (i.e. RMS below 10%).
_PAUSE_ENERGY_RATIO = 0.01


@dataclass
class _TrackAudio:
    """PCM audio of a whole track, loaded into memory."""

    raw_frames: bytes
    sample_rate: int
    sample_width: int
    n_channels: int
    n_frames: int

    @property
    def duration_ms(self) -> int:
        return self.n_frames * 1000 // self.sample_rate if self.sample_rate else 0


def find_longest_pause(
    raw_frames: bytes,
    sample_rate: int,
    sample_width: int,
    n_channels: int,
    lo_ms: int,
    hi_ms: int,
) -> int:
    """Find the middle of the longest pause within [lo_ms, hi_ms] of PCM audio.

    Falls back to the quietest frame when there is no pause, and to the window
    midpoint for audio that is not 16-bit PCM.
    """
    midpoint = (lo_ms + hi_ms) // 2
    if sample_width != 2:
        return midpoint

    frame_size = sample_width * n_channels
    start_byte = lo_ms * sample_rate // 1000 * frame_size
    end_byte = hi_ms * sample_rate // 1000 * frame_size
    samples = array("h")
    samples.frombytes(raw_frames[start_byte:end_byte])
    if sys.byteorder == "big":
        samples.byteswap()

    step = sample_rate * _PAUSE_FRAME_MS // 1000 * n_channels
    energies = [
        sum(s * s for s in samples[i : i + step]) for i in range(0, len(samples) - step + 1, step)
    ]
    if not energies:
        return midpoint

    loud = sorted(energies)[int(len(energies) * 0.95)]
    threshold = loud * _PAUSE_ENERGY_RATIO

    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i, energy in enumerate(energies):
        if energy <= threshold:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0

    if best_len == 0:
        quietest = min(range(len(energies)), key=energies.__getitem__)
        best_start, best_len = quietest, 1

    return lo_ms + (best_start * 2 + best_len) * _PAUSE_FRAME_MS // 2


class RemoteTranscriber:
    """Single-server transcription client using OpenAI-compatible /v1/audio/transcriptions."""

    def __init__(
        self,
        server_url: str,
        timeout: float,
        model: str,
        no_speech_prob_threshold: float,
        avg_logprob_threshold: float,
        hallucination_filter: HallucinationFilter | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.avg_logprob_threshold = avg_logprob_threshold
        self.hallucination_filter = hallucination_filter or HallucinationFilter(
            no_speech_prob_threshold=no_speech_prob_threshold,
            avg_logprob_threshold=avg_logprob_threshold,
        )
        self._supports_clips: bool | None = None

    def transcribe(
        self,
        audio_path: Path,
        language: str,
    ) -> list[TranscriptSegment]:
        """Transcribe an audio file via the remote API.

        Hallucinated segments are returned too, marked with ``drop_reason``.
        """
        with open(audio_path, "rb") as f:
            return self._transcribe_request(audio_path.name, f.read(), language)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str,
        filename: str = "chunk.wav",
    ) -> list[TranscriptSegment]:
        """Transcribe WAV bytes via the remote API.

        Hallucinated segments are returned too, marked with ``drop_reason``.
        """
        return self._transcribe_request(filename, audio_bytes, language)

    @speaches_retry
    def _transcribe_request(
        self,
        filename: str,
        audio_bytes: bytes,
        language: str,
    ) -> list[TranscriptSegment]:
        """Send transcription request and parse response."""
        result = self._post(
            f"{self.server_url}/v1/audio/transcriptions",
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={
                "model": self.model,
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=self.timeout,
        )
        parsed = (self._parse_segment(seg) for seg in result.get("segments", []))
        return [seg for seg in parsed if seg is not None]

    def supports_clips(self) -> bool:
        """Whether the server has the track + clips endpoint. Checked once."""
        if self._supports_clips is None:
            try:
                response = httpx.get(f"{self.server_url}/openapi.json", timeout=self.timeout)
                response.raise_for_status()
                self._supports_clips = _CLIPS_PATH in response.json().get("paths", {})
            except (httpx.HTTPError, ValueError) as e:
                logger.warning(
                    "Could not check the clips endpoint",
                    extra={"server": self.server_url, "error": str(e)},
                )
                # Not cached: the server may just be restarting
                return False
        return self._supports_clips

    @speaches_retry
    def transcribe_clips(
        self,
        audio_path: Path,
        clips: list[SpeechSegment],
        language: str,
        pad_ms: int,
    ) -> tuple[dict[int, list[TranscriptSegment]], dict[int, str]]:
        """Transcribe clips of a track in one request: the track is uploaded once.

        Returns ``(segments, failed)`` keyed by the clip's index in ``clips``:
        the clip's segments in time order, timestamps in track coordinates, and
        the failure reason for clips the server could not process. The server
        cuts a long clip at pauses, so a clip may yield several segments.
        Hallucinated segments are returned too, marked with ``drop_reason``.

        Raises:
            ClipsEndpointUnavailable: If the server does not have the endpoint.
        """
        endpoint = f"{self.server_url}{_CLIPS_PATH}"
        payload = [
            {
                "start": clip.start_ms / 1000,
                "end": clip.end_ms / 1000,
                **({"speaker": clip.speaker} if clip.speaker else {}),
            }
            for clip in clips
        ]
        try:
            with open(audio_path, "rb") as f:
                result = self._post(
                    endpoint,
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={
                        "model": self.model,
                        "language": language,
                        "clips": json.dumps(payload, ensure_ascii=False),
                        "pad_ms": str(pad_ms),
                    },
                    timeout=max(self.timeout, _TRACK_TIMEOUT_S),
                )
        except SpeachesAPIError as e:
            # A 404 about the model ("... is not enabled on this server") is a
            # configuration problem, not a missing endpoint.
            model_problem = "is not enabled" in e.detail or "is not available" in e.detail
            if e.status_code in (404, 405) and not model_problem:
                self._supports_clips = False
                raise ClipsEndpointUnavailable(
                    str(e), status_code=e.status_code, endpoint=endpoint, detail=e.detail
                ) from e
            raise

        segments: dict[int, list[TranscriptSegment]] = {}
        for seg in result.get("segments", []):
            parsed = self._parse_segment(seg)
            index = seg.get("clip_index")
            if parsed is not None and isinstance(index, int) and 0 <= index < len(clips):
                segments.setdefault(index, []).append(parsed)
        for clip_segments in segments.values():
            clip_segments.sort(key=lambda s: s.start_ms)
        failed = {
            item["clip_index"]: str(item.get("reason", ""))
            for item in result.get("failed_clips", [])
            if isinstance(item.get("clip_index"), int) and 0 <= item["clip_index"] < len(clips)
        }
        return segments, failed

    def _post(self, endpoint: str, files: dict, data: dict, timeout: float) -> dict:
        """POST a multipart request, mapping HTTP failures to SpeachesAPIError."""
        try:
            response = httpx.post(endpoint, files=files, data=data, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SpeachesAPIError(
                f"Transcription failed: {e.response.status_code}",
                status_code=e.response.status_code,
                endpoint=endpoint,
                detail=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise SpeachesAPIError(
                f"Transcription connection error: {e}",
                endpoint=endpoint,
            ) from e
        return response.json()  # type: ignore[no-any-return]

    def _parse_segment(self, seg: dict) -> TranscriptSegment | None:
        """Build a segment from the API response and run the hallucination filter."""
        text = seg.get("text", "").strip()
        if not text:
            return None

        no_speech_prob = seg.get("no_speech_prob", 0.0)
        avg_logprob = seg.get("avg_logprob", 0.0)

        logger.debug(
            "Segment: '%s' (no_speech_prob=%.3f, avg_logprob=%.3f, start=%.1f, end=%.1f)",
            text[:80],
            no_speech_prob,
            avg_logprob,
            seg.get("start", 0),
            seg.get("end", 0),
        )

        drop_reason = self.hallucination_filter.check(text, no_speech_prob, avg_logprob)
        if drop_reason is not None:
            logger.debug(
                "Filtered hallucinated segment (%s): '%s' "
                "(no_speech_prob=%.3f, avg_logprob=%.3f, start=%.1f, end=%.1f)",
                drop_reason,
                text[:80],
                no_speech_prob,
                avg_logprob,
                seg.get("start", 0),
                seg.get("end", 0),
            )

        return TranscriptSegment(
            start_ms=int(seg["start"] * 1000),
            end_ms=int(seg["end"] * 1000),
            text=text,
            drop_reason=drop_reason,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
        )


class Transcriber:
    """Distribute transcription of diarized segments across remote servers.

    Takes diarized SpeechSegments (with speaker labels) and transcribes each chunk
    via remote speaches API servers.
    """

    def __init__(
        self,
        server_urls: list[str],
        language: str,
        timeout: float,
        model: str,
        max_gap_ms: int,
        max_chunk_ms: int,
        no_speech_prob_threshold: float,
        avg_logprob_threshold: float,
        max_inflight: int | None = None,
        hallucination_filter: HallucinationFilter | None = None,
        chunk_padding_ms: int = 0,
        min_chunk_ms: int = 0,
        dedup_similarity: float = 0.0,
        dedup_min_words: int = 4,
        segment_mode: str = SEGMENT_MODE_CHUNKS,
    ):
        if not server_urls:
            raise ConfigurationError("At least one transcription server URL is required")
        if segment_mode not in SEGMENT_MODES:
            raise ConfigurationError(f"segment_mode must be one of: {', '.join(SEGMENT_MODES)}")
        if max_inflight is not None and max_inflight < 1:
            raise ConfigurationError("max_inflight must be a positive integer")
        self.clients = [
            RemoteTranscriber(
                url,
                timeout,
                model,
                no_speech_prob_threshold,
                avg_logprob_threshold,
                hallucination_filter,
            )
            for url in server_urls
        ]
        self.language = language
        self.max_gap_ms = max_gap_ms
        self.max_chunk_ms = max_chunk_ms
        self.chunk_padding_ms = chunk_padding_ms
        self.min_chunk_ms = min_chunk_ms
        # 0 disables the duplicate check at chunk joints
        self.dedup_similarity = dedup_similarity
        self.dedup_min_words = dedup_min_words
        # "clips": one request per track (the track plus the list of clips),
        # falling back to per-chunk requests on servers without the endpoint.
        # With several servers only the first one is used in this mode.
        self.segment_mode = segment_mode
        # Cap on concurrent in-flight requests. Speaches has no cross-request
        # batching, so concurrency is the only way to use multiple servers (and
        # each server's threadpool) at once. Default to a few requests per server.
        self.max_inflight = max_inflight if max_inflight is not None else len(server_urls) * 3

    @classmethod
    def from_config(
        cls,
        cfg: "TranscriptionConfig",
        server_urls: list[str],
        language: str,
    ) -> "Transcriber":
        """Create a transcriber from the ``transcription`` config section."""
        return cls(
            server_urls,
            language=language,
            timeout=cfg.timeout,
            model=cfg.model,
            max_gap_ms=cfg.max_gap_ms,
            max_chunk_ms=cfg.max_chunk_ms,
            no_speech_prob_threshold=cfg.no_speech_prob_threshold,
            avg_logprob_threshold=cfg.avg_logprob_threshold,
            max_inflight=cfg.max_inflight or None,
            hallucination_filter=HallucinationFilter(
                no_speech_prob_threshold=cfg.no_speech_prob_threshold,
                avg_logprob_threshold=cfg.avg_logprob_threshold,
                legacy_max_words=cfg.legacy_filter_max_words,
                logprob_floor=cfg.hallucination_logprob_threshold,
                phrases=cfg.hallucination_phrases,
                phrase_max_extra_words=cfg.hallucination_phrase_max_extra_words,
                drop_captions=cfg.hallucination_drop_captions,
            ),
            chunk_padding_ms=cfg.chunk_padding_ms,
            min_chunk_ms=cfg.min_chunk_ms,
            dedup_similarity=cfg.dedup_similarity,
            dedup_min_words=cfg.dedup_min_words,
            segment_mode=cfg.segment_mode,
        )

    def transcribe_file(
        self,
        audio_path: Path,
        speaker: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an entire audio file without segmentation.

        Useful for named tracks where the speaker is already known.
        Uses a longer timeout since the file may be large.
        """
        t0 = time.perf_counter()
        client = self.clients[0]
        # Use longer timeout for whole-file transcription
        orig_timeout = client.timeout
        client.timeout = max(orig_timeout, 600.0)
        try:
            segments = client.transcribe(audio_path, self.language)
        finally:
            client.timeout = orig_timeout
        for seg in segments:
            seg.speaker = speaker

        result = TranscriptionResult(
            segments=[s for s in segments if s.drop_reason is None],
            dropped=[s for s in segments if s.drop_reason is not None],
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "File transcription completed",
            extra={
                "file": audio_path.name,
                "segments": len(result.segments),
                "dropped": len(result.dropped),
                "speaker": speaker,
                "elapsed_ms": round(elapsed_ms),
            },
        )
        return result

    def plan_chunks(
        self,
        segments: list[SpeechSegment],
        find_split: Callable[[int, int], int],
    ) -> list[SpeechSegment]:
        """Turn diarized segments into single-speaker chunks for transcription."""
        clips = self.segment_mode == SEGMENT_MODE_CLIPS
        max_chunk_ms = min(self.max_chunk_ms, _CLIPS_MAX_MS) if clips else self.max_chunk_ms

        chunks = resolve_speaker_overlaps(segments)
        chunks = split_long_segments(chunks, max_chunk_ms, find_split)
        # Parts of a split segment come out in a row: put a segment nested in
        # it back in time order, or the merge would see it after the last part.
        chunks.sort(key=lambda s: (s.start_ms, s.end_ms))
        chunks = merge_close_segments(chunks, self.max_gap_ms, max_chunk_ms)
        if self.min_chunk_ms > 0:
            chunks = absorb_short_chunks(chunks, self.min_chunk_ms, self.max_gap_ms)
            if clips:
                # An absorbed short chunk may have pushed its neighbour over the limit
                chunks = split_long_segments(chunks, _CLIPS_MAX_MS, find_split)
        return chunks

    def transcribe_segments(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe speech segments from an audio file.

        Loads audio into memory, merges close segments into larger chunks,
        slices each chunk as WAV bytes, and transcribes via remote API.

        A chunk the server fails to transcribe does not abort the run: it is
        reported in ``failed_chunks`` and the rest of the track is kept.

        Args:
            audio_path: Path to the full audio track (16kHz mono WAV).
            segments: Diarized speech segments with speaker labels.
            progress_callback: Called ``(completed_ms, total_ms)`` after each
                chunk finishes, weighted by chunk duration.

        Returns:
            TranscriptionResult with transcript segments (text, timestamps,
            speakers), dropped segments and failed chunks.

        Raises:
            SpeachesAPIError: If no chunk could be transcribed.
        """
        if not segments:
            return TranscriptionResult()

        t0 = time.perf_counter()

        # Load full audio into memory once
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw_frames = wf.readframes(n_frames)
        audio = _TrackAudio(raw_frames, sample_rate, sample_width, n_channels, n_frames)

        merged = self.plan_chunks(
            segments,
            lambda lo, hi: find_longest_pause(
                raw_frames, sample_rate, sample_width, n_channels, lo, hi
            ),
        )

        outcome = None
        if self.segment_mode == SEGMENT_MODE_CLIPS:
            outcome = self._transcribe_clips(audio_path, merged, segments)
            if outcome is not None and progress_callback is not None:
                # One request per track: there is no per-chunk progress to report
                total_ms = sum(s.duration_ms for s in merged)
                progress_callback(total_ms, total_ms)
        if outcome is None:
            outcome = self._transcribe_chunks(
                audio_path, audio, merged, segments, progress_callback
            )
        chunk_results, failed = outcome

        duplicates = self._mark_joint_duplicates(merged, chunk_results)

        results = TranscriptionResult(failed_chunks=[merged[i] for i in failed])
        for seg in (seg for segs in chunk_results for seg in segs):
            if seg.drop_reason is None:
                results.segments.append(seg)
            else:
                results.dropped.append(seg)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        speech_duration_ms = max((s.end_ms for s in segments), default=0)
        speech_rtf = elapsed_ms / speech_duration_ms if speech_duration_ms > 0 else 0
        logger.info(
            "Segment transcription completed",
            extra={
                "file": audio_path.name,
                "segments_in": len(segments),
                "chunks": len(merged),
                "chunks_failed": len(failed),
                "segments_out": len(results.segments),
                "segments_dropped": len(results.dropped),
                "joint_duplicates": duplicates,
                "speech_duration_ms": speech_duration_ms,
                "elapsed_ms": round(elapsed_ms),
                "speech_rtf": round(speech_rtf, 2),
            },
        )
        return results

    def _localize(
        self,
        seg: TranscriptSegment,
        chunk: SpeechSegment,
        offset_ms: int,
        segments: list[SpeechSegment],
    ) -> None:
        """Move a segment to track coordinates and assign its speaker.

        Text heard in the padding is pulled back inside the chunk, so it stays
        with this chunk's speaker rather than the neighbour's.
        """
        seg.start_ms = min(max(seg.start_ms + offset_ms, chunk.start_ms), chunk.end_ms)
        seg.end_ms = min(max(seg.end_ms + offset_ms, seg.start_ms), chunk.end_ms)
        # Assign speaker from the diarization chunk
        seg.speaker = self._find_speaker(seg.start_ms, seg.end_ms, segments, default=chunk.speaker)

    def _transcribe_clips(
        self,
        audio_path: Path,
        chunks: list[SpeechSegment],
        segments: list[SpeechSegment],
    ) -> tuple[list[list[TranscriptSegment]], list[int]] | None:
        """Transcribe all chunks in one request: the track plus the list of clips.

        Returns per-chunk segments and the indices of failed chunks, or None if
        the server has no clips endpoint and the per-chunk path must be used.
        """
        client = self.clients[0]
        pad_ms = min(self.chunk_padding_ms, _CLIPS_MAX_PAD_MS)
        try:
            if not client.supports_clips():
                raise ClipsEndpointUnavailable("Not in the server's API", endpoint=_CLIPS_PATH)
            found, failed = client.transcribe_clips(audio_path, chunks, self.language, pad_ms)
        except ClipsEndpointUnavailable:
            logger.warning(
                "Clips endpoint is not available, falling back to per-chunk requests",
                extra={"server": client.server_url},
            )
            return None

        # Only a decoding error is worth another attempt; the other reasons
        # describe the clip itself and would repeat.
        retry = sorted(i for i, reason in failed.items() if reason.startswith("error"))
        if retry:
            logger.warning(
                "Retrying failed clips", extra={"file": audio_path.name, "clips": len(retry)}
            )
            try:
                again, again_failed = client.transcribe_clips(
                    audio_path, [chunks[i] for i in retry], self.language, pad_ms
                )
            except SpeachesAPIError as e:
                logger.warning(
                    "Retry of failed clips failed",
                    extra={"file": audio_path.name, "error": str(e)},
                )
            else:
                for pos, i in enumerate(retry):
                    del failed[i]
                    if pos in again:
                        found[i] = again[pos]
                    elif pos in again_failed:
                        failed[i] = again_failed[pos]

        chunk_results: list[list[TranscriptSegment]] = [[] for _ in chunks]
        for i, clip_segments in found.items():
            for seg in clip_segments:
                self._localize(seg, chunks[i], 0, segments)
            chunk_results[i] = clip_segments

        failed_indices = []
        for i, reason in sorted(failed.items()):
            if reason in _CLIP_NO_TEXT_REASONS:
                # Nothing to transcribe in the clip: not a failure
                continue
            failed_indices.append(i)
            logger.warning(
                "Clip transcription failed",
                extra={
                    "file": audio_path.name,
                    "start_ms": chunks[i].start_ms,
                    "end_ms": chunks[i].end_ms,
                    "speaker": chunks[i].speaker,
                    "error": reason,
                },
            )

        if failed_indices and not found:
            raise SpeachesAPIError(
                f"No clip could be transcribed: {failed[failed_indices[0]]}",
                endpoint=f"{client.server_url}{_CLIPS_PATH}",
            )
        return chunk_results, failed_indices

    def _transcribe_chunks(
        self,
        audio_path: Path,
        audio: "_TrackAudio",
        merged: list[SpeechSegment],
        segments: list[SpeechSegment],
        progress_callback: Callable[[int, int], None] | None,
    ) -> tuple[list[list[TranscriptSegment]], list[int]]:
        """Transcribe chunks one request each, sliced from the track in memory.

        Returns per-chunk segments and the indices of failed chunks.
        """
        raw_frames = audio.raw_frames
        sample_rate = audio.sample_rate
        sample_width = audio.sample_width
        n_channels = audio.n_channels

        bounds = pad_chunks(merged, self.chunk_padding_ms, audio.duration_ms)

        total_ms = sum(s.duration_ms for s in merged)
        pbar = tqdm(total=total_ms, unit="ms", unit_scale=True, desc="  Transcribing", leave=False)

        frame_size = sample_width * n_channels

        def process_chunk(i: int, chunk: SpeechSegment) -> list[TranscriptSegment]:
            """Slice, transcribe, and localize a single chunk. Runs in a worker thread.

            Reads only from the shared, immutable ``raw_frames`` / ``segments``,
            so it is safe to run concurrently.
            """
            client = self.clients[i % len(self.clients)]
            audio_start_ms, audio_end_ms = bounds[i]

            # Slice audio in memory (read-only view into shared raw_frames)
            start_sample = audio_start_ms * sample_rate // 1000
            end_sample = audio_end_ms * sample_rate // 1000
            chunk_frames = raw_frames[start_sample * frame_size : end_sample * frame_size]

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(chunk_frames)

            transcript_segs = client.transcribe_bytes(buf.getvalue(), self.language)

            for seg in transcript_segs:
                self._localize(seg, chunk, audio_start_ms, segments)
            return transcript_segs

        # Send chunks concurrently: Speaches handles requests in a threadpool with
        # no cross-request batching, so parallel I/O is the only way to keep every
        # server busy. Results are collected by chunk index to stay deterministic.
        chunk_results: list[list[TranscriptSegment]] = [[] for _ in merged]
        failed: dict[int, SpeachesAPIError] = {}
        succeeded = 0
        max_workers = min(len(merged), self.max_inflight)
        completed_ms = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(merged)
            }
            try:
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        chunk_results[i] = future.result()
                        succeeded += 1
                    except SpeachesAPIError as e:
                        # Retries are already exhausted: give up on this chunk
                        # only, the rest of the track is still worth saving.
                        failed[i] = e
                        logger.warning(
                            "Chunk transcription failed",
                            extra={
                                "file": audio_path.name,
                                "start_ms": merged[i].start_ms,
                                "end_ms": merged[i].end_ms,
                                "speaker": merged[i].speaker,
                                "error": str(e),
                                "status_code": e.status_code,
                            },
                        )
                        if succeeded == 0 and len(failed) >= _MAX_FAILURES_WITHOUT_SUCCESS:
                            raise
                    pbar.update(merged[i].duration_ms)
                    if progress_callback is not None:
                        completed_ms += merged[i].duration_ms
                        progress_callback(completed_ms, total_ms)
            except Exception:
                # Drop still-queued chunks so the pool only waits out the ones
                # already running.
                for f in futures:
                    f.cancel()
                raise
            finally:
                pbar.close()

        if failed and succeeded == 0:
            raise next(iter(failed.values()))

        return chunk_results, sorted(failed)

    def _mark_joint_duplicates(
        self,
        chunks: list[SpeechSegment],
        chunk_results: list[list[TranscriptSegment]],
    ) -> int:
        """Mark a phrase repeated across the joint of two neighbouring chunks.

        Compares the last kept segment of a chunk with the first kept segment of
        the next one; the repeat is marked as dropped. Returns the count.
        """
        if self.dedup_similarity <= 0:
            return 0

        marked = 0
        for i in range(len(chunks) - 1):
            if chunks[i + 1].start_ms - chunks[i].end_ms > self.max_gap_ms:
                continue
            tail = next((s for s in reversed(chunk_results[i]) if s.drop_reason is None), None)
            head = next((s for s in chunk_results[i + 1] if s.drop_reason is None), None)
            if tail is None or head is None:
                continue
            tail_words = normalize_words(tail.text)
            head_words = normalize_words(head.text)
            if min(len(tail_words), len(head_words)) < self.dedup_min_words:
                continue
            ratio = SequenceMatcher(None, tail_words, head_words, autojunk=False).ratio()
            if ratio >= self.dedup_similarity:
                head.drop_reason = REASON_DUPLICATE
                marked += 1
                logger.debug(
                    "Filtered duplicate at chunk joint: '%s' (similarity=%.2f, start=%.1f)",
                    head.text[:80],
                    ratio,
                    head.start_ms / 1000,
                )
        return marked

    @staticmethod
    def _find_speaker(
        start_ms: int,
        end_ms: int,
        segments: list[SpeechSegment],
        default: str | None = None,
    ) -> str:
        """Find speaker with maximum time overlap."""
        best_speaker = default or "Unknown"
        best_overlap = 0

        for seg in segments:
            overlap = max(0, min(end_ms, seg.end_ms) - max(start_ms, seg.start_ms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker or "Unknown"

        return best_speaker


def failed_chunk_placeholders(result: TranscriptionResult) -> list[TranscriptSegment]:
    """Failed chunks as dropped segments without text, for the review journal."""
    return [
        TranscriptSegment(
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            text="",
            speaker=chunk.speaker,
            drop_reason=REASON_FAILED,
        )
        for chunk in result.failed_chunks
    ]
