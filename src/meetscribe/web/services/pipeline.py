"""Pipeline service wrappers for web UI using remote speaches API."""

import logging
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from meetscribe import config
from meetscribe.database import (
    delete_voiceprint,
    load_voiceprints,
    save_voiceprint,
)
from meetscribe.pipeline import (
    DiarizationPipeline,
    EmbeddingExtractor,
    SpeechSegment,
    Transcriber,
    audio,
)
from meetscribe.servers import AppConfig, load_config
from meetscribe.team import TeamContext, resolve_team

logger = logging.getLogger(__name__)


def _load_config() -> AppConfig:
    """Load the application configuration."""
    return load_config(config.CONFIG_FILE)


class PipelineRunner:
    """Runs pipeline operations with progress callbacks via remote speaches API.

    Does NOT hold a persistent DB connection — resolves team context
    on each call so it's safe to use from background threads.
    """

    def __init__(self, team_name: str | None = None):
        self.team_name = team_name
        self._cfg: AppConfig | None = None

    def _resolve(self) -> TeamContext:
        """Resolve team context (opens a fresh DB connection for this thread)."""
        return resolve_team(self.team_name)

    @property
    def cfg(self) -> AppConfig:
        if self._cfg is None:
            self._cfg = _load_config()
        return self._cfg

    def _create_diarization(self, team_ctx: TeamContext) -> DiarizationPipeline:
        """Create a diarization pipeline with current voiceprints."""
        voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)
        return DiarizationPipeline(
            vad_url=self.cfg.get_vad_url(),
            embedding_url=self.cfg.get_embeddings_url(),
            voiceprints=voiceprints,
            threshold=self.cfg.embeddings.threshold,
            vad_timeout=self.cfg.vad.timeout,
            embedding_timeout=self.cfg.embeddings.timeout,
            min_duration_ms=self.cfg.embeddings.min_duration_ms,
            unknown_cluster_threshold=self.cfg.embeddings.unknown_cluster_threshold,
            confident_gap=self.cfg.embeddings.confident_gap,
            min_threshold=self.cfg.embeddings.min_threshold,
            max_workers=self.cfg.embeddings.max_workers,
            embedding_model=self.cfg.embeddings.model,
        )

    def _create_transcriber(self, language: str) -> Transcriber:
        """Create a transcriber with remote servers."""
        return Transcriber(
            self.cfg.get_transcription_urls(),
            language=language,
            timeout=self.cfg.transcription.timeout,
            model=self.cfg.transcription.model,
            max_gap_ms=self.cfg.transcription.max_gap_ms,
            max_chunk_ms=self.cfg.transcription.max_chunk_ms,
        )

    def extract_samples(
        self,
        track_paths: list[Path],
        track_diarize: dict[int, bool] | None = None,
        progress_callback: Any | None = None,
        max_enrolled_samples: int = 10,
    ) -> Generator[dict, None, None]:
        """Extract speaker samples from tracks. Yields progress and sample info."""
        # Filter tracks that need diarization
        tracks_to_process = []
        for idx, path in enumerate(track_paths):
            track_num = idx + 1
            if track_diarize is None or track_diarize.get(track_num, True):
                tracks_to_process.append((track_num, path))

        if not tracks_to_process:
            yield {"step": 1, "total": 1, "message": "No tracks need diarization", "samples": []}
            return

        total_steps = len(tracks_to_process) * 3 + 1  # connect + (vad + embed + id) per track

        # Step 1: Create pipeline
        if progress_callback:
            yield {"step": 1, "total": total_steps, "message": "Connecting to servers..."}
        team_ctx = self._resolve()
        try:
            diarization = self._create_diarization(team_ctx)

            step = 1
            all_samples = []
            known_speaker_counts: dict[str, int] = {}

            for track_num, track_path in tracks_to_process:
                # VAD
                step += 1
                if progress_callback:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: Running VAD...",
                    }

                segments = diarization.vad.detect(track_path)
                if not segments:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: No speech found",
                    }
                    continue

                # Embeddings
                step += 1
                if progress_callback:
                    seg_count = len(segments)
                    msg = f"Track {track_num}: Extracting embeddings ({seg_count} segments)..."
                    yield {"step": step, "total": total_steps, "message": msg}

                segments_with_emb = diarization.embeddings.extract_segments(
                    track_path, segments, diarization.max_workers
                )

                # Identification
                step += 1
                if progress_callback:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: Identifying speakers...",
                    }

                labeled_segments = diarization.identifier.identify_segments(segments_with_emb)
                speakers = {s.speaker for s in labeled_segments if s.speaker}

                # Extract audio samples (longest segments per speaker)
                min_duration_ms = 3000
                max_samples_per_speaker = 5

                # Group segments by speaker
                speaker_segments: dict[str, list[SpeechSegment]] = {}
                for seg in labeled_segments:
                    if seg.speaker and seg.duration_ms >= min_duration_ms:
                        speaker_segments.setdefault(seg.speaker, []).append(seg)

                for speaker_name, segs in speaker_segments.items():
                    segs.sort(key=lambda s: s.duration_ms, reverse=True)
                    is_known = not speaker_name.startswith("Unknown")

                    for i, seg in enumerate(segs[:max_samples_per_speaker]):
                        # Extract segment audio via FFmpeg
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False, dir=config.TMP_DIR
                        ) as tmp:
                            chunk_path = Path(tmp.name)

                        try:
                            audio.extract_segment(track_path, chunk_path, seg.start_ms, seg.end_ms)
                            audio_bytes = chunk_path.read_bytes()
                        finally:
                            chunk_path.unlink(missing_ok=True)

                        sample_info = {
                            "track_num": track_num,
                            "cluster_id": hash(speaker_name) % 1000,
                            "cluster_name": speaker_name,
                            "duration_ms": int(seg.duration_ms),
                            "audio_bytes": audio_bytes,
                            "filename": f"track{track_num}_{speaker_name}_s{i}.wav",
                            "is_known": is_known,
                            "known_speaker_name": speaker_name if is_known else None,
                        }
                        all_samples.append(sample_info)

                        # Copy identified samples to enrolled folder (up to limit)
                        if is_known:
                            current_count = known_speaker_counts.get(speaker_name, 0)
                            if current_count < max_enrolled_samples:
                                enrolled_dir = team_ctx.enrolled_samples_dir / speaker_name
                                enrolled_dir.mkdir(parents=True, exist_ok=True)
                                timestamp = int(time.time() * 1000)
                                dest = enrolled_dir / f"auto_{timestamp}_{track_num}_{i}.wav"
                                dest.write_bytes(audio_bytes)
                                known_speaker_counts[speaker_name] = current_count + 1

                yield {
                    "step": step,
                    "total": total_steps,
                    "message": f"Track {track_num}: Found {len(speakers)} speakers",
                    "speakers": [
                        {
                            "name": name,
                            "is_known": not name.startswith("Unknown"),
                        }
                        for name in sorted(speakers)
                    ],
                }

            yield {
                "step": total_steps,
                "total": total_steps,
                "message": "Done",
                "samples": all_samples,
            }
        finally:
            team_ctx.conn.close()

    def enroll_speaker(
        self, name: str, sample_paths: list[Path], progress_callback: Any | None = None
    ) -> Generator[dict, None, None]:
        """Enroll a speaker from samples. Yields progress."""
        total_steps = 2

        yield {"step": 1, "total": total_steps, "message": "Connecting to server..."}
        extractor = EmbeddingExtractor(
            self.cfg.get_embeddings_url(),
            self.cfg.embeddings.timeout,
            self.cfg.embeddings.min_duration_ms,
            model=self.cfg.embeddings.model,
        )

        sample_count = len(sample_paths)
        yield {
            "step": 2,
            "total": total_steps,
            "message": f"Enrolling {name} from {sample_count} samples...",
        }

        # Extract embeddings and average them
        embeddings: list[list[float]] = []
        for path in sample_paths:
            emb = extractor.extract_from_file(path)
            embeddings.append(emb)

        avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
        team_ctx = self._resolve()
        try:
            save_voiceprint(
                team_ctx.conn,
                team_ctx.id,
                name,
                avg_embedding,
                self.cfg.transcription.model,
            )
        finally:
            team_ctx.conn.close()

        yield {
            "step": total_steps,
            "total": total_steps,
            "message": f"Enrolled {name}",
            "embedding_dim": len(avg_embedding),
        }

    def transcribe(
        self,
        track_paths: list[Path],
        track_speakers: dict[int, str | None],
        language: str | None = None,
        progress_callback: Any | None = None,
        progress_queue: Any | None = None,
    ) -> Generator[dict, None, None]:
        """Transcribe tracks. Yields progress and results."""
        effective_language = language or self.cfg.transcription.language
        total_steps = len(track_paths) * 2 + 2

        yield {"step": 1, "total": total_steps, "message": "Connecting to servers..."}
        team_ctx = self._resolve()
        try:
            diarization = self._create_diarization(team_ctx)
            transcriber = self._create_transcriber(effective_language)

            step = 1
            all_segments = []

            for track_idx, track_path in enumerate(track_paths):
                track_num = track_idx + 1
                speaker_name = track_speakers.get(track_num)

                step += 1
                yield {
                    "step": step,
                    "total": total_steps,
                    "message": f"Track {track_num}: Processing...",
                }

                # Always run VAD first — sending large files to the server fails
                segments = diarization.vad.detect(track_path)
                if not segments:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: No speech found",
                    }
                    continue

                if speaker_name:
                    # Named track: assign speaker to all VAD segments, then transcribe
                    for seg in segments:
                        seg.speaker = speaker_name
                    segs = transcriber.transcribe_segments(track_path, segments)
                    all_segments.extend(segs)
                else:
                    # Diarize: embeddings -> identification (VAD already done above)
                    segments_with_emb = diarization.embeddings.extract_segments(
                        track_path, segments, diarization.max_workers
                    )
                    labeled = diarization.identifier.identify_segments(segments_with_emb)

                    step += 1
                    seg_count = len(labeled)
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: Transcribing {seg_count} segments...",
                        "progress": 0,
                    }

                    segs = transcriber.transcribe_segments(track_path, labeled)
                    all_segments.extend(segs)

            # Merge
            step += 1
            yield {"step": step, "total": total_steps, "message": "Merging transcripts..."}

            all_segments.sort(key=lambda x: x.start_ms)

            def format_segment(s):
                mins = s.start_ms // 60000
                secs = (s.start_ms // 1000) % 60
                speaker = s.speaker or "Unknown"
                return f"**[{mins:02d}:{secs:02d}] {speaker}:** {s.text}"

            dialogue = "\n\n".join(format_segment(s) for s in all_segments)

            yield {
                "step": total_steps,
                "total": total_steps,
                "message": "Done",
                "transcript": dialogue,
                "segment_count": len(all_segments),
            }
        finally:
            team_ctx.conn.close()


# Singleton pipeline runner (per team)
_pipeline_runners: dict[str, PipelineRunner] = {}


def get_pipeline_runner(team_name: str | None = None) -> PipelineRunner:
    """Get the pipeline runner singleton for a team."""
    key = team_name or "default"
    if key not in _pipeline_runners:
        _pipeline_runners[key] = PipelineRunner(team_name)
    return _pipeline_runners[key]


def list_team_speakers(team_name: str | None = None) -> list[str]:
    """List enrolled speakers for a team."""
    team_ctx = resolve_team(team_name)
    try:
        voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)
        return sorted(voiceprints.keys())
    finally:
        team_ctx.conn.close()


def remove_team_speaker(name: str, team_name: str | None = None) -> bool:
    """Remove a speaker from a team."""
    team_ctx = resolve_team(team_name)
    try:
        return delete_voiceprint(team_ctx.conn, team_ctx.id, name)
    finally:
        team_ctx.conn.close()
