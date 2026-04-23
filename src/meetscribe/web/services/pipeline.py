"""Pipeline service wrappers for web UI using remote speaches API."""

import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

from meetscribe import config
from meetscribe.config import AppConfig, get_config
from meetscribe.database import (
    delete_voiceprint,
    load_voiceprints,
    save_voiceprint,
)
from meetscribe.pipeline import (
    DiarizationPipeline,
    EmbeddingExtractor,
    Transcriber,
    TranscriptSegment,
    audio,
    enroll_samples,
)
from meetscribe.pipeline.models import collect_sample_segments
from meetscribe.team import TeamContext, resolve_team

logger = logging.getLogger(__name__)


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
            self._cfg = get_config()
        return self._cfg

    def _create_diarization(self, team_ctx: TeamContext) -> DiarizationPipeline:
        """Create a diarization pipeline with current voiceprints."""
        voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)
        return DiarizationPipeline(
            diarization_url=self.cfg.get_diarization_url(),
            embedding_url=self.cfg.get_embeddings_url(),
            voiceprints=voiceprints,
            threshold=self.cfg.embeddings.threshold,
            confident_gap=self.cfg.embeddings.confident_gap,
            min_threshold=self.cfg.embeddings.min_threshold,
            diarization_timeout=self.cfg.diarization.timeout,
            embedding_timeout=self.cfg.embeddings.timeout,
            min_duration_ms=self.cfg.embeddings.min_duration_ms,
            embedding_model=self.cfg.embeddings.model,
            diarization_model=self.cfg.diarization.model,
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
            no_speech_prob_threshold=self.cfg.transcription.no_speech_prob_threshold,
            avg_logprob_threshold=self.cfg.transcription.avg_logprob_threshold,
        )

    def extract_samples(
        self,
        track_paths: list[Path],
        track_diarize: dict[int, bool] | None = None,
        progress_callback: Any | None = None,
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

        total_steps = len(tracks_to_process) + 2  # connect + diarize per track + done

        # Step 1: Create pipeline
        if progress_callback:
            yield {"step": 1, "total": total_steps, "message": "Connecting to servers..."}
        team_ctx = self._resolve()
        try:
            diarization = self._create_diarization(team_ctx)

            step = 1
            all_samples = []
            for track_num, track_path in tracks_to_process:
                step += 1
                if progress_callback:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: Diarizing...",
                    }

                labeled_segments = diarization.diarize(track_path)
                if not labeled_segments:
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: No speech found",
                    }
                    continue

                speakers = {s.speaker for s in labeled_segments if s.speaker}

                # Extract audio samples
                emb_cfg = self.cfg.embeddings
                speaker_segments = collect_sample_segments(
                    labeled_segments,
                    min_duration_ms=emb_cfg.sample_min_duration_ms,
                    max_duration_ms=emb_cfg.sample_max_duration_ms,
                    ideal_ms=emb_cfg.sample_ideal_duration_ms,
                )

                for speaker_name, segs in speaker_segments.items():
                    segs.sort(key=lambda s: abs(s.duration_ms - emb_cfg.sample_ideal_duration_ms))
                    is_known = not speaker_name.startswith("Unknown")

                    for i, seg in enumerate(segs[: emb_cfg.max_samples_per_speaker]):
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

        team_ctx = self._resolve()
        try:
            enrolled_dir = team_ctx.enrolled_samples_dir / name
            avg_embedding, total_count, new_count = enroll_samples(
                extractor, sample_paths, enrolled_dir
            )

            yield {
                "step": 2,
                "total": total_steps,
                "message": f"Enrolling {name} from {total_count} samples ({new_count} new)...",
            }

            save_voiceprint(
                team_ctx.conn,
                team_ctx.id,
                name,
                avg_embedding,
                self.cfg.embeddings.model,
            )

            yield {
                "step": total_steps,
                "total": total_steps,
                "message": f"Enrolled {name}",
                "embedding_dim": len(avg_embedding),
            }
        finally:
            team_ctx.conn.close()

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
        named_count = sum(1 for i in range(len(track_paths)) if track_speakers.get(i + 1))
        diarized_count = len(track_paths) - named_count
        total_steps = named_count + diarized_count * 2 + 2

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

                if speaker_name:
                    # Named track: transcribe whole file with speaker
                    segs = transcriber.transcribe_file(track_path, speaker=speaker_name)
                else:
                    # Diarize track
                    segments = diarization.diarize(track_path)
                    if not segments:
                        yield {
                            "step": step,
                            "total": total_steps,
                            "message": f"Track {track_num}: No speech found",
                        }
                        continue

                    step += 1
                    seg_count = len(segments)
                    yield {
                        "step": step,
                        "total": total_steps,
                        "message": f"Track {track_num}: Transcribing {seg_count} segments...",
                        "progress": 0,
                    }

                    segs = transcriber.transcribe_segments(track_path, segments)

                for seg in segs:
                    seg.track_num = track_num
                all_segments.extend(segs)

            # Merge
            step += 1
            yield {"step": step, "total": total_steps, "message": "Merging transcripts..."}

            all_segments.sort(key=lambda x: x.start_ms)

            def format_segment(s: TranscriptSegment) -> str:
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
                "segments": [
                    {
                        "track_num": s.track_num or 1,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "speaker": s.speaker,
                        "text": s.text,
                    }
                    for s in all_segments
                ],
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
    from meetscribe.database import get_team

    from .session import get_session_service

    conn = get_session_service().conn
    name = team_name or "default"
    team = get_team(conn, name)
    if not team:
        return []
    voiceprints = load_voiceprints(conn, team["id"])
    return sorted(voiceprints.keys())


def remove_team_speaker(name: str, team_name: str | None = None) -> bool:
    """Remove a speaker from a team."""
    from meetscribe.database import get_team as db_get_team

    from .session import get_session_service

    conn = get_session_service().conn
    tname = team_name or "default"
    team = db_get_team(conn, tname)
    if not team:
        return False
    return delete_voiceprint(conn, team["id"], name)
