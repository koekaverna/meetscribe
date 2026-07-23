"""Pipeline service wrappers for web UI using remote speaches API."""

import logging
import shutil
import tempfile
import wave
from collections.abc import Generator
from pathlib import Path
from typing import Any

from meetscribe import config
from meetscribe.config import AppConfig, get_config
from meetscribe.database import (
    delete_voiceprint,
    get_db,
    get_team,
    get_voiceprint,
    list_voiceprint_meta,
    load_voiceprints,
    rename_voiceprint,
    save_voiceprint,
)
from meetscribe.pipeline import (
    DiarizationPipeline,
    EmbeddingExtractor,
    Transcriber,
    TranscriptSegment,
    audio,
    compute_voiceprint,
    enroll_samples,
)
from meetscribe.pipeline.models import collect_sample_segments, filter_segments_by_speaker
from meetscribe.team import TeamContext, resolve_team

from ..models import GlobalSpeaker, SpeakerSample

logger = logging.getLogger(__name__)


def _make_extractor(cfg: AppConfig) -> EmbeddingExtractor:
    """Create an embedding extractor from app config."""
    return EmbeddingExtractor(
        cfg.get_embeddings_url(),
        cfg.embeddings.timeout,
        cfg.embeddings.min_duration_ms,
        model=cfg.embeddings.model,
    )


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
        voiceprints = load_voiceprints(get_db(), team_ctx.id)
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
            max_inflight=self.cfg.transcription.max_inflight or None,
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

    def enroll_speaker(
        self, name: str, sample_paths: list[Path], progress_callback: Any | None = None
    ) -> Generator[dict, None, None]:
        """Enroll a speaker from samples. Yields progress."""
        total_steps = 2

        yield {"step": 1, "total": total_steps, "message": "Connecting to server..."}
        extractor = _make_extractor(self.cfg)

        team_ctx = self._resolve()
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
            get_db(),
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

    def transcribe(
        self,
        track_paths: list[Path],
        track_speakers: dict[int, str | None],
        language: str | None = None,
        track_open_space: dict[int, bool] | None = None,
        progress_callback: Any | None = None,
    ) -> Generator[dict, None, None]:
        """Transcribe tracks. Yields progress and results.

        A named track flagged in track_open_space is an open-space mic recording:
        it is diarized and only the assigned speaker's segments are kept, dropping
        other people's voices the mic picked up. Auto-diarized tracks keep everyone.
        """
        effective_language = language or self.cfg.transcription.language
        open_space = track_open_space or {}

        def _diarizes(track_num: int, speaker_name: str | None) -> bool:
            # Auto-diarize tracks, plus named mic tracks filtered to one speaker.
            return not speaker_name or open_space.get(track_num, False)

        diarized_count = sum(
            1 for i in range(len(track_paths)) if _diarizes(i + 1, track_speakers.get(i + 1))
        )
        named_count = len(track_paths) - diarized_count
        total_steps = named_count + diarized_count * 2 + 2

        yield {"step": 1, "total": total_steps, "message": "Connecting to servers..."}
        team_ctx = self._resolve()
        diarization = self._create_diarization(team_ctx)
        transcriber = self._create_transcriber(effective_language)

        step = 1
        all_segments = []

        for track_idx, track_path in enumerate(track_paths):
            track_num = track_idx + 1
            speaker_name = track_speakers.get(track_num)
            filter_to_speaker = bool(speaker_name) and open_space.get(track_num, False)

            step += 1
            yield {
                "step": step,
                "total": total_steps,
                "message": f"Track {track_num}: Processing...",
            }

            if speaker_name and not filter_to_speaker:
                # Named track: transcribe whole file with speaker
                segs = transcriber.transcribe_file(track_path, speaker=speaker_name)
            else:
                # Diarize track (auto-diarize, or an open-space mic filtered to its speaker)
                segments = diarization.diarize(track_path)
                if filter_to_speaker and speaker_name:
                    found = sorted({s.speaker for s in segments if s.speaker})
                    segments = filter_segments_by_speaker(segments, speaker_name)
                    if not segments and found:
                        # A silent skip here hides a mistyped name: the track has
                        # voices, just none labeled with the assigned speaker.
                        raise ValueError(
                            f"Track {track_num}: open-space speaker '{speaker_name}' does not"
                            f" match any voice in the track (found: {', '.join(found)})."
                            " Check the assigned name in the Configure step."
                        )
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


# Singleton pipeline runner (per team)
_pipeline_runners: dict[str, PipelineRunner] = {}


def get_pipeline_runner(team_name: str | None = None) -> PipelineRunner:
    """Get the pipeline runner singleton for a team."""
    key = team_name or "default"
    if key not in _pipeline_runners:
        _pipeline_runners[key] = PipelineRunner(team_name)
    return _pipeline_runners[key]


def _safe_speaker_dir(team_name: str, name: str) -> Path | None:
    """Samples dir for an enrolled speaker, or None if the name could escape it."""
    # Path("..").name == ".." — the explicit check is not redundant
    if not name or name == ".." or name != Path(name).name:
        return None
    return config.get_team_enrolled_dir(team_name) / name


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() * 1000 // rate if rate else 0
    except (wave.Error, EOFError, OSError):
        # A corrupt sample must not take down the whole dashboard
        return 0


def _resolve_speaker(name: str, team_name: str | None) -> tuple[int, Path]:
    """Return (team_id, samples_dir) for an enrolled speaker.

    Raises LookupError if the team or speaker doesn't exist or the name is unsafe.
    """
    conn = get_db()
    tname = team_name or "default"
    team = get_team(conn, tname)
    samples_dir = _safe_speaker_dir(tname, name)
    if team is None or samples_dir is None or get_voiceprint(conn, team["id"], name) is None:
        raise LookupError(f"Speaker not found: {name}")
    return team["id"], samples_dir


def _sample_path(samples_dir: Path, filename: str) -> Path:
    """Resolve a sample filename inside a speaker dir. Raises LookupError if invalid."""
    if filename != Path(filename).name or not filename.endswith(".wav"):
        raise LookupError(f"Sample not found: {filename}")
    path = samples_dir / filename
    if not path.is_file():
        raise LookupError(f"Sample not found: {filename}")
    return path


def list_team_speakers(team_name: str | None = None) -> list[GlobalSpeaker]:
    """List enrolled speakers for a team with voiceprint quality stats."""
    conn = get_db()
    tname = team_name or "default"
    team = get_team(conn, tname)
    if not team:
        return []
    speakers = []
    for row in list_voiceprint_meta(conn, team["id"]):
        samples_dir = _safe_speaker_dir(tname, row["name"])
        wavs = sorted(samples_dir.glob("*.wav")) if samples_dir and samples_dir.is_dir() else []
        speakers.append(
            GlobalSpeaker(
                name=row["name"],
                model=row["model"],
                sample_count=len(wavs),
                total_duration_ms=sum(_wav_duration_ms(p) for p in wavs),
                created_at=row["created_at"],
            )
        )
    return speakers


def list_speaker_samples(name: str, team_name: str | None = None) -> list[SpeakerSample]:
    """List enrolled samples of a speaker. Raises LookupError if the speaker is unknown."""
    _, samples_dir = _resolve_speaker(name, team_name)
    wavs = sorted(samples_dir.glob("*.wav")) if samples_dir.is_dir() else []
    return [SpeakerSample(filename=p.name, duration_ms=_wav_duration_ms(p)) for p in wavs]


def get_speaker_sample_path(name: str, filename: str, team_name: str | None = None) -> Path:
    """Path to an enrolled sample. Raises LookupError if speaker or sample is unknown."""
    _, samples_dir = _resolve_speaker(name, team_name)
    return _sample_path(samples_dir, filename)


def delete_speaker_sample(name: str, filename: str, team_name: str | None = None) -> None:
    """Delete one enrolled sample and recompute the voiceprint from the rest.

    Raises LookupError if speaker or sample is unknown, ValueError for the last
    sample (the voiceprint would have no source data — delete the speaker instead).
    """
    team_id, samples_dir = _resolve_speaker(name, team_name)
    target = _sample_path(samples_dir, filename)
    remaining = [p for p in sorted(samples_dir.glob("*.wav")) if p != target]
    if not remaining:
        raise ValueError("Cannot delete the last sample — delete the speaker instead")
    cfg = get_config()
    # Recompute before unlinking: if the embeddings API fails, nothing has changed
    embedding = compute_voiceprint(_make_extractor(cfg), remaining)
    target.unlink()
    save_voiceprint(get_db(), team_id, name, embedding, cfg.embeddings.model)


def rename_team_speaker(old: str, new: str, team_name: str | None = None) -> None:
    """Rename an enrolled speaker: voiceprint row + samples dir.

    Raises LookupError if the speaker is unknown, ValueError if the new name is
    invalid, FileExistsError if the new name is already taken.
    """
    team_id, old_dir = _resolve_speaker(old, team_name)
    tname = team_name or "default"
    new = new.strip()
    new_dir = _safe_speaker_dir(tname, new)
    if new_dir is None:
        raise ValueError(f"Invalid speaker name: {new!r}")
    if new == old:
        return
    # A leftover dir (pre-cleanup deletes) counts as a collision too — renaming
    # onto it would silently mix another speaker's samples into this voiceprint
    if get_voiceprint(get_db(), team_id, new) is not None or new_dir.exists():
        raise FileExistsError(f"Speaker '{new}' already exists")
    # Dir first: a failed DB update can undo the dir rename, but a failed dir
    # rename after a committed DB update would leave the samples orphaned
    if old_dir.is_dir():
        old_dir.rename(new_dir)
    try:
        rename_voiceprint(get_db(), team_id, old, new)
    except Exception:
        if new_dir.is_dir():
            new_dir.rename(old_dir)
        raise


def remove_team_speaker(name: str, team_name: str | None = None) -> bool:
    """Remove a speaker from a team: voiceprint + enrolled samples dir."""
    conn = get_db()
    tname = team_name or "default"
    team = get_team(conn, tname)
    if not team:
        return False
    if not delete_voiceprint(conn, team["id"], name):
        return False
    samples_dir = _safe_speaker_dir(tname, name)
    if samples_dir is not None and samples_dir.is_dir():
        shutil.rmtree(samples_dir, ignore_errors=True)
    return True
