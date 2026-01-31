"""Pipeline service wrappers for web UI."""

import io
import os
import re
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
import torchaudio

from meetscribe import config
from meetscribe.pipeline import (
    EmbeddingExtractor,
    SpeakerIdentifier,
    SpectralClusterer,
    Transcriber,
    VADProcessor,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


FFMPEG_BIN = "ffmpeg"


def setup_environment() -> None:
    """Configure runtime environment."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"
    os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

    if sys.platform == "win32":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if site_packages.exists():
            for lib_dir in site_packages.iterdir():
                bin_dir = lib_dir / "bin"
                if bin_dir.exists():
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_environment_setup = False


def ensure_environment() -> None:
    """Ensure environment is set up (called once)."""
    global _environment_setup
    if not _environment_setup:
        setup_environment()
        _environment_setup = True


def probe_audio_tracks(file_path: Path) -> list[int]:
    """Return list of audio stream indices in the file."""
    result = subprocess.run(
        [FFMPEG_BIN, "-i", str(file_path), "-hide_banner"],
        capture_output=True,
        text=True,
    )
    indices = []
    for m in re.finditer(r"Stream #0:(\d+).*?: Audio:", result.stderr):
        indices.append(int(m.group(1)))
    return indices


def extract_audio_track(video_path: Path, output_path: Path, track_index: int) -> None:
    """Extract audio track from video."""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{track_index}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert audio file to 16kHz mono WAV."""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert audio: {result.stderr}")


class PipelineRunner:
    """Runs pipeline operations with progress callbacks."""

    def __init__(self):
        ensure_environment()
        self.vad: VADProcessor | None = None
        self.extractor: EmbeddingExtractor | None = None
        self.clusterer: SpectralClusterer | None = None
        self.identifier: SpeakerIdentifier | None = None
        self.transcriber: Transcriber | None = None

    def _ensure_diarization_models(self, max_speakers: int = 10, threshold: float = 0.7) -> None:
        """Load diarization models if not already loaded."""
        if self.vad is None:
            self.vad = VADProcessor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
        if self.extractor is None:
            self.extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
        if self.clusterer is None:
            self.clusterer = SpectralClusterer(min_speakers=2, max_speakers=max_speakers)
        if self.identifier is None:
            self.identifier = SpeakerIdentifier(
                config.VOICEPRINTS_DIR, self.extractor, threshold=threshold
            )

    def _ensure_transcription_model(self, model_size: str = "medium") -> None:
        """Load transcription model if not already loaded."""
        if self.transcriber is None or self.transcriber.model_size != model_size:
            self.transcriber = Transcriber(model_size=model_size, device=DEFAULT_DEVICE)

    def extract_samples(
        self,
        track_paths: list[Path],
        track_diarize: dict[int, bool] | None = None,
        progress_callback: Any | None = None,
        max_enrolled_samples: int = 10,
    ) -> Generator[dict, None, None]:
        """Extract speaker samples from tracks. Yields progress and sample info.

        Args:
            track_paths: List of track file paths
            track_diarize: Dict mapping track_num -> should_diarize. If False, skip track.
            progress_callback: If truthy, yield progress updates
            max_enrolled_samples: Max auto-samples to copy per known speaker to enrolled folder
        """
        # Filter tracks that need diarization
        tracks_to_process = []
        for idx, path in enumerate(track_paths):
            track_num = idx + 1
            if track_diarize is None or track_diarize.get(track_num, True):
                tracks_to_process.append((track_num, path))

        if not tracks_to_process:
            yield {"step": 1, "total": 1, "message": "No tracks need diarization", "samples": []}
            return

        total_steps = len(tracks_to_process) * 3 + 1  # load + (vad + embed + cluster) per track

        # Step 1: Load models
        if progress_callback:
            yield {"step": 1, "total": total_steps, "message": "Loading models..."}
        self._ensure_diarization_models()

        step = 1
        all_samples = []
        # Track samples per known speaker for enrolled folder limit
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

            speech_segs = self.vad.process(track_path)
            if not speech_segs:
                yield {
                    "step": step,
                    "total": total_steps,
                    "message": f"Track {track_num}: No speech found",
                }
                continue

            # Embeddings
            step += 1
            if progress_callback:
                seg_count = len(speech_segs)
                msg = f"Track {track_num}: Extracting embeddings ({seg_count} segments)..."
                yield {"step": step, "total": total_steps, "message": msg}

            embeddings = []
            for seg in speech_segs:
                audio = self.vad.extract_segment_audio(track_path, seg)
                embeddings.append(self.extractor.extract_from_tensor(audio))

            # Clustering
            step += 1
            if progress_callback:
                yield {
                    "step": step,
                    "total": total_steps,
                    "message": f"Track {track_num}: Clustering speakers...",
                }

            time_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
            diarized = self.clusterer.cluster(embeddings, time_segs)
            centroids = self.clusterer.get_cluster_centroids(diarized)
            matches = self.identifier.identify_clusters(centroids)

            cluster_names = {cid: m.name for cid, m in matches.items()}
            cluster_is_known = {cid: m.is_known for cid, m in matches.items()}

            # Assign cluster IDs to segments
            for seg, diar in zip(speech_segs, diarized):
                seg.cluster_id = diar.cluster_id

            # Extract samples (longest segments per cluster)
            waveform, sr = torchaudio.load(str(track_path))
            min_duration_ms = 3000
            max_samples_per_cluster = 5

            for cluster_id in centroids:
                cluster_segs = [
                    s for s in speech_segs if getattr(s, "cluster_id", None) == cluster_id
                ]
                cluster_segs = [s for s in cluster_segs if s.duration_ms >= min_duration_ms]
                cluster_segs.sort(key=lambda s: s.duration_ms, reverse=True)

                is_known = cluster_is_known.get(cluster_id, False)
                speaker_name = cluster_names[cluster_id] if is_known else None

                for i, seg in enumerate(cluster_segs[:max_samples_per_cluster]):
                    start = int(seg.start_ms * sr / 1000)
                    end = int(seg.end_ms * sr / 1000)
                    segment_audio = waveform[:, start:end]

                    # Convert to bytes
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, segment_audio, sr, format="wav")
                    audio_bytes = buffer.getvalue()

                    sample_info = {
                        "track_num": track_num,
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_names[cluster_id],
                        "duration_ms": int(seg.duration_ms),
                        "audio_bytes": audio_bytes,
                        "filename": f"track{track_num}_c{cluster_id}_s{i}.wav",
                        "is_known": is_known,
                        "known_speaker_name": speaker_name,
                    }
                    all_samples.append(sample_info)

                    # Copy identified samples to enrolled folder (up to limit)
                    if is_known and speaker_name:
                        current_count = known_speaker_counts.get(speaker_name, 0)
                        if current_count < max_enrolled_samples:
                            import time as time_module

                            enrolled_dir = config.ENROLLED_SAMPLES_DIR / speaker_name
                            enrolled_dir.mkdir(parents=True, exist_ok=True)
                            timestamp = int(time_module.time() * 1000)
                            dest = enrolled_dir / f"auto_{timestamp}_{track_num}_{cluster_id}_{i}.wav"
                            with open(dest, "wb") as f:
                                f.write(audio_bytes)
                            known_speaker_counts[speaker_name] = current_count + 1

            yield {
                "step": step,
                "total": total_steps,
                "message": f"Track {track_num}: Found {len(centroids)} speakers",
                "speakers": [
                    {"cluster_id": cid, "name": name, "is_known": matches[cid].is_known}
                    for cid, name in cluster_names.items()
                ],
            }

        yield {"step": total_steps, "total": total_steps, "message": "Done", "samples": all_samples}

    def enroll_speaker(
        self, name: str, sample_paths: list[Path], progress_callback: Any | None = None
    ) -> Generator[dict, None, None]:
        """Enroll a speaker from samples. Yields progress."""
        total_steps = 2

        yield {"step": 1, "total": total_steps, "message": "Loading models..."}
        self._ensure_diarization_models()

        sample_count = len(sample_paths)
        yield {
            "step": 2,
            "total": total_steps,
            "message": f"Enrolling {name} from {sample_count} samples...",
        }

        voiceprint = self.identifier.enroll(name, sample_paths)
        yield {
            "step": total_steps,
            "total": total_steps,
            "message": f"Enrolled {name}",
            "embedding_dim": len(voiceprint),
        }

    def transcribe(
        self,
        track_paths: list[Path],
        track_speakers: dict[int, str | None],
        model_size: str = "medium",
        language: str = "ru",
        progress_callback: Any | None = None,
        progress_queue: Any | None = None,
    ) -> Generator[dict, None, None]:
        """Transcribe tracks. Yields progress and results.

        Args:
            progress_queue: Optional queue to put real-time progress updates
                           (for SSE streaming during long transcription steps)
        """
        # load models + (process + transcribe) per track + merge
        total_steps = len(track_paths) * 2 + 2

        yield {"step": 1, "total": total_steps, "message": "Loading models..."}
        self._ensure_diarization_models()
        self._ensure_transcription_model(model_size)

        step = 1
        all_segments = []

        for track_idx, track_path in enumerate(track_paths):
            track_num = track_idx + 1
            speaker_name = track_speakers.get(track_num)

            step += 1
            yield {
                "step": step,
                "total": total_steps,
                "message": f"Track {track_num}: VAD and preprocessing...",
            }

            speech_segs = self.vad.process(track_path)
            if not speech_segs:
                yield {
                    "step": step,
                    "total": total_steps,
                    "message": f"Track {track_num}: No speech found",
                }
                continue

            if speaker_name:
                # Named track: label all with speaker name
                vad_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
                speaker_segs = [(s.start_ms, s.end_ms, speaker_name) for s in speech_segs]
            else:
                # Diarize
                embeddings = []
                for seg in speech_segs:
                    audio = self.vad.extract_segment_audio(track_path, seg)
                    embeddings.append(self.extractor.extract_from_tensor(audio))

                time_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
                diarized = self.clusterer.cluster(embeddings, time_segs)
                centroids = self.clusterer.get_cluster_centroids(diarized)
                matches = self.identifier.identify_clusters(centroids)
                cluster_names = {cid: m.name for cid, m in matches.items()}

                vad_segs = time_segs
                speaker_segs = [
                    (d.start_ms, d.end_ms, cluster_names[d.cluster_id]) for d in diarized
                ]

            step += 1
            seg_count = len(speech_segs)
            yield {
                "step": step,
                "total": total_steps,
                "message": f"Track {track_num}: Transcribing {seg_count} segments...",
                "progress": 0,
            }

            # Progress callback that puts updates into the queue for real-time SSE
            last_progress = [0]  # Use list to allow mutation in closure

            def on_progress(percent: int):
                # Only send updates every 5% to avoid flooding
                if percent >= last_progress[0] + 5 or percent == 100:
                    last_progress[0] = percent
                    if progress_queue is not None:
                        progress_queue.put((
                            "item",
                            {
                                "step": step,
                                "total": total_steps,
                                "message": f"Track {track_num}: Transcribing... {percent}%",
                                "progress": percent,
                            },
                        ))

            segs = self.transcriber.transcribe_vad_segments(
                track_path, vad_segs, speaker_segs, language, progress_callback=on_progress
            )
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


# Singleton pipeline runner
_pipeline_runner: PipelineRunner | None = None


def get_pipeline_runner() -> PipelineRunner:
    """Get the pipeline runner singleton."""
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner()
    return _pipeline_runner


def list_global_speakers() -> list[str]:
    """List enrolled global speakers."""
    if not config.VOICEPRINTS_DIR.exists():
        return []
    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_DIR, extractor)
    return identifier.list_speakers()


def remove_global_speaker(name: str) -> bool:
    """Remove a global speaker."""
    if not config.VOICEPRINTS_DIR.exists():
        return False
    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_DIR, extractor)
    return identifier.remove(name)
