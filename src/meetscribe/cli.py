#!/usr/bin/env python3
"""
MeetScribe - Meeting Transcription with Speaker Diarization

Supports video files with multiple audio tracks and audio files directly.
Per-track mode: diarize (default) or assign speaker name via --trackN.

Usage:
    meetscribe enroll "John" samples/*.wav
    meetscribe transcribe video.mp4 -o output.md
    meetscribe transcribe video.mp4 -o output.md --track1 "Host"
    meetscribe transcribe audio1.wav audio2.wav -o output.md --track1 "Name"
    meetscribe extract video.mp4 -o output_dir/
    meetscribe list-speakers
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import colorama
import torch
import torchaudio

from . import config
from .pipeline import (
    EmbeddingExtractor,
    SpeakerIdentifier,
    SpectralClusterer,
    Transcriber,
    VADProcessor,
)

# Enable ANSI colors on Windows
colorama.just_fix_windows_console()

# === Colors ===
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"
C_BLUE = "\033[94m"

# Ensure directories exist
config.ensure_dirs()

# === Logging setup ===
_log_file = config.LOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])
logging.captureWarnings(True)
warnings.filterwarnings("default")

# Defaults
DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "ru"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level state (populated by setup_environment)
_FFMPEG_BIN = None


def setup_environment():
    """Configure runtime environment: env vars, CUDA DLLs, Whisper FFmpeg patch."""
    global _FFMPEG_BIN

    # Windows: disable symlinks, use COPY strategy
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"
    os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

    # Add NVIDIA DLLs to PATH for Windows CUDA
    if sys.platform == "win32":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if site_packages.exists():
            for lib_dir in site_packages.iterdir():
                bin_dir = lib_dir / "bin"
                if bin_dir.exists():
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    # Patch Whisper to use imageio-ffmpeg binary
    import imageio_ffmpeg

    _FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

    from subprocess import CalledProcessError, run

    import numpy as np

    def load_audio_patched(file: str, sr: int = 16000):
        cmd = [
            _FFMPEG_BIN,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    import whisper.audio

    whisper.audio.load_audio = load_audio_patched


# === Helpers ===


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


@contextmanager
def step(num: int, total: int, desc: str, emoji: str = "🔄"):
    """Context manager that prints step header and elapsed time on completion."""
    print(f"\n{C_CYAN}[{num}/{total}]{C_RESET} {emoji}  {C_BOLD}{desc}{C_RESET}")
    t = time.time()
    yield
    elapsed = time.time() - t
    print(f"  {C_GREEN}✅ Done {C_DIM}({_format_elapsed(elapsed)}){C_RESET}")


@contextmanager
def substep(desc: str, emoji: str = "🔄"):
    """Context manager that prints sub-step and elapsed time."""
    print(f"    {emoji}  {desc}")
    t = time.time()
    yield
    elapsed = time.time() - t
    print(f"      {C_DIM}({_format_elapsed(elapsed)}){C_RESET}")


def ok(msg: str) -> None:
    """Print a success sub-status line."""
    print(f"  {C_GREEN}✔{C_RESET}  {msg}")


def warn(msg: str) -> None:
    """Print a warning sub-status line."""
    print(f"  {C_YELLOW}⚠️{C_RESET}  {msg}")


def info(msg: str) -> None:
    """Print an info sub-status line."""
    print(f"  {C_DIM}→{C_RESET}  {msg}")


def run_cmd(cmd: list[str], desc: str) -> None:
    """Run command with error handling."""
    info(desc)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed: {result.stderr}")


def extract_audio(video_path: Path, output_path: Path, track_index: int) -> Path:
    """Extract audio track from video."""
    run_cmd(
        [
            _FFMPEG_BIN,
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
        ],
        f"Extracting track {track_index}",
    )
    return output_path


def probe_audio_tracks(file_path: Path) -> list[int]:
    """Return list of audio stream indices in the file using ffmpeg -i."""
    result = subprocess.run(
        [_FFMPEG_BIN, "-i", str(file_path), "-hide_banner"],
        capture_output=True,
        text=True,
    )
    # ffmpeg -i exits with 1 when no output specified, but still prints stream info to stderr
    indices = []
    for m in re.finditer(r"Stream #0:(\d+).*?: Audio:", result.stderr):
        indices.append(int(m.group(1)))
    return indices


def save_unknown_samples(
    audio_path: Path,
    segments: list,
    cluster_names: dict[int, str],
    date_str: str,
    min_duration_ms: int = 3000,
    max_samples: int = 5,
) -> None:
    """Save audio samples for unknown speakers."""
    unknown_clusters = {cid for cid, name in cluster_names.items() if name.startswith("Unknown")}
    if not unknown_clusters:
        return

    waveform, sr = torchaudio.load(str(audio_path))
    samples_dir = config.SAMPLES_DIR / "unknown"

    for cluster_id in unknown_clusters:
        cluster_dir = samples_dir / f"{date_str}-speaker{cluster_id}"

        # Filter and sort by duration (longest first)
        cluster_segs = [s for s in segments if getattr(s, "cluster_id", None) == cluster_id]
        cluster_segs = [s for s in cluster_segs if s.duration_ms >= min_duration_ms]
        cluster_segs.sort(key=lambda s: s.duration_ms, reverse=True)

        if not cluster_segs:
            warn(f"No samples >= {min_duration_ms / 1000:.0f}s for {cluster_names[cluster_id]}")
            continue

        cluster_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(cluster_segs[:max_samples]):
            start = int(seg.start_ms * sr / 1000)
            end = int(seg.end_ms * sr / 1000)
            duration_s = seg.duration_ms / 1000
            torchaudio.save(
                str(cluster_dir / f"sample_{i:02d}_{duration_s:.1f}s.wav"),
                waveform[:, start:end],
                sr,
            )

        info(
            f"Saved {min(len(cluster_segs), max_samples)} samples"
            f" for {cluster_names[cluster_id]} (longest: {cluster_segs[0].duration_ms / 1000:.1f}s)"
        )


def format_ts(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}"


def merge_transcripts(segments: list) -> tuple[str, int]:
    """Merge transcripts into dialogue sorted by time."""
    segments.sort(key=lambda x: x.start_ms)
    dialogue = "\n\n".join(
        f"**[{format_ts(s.start_ms)}] {s.speaker or 'Unknown'}:** {s.text}" for s in segments
    )
    return dialogue, len(segments)


def parse_track_args(extra_args: list[str]) -> dict[int, str]:
    """Parse --trackN 'Name' arguments from extra args. Returns {track_num: name}."""
    track_names = {}
    i = 0
    while i < len(extra_args):
        m = re.match(r"^--track(\d+)$", extra_args[i])
        if m:
            track_num = int(m.group(1))
            if i + 1 >= len(extra_args):
                raise ValueError(f"--track{track_num} requires a name argument")
            track_names[track_num] = extra_args[i + 1]
            i += 2
        else:
            raise ValueError(f"Unknown argument: {extra_args[i]}")
    return track_names


# === Pipeline helpers ===


def _load_pipeline(max_speakers: int, threshold: float):
    """Load diarization pipeline models (VAD, embeddings, clustering, identification)."""
    vad = VADProcessor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    clusterer = SpectralClusterer(min_speakers=2, max_speakers=max_speakers)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor, threshold=threshold)
    return vad, extractor, clusterer, identifier


def _run_vad(vad, audio_path: Path):
    """Run VAD on an audio track. Returns speech segments."""
    speech_segs = vad.process(audio_path)
    ok(f"{len(speech_segs)} speech segments")
    return speech_segs


def _run_embeddings(vad, extractor, speech_segs, audio_path: Path):
    """Extract embeddings for speech segments."""
    embeddings = []
    for seg in speech_segs:
        audio = vad.extract_segment_audio(audio_path, seg)
        embeddings.append(extractor.extract_from_tensor(audio))
    ok(f"{len(embeddings)} embeddings")
    return embeddings


def _run_clustering(clusterer, identifier, speech_segs, embeddings):
    """Run clustering and identification. Returns (diarized, cluster_names)."""
    time_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
    diarized = clusterer.cluster(embeddings, time_segs)

    centroids = clusterer.get_cluster_centroids(diarized)
    matches = identifier.identify_clusters(centroids)

    cluster_names = {cid: m.name for cid, m in matches.items()}
    known = sum(1 for m in matches.values() if m.is_known)
    ok(f"{len(centroids)} clusters ({known} known)")

    for cid, m in matches.items():
        if m.is_known:
            print(f"    {C_GREEN}✔{C_RESET} Cluster {cid}: {m.name} ({m.confidence:.2f})")
        else:
            print(f"    {C_YELLOW}❓{C_RESET} Cluster {cid}: {m.name} ({m.confidence:.2f})")

    # Update segments with cluster IDs
    for seg, diar in zip(speech_segs, diarized):
        seg.cluster_id = diar.cluster_id

    return diarized, cluster_names


def _run_transcription(transcriber, vad_segs, speaker_segs, track: Path, language: str):
    """Run transcription on pre-processed segments."""
    segs = transcriber.transcribe_vad_segments(track, vad_segs, speaker_segs, language)
    ok(f"{len(segs)} transcribed segments")
    return segs


# === Commands ===


def cmd_enroll(args):
    """Enroll a speaker."""
    samples_path = Path(args.samples_path)

    if samples_path.is_dir():
        audio_files = list(samples_path.glob("*.wav")) + list(samples_path.glob("*.mp3"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {samples_path}")
    else:
        audio_files = [samples_path]

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🎙️  {C_BOLD}Speaker Enrollment{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  👤 Name:    {C_CYAN}{args.name}{C_RESET}")
    print(f"  📁 Samples: {len(audio_files)} files")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    t = time.time()

    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor)
    voiceprint = identifier.enroll(args.name, audio_files)

    elapsed = time.time() - t
    print(
        f"\n  {C_GREEN}✅ Enrolled:{C_RESET} {C_BOLD}{args.name}{C_RESET}"
        f" {C_DIM}({_format_elapsed(elapsed)}){C_RESET}"
    )
    print(f"  📐 Embedding dim: {len(voiceprint)}")
    print(f"  💾 Saved to: {C_DIM}{config.VOICEPRINTS_FILE}{C_RESET}\n")


def cmd_list(args):
    """List enrolled speakers."""
    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  👥 {C_BOLD}Enrolled Speakers{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}\n")

    if not config.VOICEPRINTS_FILE.exists():
        warn("No speakers enrolled.")
        return

    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor)

    for i, name in enumerate(identifier.list_speakers(), 1):
        print(f"  {C_GREEN}✔{C_RESET} {i}. {name}")
    print()


def cmd_extract(args):
    """Extract audio tracks from a video file."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    output_dir = Path(args.output) if args.output else video_path.parent / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🎵 {C_BOLD}Extract Audio Tracks{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  📹 Input:  {C_BOLD}{video_path.name}{C_RESET}")
    print(f"  📂 Output: {C_DIM}{output_dir}{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    track_indices = probe_audio_tracks(video_path)
    if not track_indices:
        warn("No audio tracks found")
        return

    ok(f"Found {len(track_indices)} audio track(s)")

    t = time.time()
    for i, stream_idx in enumerate(track_indices, 1):
        out_file = output_dir / f"track_{i}.wav"
        with substep(f"Track {i} (stream {stream_idx})", "🎵"):
            extract_audio(video_path, out_file, stream_idx)
            ok(f"→ {out_file.name}")

    elapsed = time.time() - t
    print(f"\n  {C_GREEN}✅ Extracted {len(track_indices)} track(s)"
          f" {C_DIM}({_format_elapsed(elapsed)}){C_RESET}\n")


def _resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    """Resolve input arguments: expand directories and glob patterns to audio files."""
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    resolved = []
    for raw in raw_inputs:
        p = Path(raw)
        if p.is_dir():
            # Directory: collect audio files sorted by name
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in audio_exts)
            if not files:
                raise FileNotFoundError(f"No audio files found in {p}")
            resolved.extend(files)
        elif "*" in raw or "?" in raw:
            # Glob pattern: expand
            files = sorted(Path(f) for f in glob.glob(raw) if Path(f).suffix.lower() in audio_exts)
            if not files:
                raise FileNotFoundError(f"No audio files matching: {raw}")
            resolved.extend(files)
        else:
            resolved.append(p)
    return resolved


def cmd_transcribe(args, extra_args: list[str]):
    """Transcribe meeting with diarization. Supports multi-track video/audio."""
    input_paths = _resolve_inputs(args.input)
    output_path = Path(args.output)

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    track_names = parse_track_args(extra_args)

    date_str = datetime.now().strftime("%Y-%m-%d")
    if output_path.is_dir():
        output_path = output_path / f"{date_str}-meeting.md"

    # Determine if input is a single video (probe for tracks) or audio files
    is_single_video = len(input_paths) == 1 and input_paths[0].suffix.lower() in (
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".flv",
        ".wmv",
    )

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🎙️  {C_BOLD}MeetScribe{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    if is_single_video:
        print(f"  📹 Input:  {C_BOLD}{input_paths[0].name}{C_RESET}")
    else:
        print(f"  🎵 Input:  {C_BOLD}{len(input_paths)} audio file(s){C_RESET}")
    if track_names:
        for tn, name in sorted(track_names.items()):
            print(f"  🏷️  Track {tn}: {C_CYAN}{name}{C_RESET}")
    print(f"  💻 Device: {C_CYAN}{DEFAULT_DEVICE}{C_RESET}")
    print(f"  📄 Output: {C_DIM}{output_path}{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    total_start = time.time()

    # Determine track files
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        if is_single_video:
            stream_indices = probe_audio_tracks(input_paths[0])
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                with substep(f"Track {i} (stream {stream_idx})", "🎵"):
                    extract_audio(input_paths[0], out, stream_idx)
                track_files.append(out)
            ok(f"Extracted {len(track_files)} audio track(s)")
        else:
            # Audio files: each is a track, convert to 16kHz mono wav
            track_files = []
            for i, path in enumerate(input_paths, 1):
                if path.suffix.lower() == ".wav":
                    track_files.append(path)
                else:
                    out = work_dir / f"track_{i}.wav"
                    with substep(f"Converting {path.name}", "🎵"):
                        run_cmd(
                            [
                                _FFMPEG_BIN,
                                "-y",
                                "-i",
                                str(path),
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                str(out),
                            ],
                            f"Converting {path.name} to wav",
                        )
                    track_files.append(out)

        num_tracks = len(track_files)

        # Load models
        with step(1, 3, "Loading models", "🧠"):
            vad, extractor, clusterer, identifier = _load_pipeline(
                args.max_speakers, args.threshold
            )
            transcriber = Transcriber(model_size=args.model, device=DEFAULT_DEVICE)

        # Process each track
        all_segments = []
        with step(2, 3, f"Processing {num_tracks} track(s)", "🎤"):
            for track_num in range(1, num_tracks + 1):
                track_path = track_files[track_num - 1]
                speaker_name = track_names.get(track_num)

                print(f"\n    {C_BLUE}── Track {track_num} ", end="")
                if speaker_name:
                    print(f"({speaker_name}) ──{C_RESET}")
                else:
                    print(f"(diarize) ──{C_RESET}")

                # VAD
                with substep("VAD", "🎤"):
                    speech_segs = _run_vad(vad, track_path)
                if not speech_segs:
                    warn(f"No speech in track {track_num}")
                    continue

                if speaker_name:
                    # Named track: label all segments with the given name
                    vad_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
                    speaker_segs = [(s.start_ms, s.end_ms, speaker_name) for s in speech_segs]
                    with substep("Transcription", "✍️"):
                        segs = _run_transcription(
                            transcriber, vad_segs, speaker_segs, track_path, args.language
                        )
                    all_segments.extend(segs)
                else:
                    # Diarize track
                    with substep("Embeddings", "🔊"):
                        embeddings = _run_embeddings(vad, extractor, speech_segs, track_path)
                    if embeddings:
                        with substep("Diarization", "👥"):
                            diarized, cluster_names = _run_clustering(
                                clusterer, identifier, speech_segs, embeddings
                            )
                        save_unknown_samples(track_path, speech_segs, cluster_names, date_str)
                        vad_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
                        speaker_segs = [
                            (d.start_ms, d.end_ms, cluster_names[d.cluster_id]) for d in diarized
                        ]
                        with substep("Transcription", "✍️"):
                            segs = _run_transcription(
                                transcriber, vad_segs, speaker_segs, track_path, args.language
                            )
                        all_segments.extend(segs)

        # Save
        with step(3, 3, "Writing output", "💾"):
            dialogue, total = merge_transcripts(all_segments)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(dialogue, encoding="utf-8")
            ok(f"{total} segments written")

    total_elapsed = time.time() - total_start
    print(f"\n{C_GREEN}{'═' * 60}{C_RESET}")
    elapsed_str = _format_elapsed(total_elapsed)
    print(f"  🎉 {C_GREEN}{C_BOLD}Done!{C_RESET} {C_DIM}({elapsed_str}){C_RESET}")
    print(f"  📄 Output: {output_path}")
    print(f"  📊 Total: {total} segments across {num_tracks} track(s)")
    print(f"{C_GREEN}{'═' * 60}{C_RESET}\n")


def cmd_extract_samples(args):
    """Extract speaker samples without transcription (fast)."""
    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🔊 {C_BOLD}Extract Speaker Samples{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  📹 Video:  {C_BOLD}{video_path.name}{C_RESET}")
    print(f"  💻 Device: {C_CYAN}{DEFAULT_DEVICE}{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    total_start = time.time()

    with step(1, 4, "Loading models", "🧠"):
        vad, extractor, clusterer, identifier = _load_pipeline(args.max_speakers, args.threshold)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        with step(2, 4, "Extracting audio", "🎵"):
            stream_indices = probe_audio_tracks(video_path)
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                extract_audio(video_path, out, stream_idx)
                track_files.append(out)
            ok(f"Extracted {len(track_files)} track(s)")

        for track_num, track_path in enumerate(track_files, 1):
            print(f"\n    {C_BLUE}── Track {track_num} ──{C_RESET}")

            with step(3, 4, f"VAD + embeddings (track {track_num})", "🎤"):
                speech_segs = _run_vad(vad, track_path)
                if not speech_segs:
                    warn(f"No speech in track {track_num}")
                    continue
                embeddings = _run_embeddings(vad, extractor, speech_segs, track_path)

            with step(4, 4, f"Diarization & saving (track {track_num})", "👥"):
                diarized, cluster_names = _run_clustering(
                    clusterer, identifier, speech_segs, embeddings
                )
                save_unknown_samples(track_path, speech_segs, cluster_names, date_str)

    total_elapsed = time.time() - total_start
    samples_dir = config.SAMPLES_DIR / "unknown"
    print(f"\n{C_GREEN}{'═' * 60}{C_RESET}")
    elapsed_str = _format_elapsed(total_elapsed)
    print(f"  🎉 {C_GREEN}{C_BOLD}Done!{C_RESET} {C_DIM}({elapsed_str}){C_RESET}")
    print(f"  📂 Samples: {samples_dir}")
    print(f"{C_GREEN}{'═' * 60}{C_RESET}\n")


def cmd_info(args):
    """Show data directories and configuration."""
    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  ℹ️  {C_BOLD}MeetScribe Configuration{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}\n")
    print(f"  📂 Data:       {C_DIM}{config.DATA_DIR}{C_RESET}")
    print(f"  📦 Cache:      {C_DIM}{config.CACHE_DIR}{C_RESET}")
    print(f"  🧠 Models:     {C_DIM}{config.MODELS_DIR}{C_RESET}")
    print(f"  🔑 Voiceprints:{C_DIM} {config.VOICEPRINTS_FILE}{C_RESET}")
    print(f"  🎤 Samples:    {C_DIM}{config.SAMPLES_DIR}{C_RESET}")
    print(f"  📋 Logs:       {C_DIM}{config.LOGS_DIR}{C_RESET}")
    print(f"\n  💻 Device: {C_CYAN}{DEFAULT_DEVICE}{C_RESET}")
    print()


def main():
    setup_environment()

    parser = argparse.ArgumentParser(
        description="MeetScribe - Meeting transcription with speaker diarization"
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # enroll
    p = subs.add_parser("enroll", help="Enroll speaker")
    p.add_argument("name", help="Speaker name")
    p.add_argument("samples_path", help="Directory with audio samples or single audio file")
    p.set_defaults(func=cmd_enroll)

    # list-speakers
    p = subs.add_parser("list-speakers", help="List enrolled speakers")
    p.set_defaults(func=cmd_list)

    # extract-samples
    p = subs.add_parser("extract-samples", help="Extract speaker samples (fast, no transcription)")
    p.add_argument("video", help="Video file")
    p.add_argument("--max-speakers", type=int, default=10, help="Max speakers")
    p.add_argument("--threshold", type=float, default=0.7, help="ID threshold")
    p.set_defaults(func=cmd_extract_samples)

    # extract
    p = subs.add_parser("extract", help="Extract audio tracks from video")
    p.add_argument("video", help="Video file")
    p.add_argument("-o", "--output", default=None, help="Output directory")
    p.set_defaults(func=cmd_extract)

    # transcribe
    p = subs.add_parser(
        "transcribe",
        help="Transcribe meeting (use --trackN 'Name' to assign speakers)",
    )
    p.add_argument("input", nargs="+", help="Video or audio file(s)")
    p.add_argument("-o", "--output", required=True, help="Output path")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Whisper model")
    p.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Language")
    p.add_argument("--max-speakers", type=int, default=10, help="Max speakers")
    p.add_argument("--threshold", type=float, default=0.7, help="ID threshold")
    p.set_defaults(func=cmd_transcribe)

    # info
    p = subs.add_parser("info", help="Show configuration and data directories")
    p.set_defaults(func=cmd_info)

    args, extra = parser.parse_known_args()
    try:
        if args.func == cmd_transcribe:
            args.func(args, extra)
        else:
            if extra:
                parser.error(f"Unrecognized arguments: {' '.join(extra)}")
            args.func(args)
    except Exception as e:
        print(f"\n  {C_RED}❌ Error:{C_RESET} {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
