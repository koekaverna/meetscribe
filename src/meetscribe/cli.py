#!/usr/bin/env python3
"""
MeetScribe - Meeting Transcription with Speaker Diarization

Track 1: Host (simple transcription)
Track 2: Guests (VAD -> Embeddings -> Diarization -> Identification)

Usage:
    meetscribe enroll "John" samples/*.wav
    meetscribe transcribe meeting.mp4 --host "John" --output ./notes/
    meetscribe list-speakers
"""

import os
import sys
import logging
import warnings
from datetime import datetime as _dt
from pathlib import Path as _LogPath

from . import config

# Ensure directories exist
config.ensure_dirs()

# === Logging setup (before other imports) ===
_log_file = config.LOGS_DIR / f"{_dt.now():%Y-%m-%d_%H-%M-%S}.log"

# File handler: all messages (DEBUG+)
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

# Console handler: ERROR+ only
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

# Root logger
logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])

# All warnings -> log file (via py.warnings logger), console only ERROR+
logging.captureWarnings(True)
warnings.filterwarnings("default")

# Windows: disable symlinks, use COPY strategy
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

# Add NVIDIA DLLs to PATH for Windows CUDA
if sys.platform == "win32":
    from pathlib import Path as WinPath

    site_packages = WinPath(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if site_packages.exists():
        for lib_dir in site_packages.iterdir():
            bin_dir = lib_dir / "bin"
            if bin_dir.exists():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

# Patch Whisper to use imageio-ffmpeg binary
import imageio_ffmpeg

_FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


def _patch_whisper_ffmpeg():
    """Patch whisper.audio.load_audio to use imageio_ffmpeg binary."""
    import numpy as np
    from subprocess import run, CalledProcessError

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


_patch_whisper_ffmpeg()


import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

from .pipeline import (
    VADProcessor,
    EmbeddingExtractor,
    SpectralClusterer,
    SpeakerIdentifier,
    Transcriber,
)

# Defaults
DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "ru"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_cmd(cmd: list[str], desc: str) -> None:
    """Run command with error handling."""
    print(f"  -> {desc}...")
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
            print(
                f"  -> No samples >= {min_duration_ms / 1000:.0f}s for {cluster_names[cluster_id]}"
            )
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

        print(
            f"  -> Saved {min(len(cluster_segs), max_samples)} samples for {cluster_names[cluster_id]} (longest: {cluster_segs[0].duration_ms / 1000:.1f}s)"
        )


def format_ts(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}"


def merge_transcripts(host_segs: list, guest_segs: list) -> tuple[str, int]:
    """Merge transcripts into dialogue sorted by time."""
    all_segs = host_segs + guest_segs
    all_segs.sort(key=lambda x: x.start_ms)
    dialogue = "\n\n".join(
        f"**[{format_ts(s.start_ms)}] {s.speaker or 'Unknown'}:** {s.text}" for s in all_segs
    )
    return dialogue, len(all_segs)


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

    print(f"\n{'=' * 60}")
    print(f"  Speaker Enrollment: {args.name}")
    print(f"  Samples: {len(audio_files)} files")
    print(f"{'=' * 60}\n")

    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor)

    voiceprint = identifier.enroll(args.name, audio_files)

    print(f"\n  [OK] Enrolled: {args.name}")
    print(f"  Embedding dim: {len(voiceprint)}")
    print(f"  Saved to: {config.VOICEPRINTS_FILE}\n")


def cmd_list(args):
    """List enrolled speakers."""
    print(f"\n{'=' * 60}")
    print(f"  Enrolled Speakers")
    print(f"{'=' * 60}\n")

    if not config.VOICEPRINTS_FILE.exists():
        print("  No speakers enrolled.\n")
        return

    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor)

    for i, name in enumerate(identifier.list_speakers(), 1):
        print(f"  {i}. {name}")
    print()


def cmd_transcribe(args):
    """Transcribe meeting with diarization."""
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    date_str = datetime.now().strftime("%Y-%m-%d")
    if output_path.is_dir():
        output_path = output_path / f"{date_str}-meeting.md"

    print(f"\n{'=' * 60}")
    print(f"  MeetScribe")
    print(f"{'=' * 60}")
    print(f"  Video:  {video_path.name}")
    print(f"  Host:   {args.host}")
    print(f"  Device: {DEFAULT_DEVICE}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    # Load models
    print("[0/7] Loading models...")
    vad = VADProcessor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    clusterer = SpectralClusterer(min_speakers=2, max_speakers=args.max_speakers)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor, threshold=args.threshold)
    transcriber = Transcriber(model_size=args.model, device=DEFAULT_DEVICE)
    print("  [OK] Models loaded")

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # Extract tracks
        print("\n[1/7] Extracting audio...")
        track1 = extract_audio(video_path, work_dir / "host.wav", 1)
        track2 = extract_audio(video_path, work_dir / "guests.wav", 2)

        # Host: VAD + transcription
        print("\n[2/7] VAD on host track...")
        host_speech = vad.process(track1)
        print(f"  [OK] {len(host_speech)} speech segments")

        if host_speech:
            print("\n[3/7] Transcribing host...")
            host_vad_segs = [(s.start_ms, s.end_ms) for s in host_speech]
            host_speaker_segs = [(s.start_ms, s.end_ms, args.host) for s in host_speech]
            host_segs = transcriber.transcribe_vad_segments(
                track1, host_vad_segs, host_speaker_segs, args.language
            )
            print(f"  [OK] {len(host_segs)} segments")
        else:
            host_segs = []

        # Guests: VAD
        print("\n[4/7] VAD on guests track...")
        speech_segs = vad.process(track2)
        print(f"  [OK] {len(speech_segs)} speech segments")

        if not speech_segs:
            print("  [!] No speech detected")
            guest_segs = []
        else:
            # Embeddings
            print("\n[5/7] Extracting embeddings...")
            embeddings = []
            for seg in speech_segs:
                audio = vad.extract_segment_audio(track2, seg)
                embeddings.append(extractor.extract_from_tensor(audio))
            print(f"  [OK] {len(embeddings)} embeddings")

            # Diarization
            print("\n[6/7] Diarization...")
            time_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
            diarized = clusterer.cluster(embeddings, time_segs)

            centroids = clusterer.get_cluster_centroids(diarized)
            matches = identifier.identify_clusters(centroids)

            cluster_names = {cid: m.name for cid, m in matches.items()}
            known = sum(1 for m in matches.values() if m.is_known)
            print(f"  [OK] {len(centroids)} clusters ({known} known)")

            for cid, m in matches.items():
                status = "[OK]" if m.is_known else "[?]"
                print(f"    {status} Cluster {cid}: {m.name} ({m.confidence:.2f})")

            # Update segments with cluster IDs and save unknown
            for seg, diar in zip(speech_segs, diarized):
                seg.cluster_id = diar.cluster_id
            save_unknown_samples(track2, speech_segs, cluster_names, date_str)

            # Transcribe guests
            print("\n[7/7] Transcribing guests...")
            vad_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
            speaker_segs = [(d.start_ms, d.end_ms, cluster_names[d.cluster_id]) for d in diarized]
            guest_segs = transcriber.transcribe_vad_segments(
                track2, vad_segs, speaker_segs, args.language
            )
            print(f"  [OK] {len(guest_segs)} segments")

    # Save
    dialogue, total = merge_transcripts(host_segs, guest_segs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dialogue, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  [OK] Done: {output_path}")
    print(f"  Host segments: {len(host_segs)}")
    print(f"  Guest segments: {len(guest_segs)}")
    print(f"  Total: {total} segments")
    print(f"{'=' * 60}\n")


def cmd_extract_samples(args):
    """Extract speaker samples without transcription (fast)."""
    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"  Extract Speaker Samples (no transcription)")
    print(f"{'=' * 60}")
    print(f"  Video:  {video_path.name}")
    print(f"  Device: {DEFAULT_DEVICE}")
    print(f"{'=' * 60}\n")

    # Load models (no Whisper needed)
    print("[0/4] Loading models...")
    vad = VADProcessor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    extractor = EmbeddingExtractor(device=DEFAULT_DEVICE, cache_dir=config.MODELS_DIR)
    clusterer = SpectralClusterer(min_speakers=2, max_speakers=args.max_speakers)
    identifier = SpeakerIdentifier(config.VOICEPRINTS_FILE, extractor, threshold=args.threshold)
    print("  [OK] Models loaded")

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # Extract only guests track
        print("\n[1/4] Extracting audio...")
        track2 = extract_audio(video_path, work_dir / "guests.wav", 2)

        # VAD
        print("\n[2/4] VAD on guests track...")
        speech_segs = vad.process(track2)
        print(f"  [OK] {len(speech_segs)} speech segments")

        if not speech_segs:
            print("  [!] No speech detected")
            return

        # Embeddings
        print("\n[3/4] Extracting embeddings...")
        embeddings = []
        for seg in speech_segs:
            audio = vad.extract_segment_audio(track2, seg)
            embeddings.append(extractor.extract_from_tensor(audio))
        print(f"  [OK] {len(embeddings)} embeddings")

        # Diarization
        print("\n[4/4] Diarization & saving samples...")
        time_segs = [(s.start_ms, s.end_ms) for s in speech_segs]
        diarized = clusterer.cluster(embeddings, time_segs)

        centroids = clusterer.get_cluster_centroids(diarized)
        matches = identifier.identify_clusters(centroids)

        cluster_names = {cid: m.name for cid, m in matches.items()}
        known = sum(1 for m in matches.values() if m.is_known)
        print(f"  [OK] {len(centroids)} clusters ({known} known)")

        for cid, m in matches.items():
            status = "[OK]" if m.is_known else "[?]"
            print(f"    {status} Cluster {cid}: {m.name} ({m.confidence:.2f})")

        # Update segments with cluster IDs and save
        for seg, diar in zip(speech_segs, diarized):
            seg.cluster_id = diar.cluster_id
        save_unknown_samples(track2, speech_segs, cluster_names, date_str)

    samples_dir = config.SAMPLES_DIR / "unknown"
    print(f"\n{'=' * 60}")
    print(f"  [OK] Done! Samples saved to: {samples_dir}")
    print(f"{'=' * 60}\n")


def cmd_info(args):
    """Show data directories and configuration."""
    print(f"\n{'=' * 60}")
    print(f"  MeetScribe Configuration")
    print(f"{'=' * 60}\n")
    print(f"  Data directory:       {config.DATA_DIR}")
    print(f"  Cache directory:      {config.CACHE_DIR}")
    print(f"  Models directory:     {config.MODELS_DIR}")
    print(f"  Voiceprints:          {config.VOICEPRINTS_FILE}")
    print(f"  Samples directory:    {config.SAMPLES_DIR}")
    print(f"  Logs directory:       {config.LOGS_DIR}")
    print(f"\n  Device: {DEFAULT_DEVICE}")
    print()


def main():
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

    # transcribe
    p = subs.add_parser("transcribe", help="Transcribe meeting")
    p.add_argument("video", help="Video file")
    p.add_argument("-H", "--host", required=True, help="Host name")
    p.add_argument("-o", "--output", required=True, help="Output path")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Whisper model")
    p.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Language")
    p.add_argument("--max-speakers", type=int, default=10, help="Max speakers")
    p.add_argument("--threshold", type=float, default=0.7, help="ID threshold")
    p.set_defaults(func=cmd_transcribe)

    # info
    p = subs.add_parser("info", help="Show configuration and data directories")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
