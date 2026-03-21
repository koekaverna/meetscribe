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
import re
import shutil
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import colorama

from . import config
from .servers import load_config

# Enable ANSI colors on Windows
colorama.just_fix_windows_console()

# Ensure directories exist
config.ensure_dirs()

# === Logging setup (before heavy imports to capture their warnings) ===
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

from .pipeline import (  # noqa: E402
    DiarizationPipeline,
    EmbeddingExtractor,
    SpeechSegment,
    Transcriber,
    audio,
    save_voiceprint,
)
from .pipeline.audio import FFmpegNotFoundError  # noqa: E402

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

# Defaults
DEFAULT_LANGUAGE = "ru"


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


def save_unknown_samples(
    audio_path: Path,
    segments: list[SpeechSegment],
    date_str: str,
    min_duration_ms: int = 3000,
    max_samples: int = 5,
) -> None:
    """Save audio samples for unknown speakers using FFmpeg."""
    unknown_speakers: dict[str, list[SpeechSegment]] = {}
    for seg in segments:
        if seg.speaker and seg.speaker.startswith("Unknown") and seg.duration_ms >= min_duration_ms:
            unknown_speakers.setdefault(seg.speaker, []).append(seg)

    if not unknown_speakers:
        return

    samples_dir = config.SAMPLES_DIR / "unknown"

    for speaker_label, segs in unknown_speakers.items():
        cluster_dir = samples_dir / f"{date_str}-{speaker_label.replace(' ', '_').lower()}"
        segs.sort(key=lambda s: s.duration_ms, reverse=True)

        cluster_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segs[:max_samples]):
            duration_s = seg.duration_ms / 1000
            out_file = cluster_dir / f"sample_{i:02d}_{duration_s:.1f}s.wav"
            audio.extract_segment(audio_path, out_file, seg.start_ms, seg.end_ms)

        info(
            f"Saved {min(len(segs), max_samples)} samples"
            f" for {speaker_label} (longest: {segs[0].duration_ms / 1000:.1f}s)"
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


# === Commands ===


def cmd_enroll(args):
    """Enroll a speaker by copying audio samples and computing voiceprint."""
    samples_path = Path(args.samples_path)

    if samples_path.is_dir():
        audio_files = sorted(
            f
            for f in samples_path.iterdir()
            if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        )
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {samples_path}")
    else:
        audio_files = [samples_path]

    for f in audio_files:
        if not f.exists():
            raise FileNotFoundError(f"File not found: {f}")

    # Load server config for embedding extraction
    servers_cfg = load_config(config.SERVERS_CONFIG)

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🎙️  {C_BOLD}Speaker Enrollment{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  👤 Name:    {C_CYAN}{args.name}{C_RESET}")
    print(f"  📁 Samples: {len(audio_files)} files")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    t = time.time()

    # Copy samples to enrolled directory
    enrolled_dir = config.ENROLLED_SAMPLES_DIR / args.name
    enrolled_dir.mkdir(parents=True, exist_ok=True)

    wav_files: list[Path] = []
    for audio_file in audio_files:
        dest = enrolled_dir / audio_file.name
        if audio_file.suffix.lower() == ".wav":
            if audio_file.resolve() != dest.resolve():
                shutil.copy2(audio_file, dest)
            wav_files.append(dest)
        else:
            wav_dest = dest.with_suffix(".wav")
            audio.convert_to_wav(audio_file, wav_dest)
            wav_files.append(wav_dest)

    ok(f"{len(wav_files)} sample(s) in {enrolled_dir}")

    # Compute voiceprint via remote embeddings API
    with substep("Computing voiceprint", "🔑"):
        extractor = EmbeddingExtractor(servers_cfg.get_embeddings_url())
        embeddings: list[list[float]] = []
        for wav_file in wav_files:
            emb = extractor.extract_from_file(wav_file)
            embeddings.append(emb)
            info(f"Embedding extracted from {wav_file.name}")

        # Average embeddings across samples
        avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
        save_voiceprint(config.VOICEPRINTS_DIR, args.name, avg_embedding)

    elapsed = time.time() - t
    print(
        f"\n  {C_GREEN}✅ Enrolled:{C_RESET} {C_BOLD}{args.name}{C_RESET}"
        f" {C_DIM}({_format_elapsed(elapsed)}){C_RESET}"
    )
    print(f"  💾 Samples:    {C_DIM}{enrolled_dir}{C_RESET}")
    print(f"  🔑 Voiceprint: {C_DIM}{config.VOICEPRINTS_DIR / f'{args.name}.json'}{C_RESET}\n")


def cmd_list(args):
    """List enrolled speakers."""
    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  👥 {C_BOLD}Enrolled Speakers{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}\n")

    if not config.ENROLLED_SAMPLES_DIR.exists():
        warn("No speakers enrolled.")
        return

    speakers = sorted(
        d.name for d in config.ENROLLED_SAMPLES_DIR.iterdir() if d.is_dir() and any(d.glob("*.wav"))
    )

    if not speakers:
        warn("No speakers enrolled.")
        return

    for i, name in enumerate(speakers, 1):
        sample_count = len(list((config.ENROLLED_SAMPLES_DIR / name).glob("*.wav")))
        print(f"  {C_GREEN}✔{C_RESET} {i}. {name} ({sample_count} samples)")
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

    track_indices = audio.probe_audio_tracks(video_path)
    if not track_indices:
        warn("No audio tracks found")
        return

    ok(f"Found {len(track_indices)} audio track(s)")

    t = time.time()
    for i, stream_idx in enumerate(track_indices, 1):
        out_file = output_dir / f"track_{i}.wav"
        with substep(f"Track {i} (stream {stream_idx})", "🎵"):
            audio.extract_audio(video_path, out_file, stream_idx)
            ok(f"→ {out_file.name}")

    elapsed = time.time() - t
    print(
        f"\n  {C_GREEN}✅ Extracted {len(track_indices)} track(s)"
        f" {C_DIM}({_format_elapsed(elapsed)}){C_RESET}\n"
    )


def _resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    """Resolve input arguments: expand directories and glob patterns to audio files."""
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    resolved = []
    for raw in raw_inputs:
        p = Path(raw)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in audio_exts)
            if not files:
                raise FileNotFoundError(f"No audio files found in {p}")
            resolved.extend(files)
        elif "*" in raw or "?" in raw:
            files = sorted(Path(f) for f in glob.glob(raw) if Path(f).suffix.lower() in audio_exts)
            if not files:
                raise FileNotFoundError(f"No audio files matching: {raw}")
            resolved.extend(files)
        else:
            resolved.append(p)
    return resolved


def cmd_transcribe(args, extra_args: list[str]):
    """Transcribe meeting with diarization using remote servers."""
    input_paths = _resolve_inputs(args.input)
    output_path = Path(args.output)

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    track_names = parse_track_args(extra_args)

    # Load server config
    servers_cfg = load_config(config.SERVERS_CONFIG)

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
    print(f"  📄 Output: {C_DIM}{output_path}{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    total_start = time.time()

    # Create pipeline components
    diarization = DiarizationPipeline(
        vad_url=servers_cfg.get_vad_url(),
        embedding_url=servers_cfg.get_embeddings_url(),
        voiceprints_dir=config.VOICEPRINTS_DIR,
    )
    transcriber = Transcriber(
        servers_cfg.get_transcription_urls(),
        language=args.language,
    )

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        if is_single_video:
            stream_indices = audio.probe_audio_tracks(input_paths[0])
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                with substep(f"Track {i} (stream {stream_idx})", "🎵"):
                    audio.extract_audio(input_paths[0], out, stream_idx)
                track_files.append(out)
            ok(f"Extracted {len(track_files)} audio track(s)")
        else:
            track_files = []
            for i, path in enumerate(input_paths, 1):
                if path.suffix.lower() == ".wav":
                    track_files.append(path)
                else:
                    out = work_dir / f"track_{i}.wav"
                    with substep(f"Converting {path.name}", "🎵"):
                        audio.convert_to_wav(path, out)
                    track_files.append(out)

        num_tracks = len(track_files)

        # Process each track
        all_segments = []
        with step(1, 2, f"Processing {num_tracks} track(s)", "🎤"):
            for track_num in range(1, num_tracks + 1):
                track_path = track_files[track_num - 1]
                speaker_name = track_names.get(track_num)

                print(f"\n    {C_BLUE}── Track {track_num} ", end="")
                if speaker_name:
                    print(f"({speaker_name}) ──{C_RESET}")
                else:
                    print(f"(diarize) ──{C_RESET}")

                if speaker_name:
                    # Named track: transcribe whole file, assign speaker
                    with substep("Transcription (named track)", "✍️"):
                        segs = transcriber.transcribe_file(track_path, speaker=speaker_name)
                        ok(f"{len(segs)} transcribed segments")
                    all_segments.extend(segs)
                else:
                    # Diarize track: VAD -> embeddings -> identification
                    with substep("Voice activity detection", "🔍"):
                        segments = diarization.vad.detect(track_path)
                        if not segments:
                            warn(f"No speech in track {track_num}")
                            continue
                        ok(f"{len(segments)} speech segments")

                    with substep("Speaker embeddings", "🔑"):
                        segments_with_emb = diarization.embeddings.extract_segments(
                            track_path, segments
                        )
                        ok(f"{len(segments_with_emb)} embeddings extracted")

                    with substep("Speaker identification", "👥"):
                        segments = diarization.identifier.identify_segments(segments_with_emb)
                        speakers = {s.speaker for s in segments if s.speaker}
                        ok(f"{len(speakers)} speaker(s) identified")
                        for spk in sorted(speakers):
                            is_known = not spk.startswith("Unknown")
                            marker = f"{C_GREEN}✔" if is_known else f"{C_YELLOW}❓"
                            print(f"    {marker}{C_RESET} {spk}")

                    save_unknown_samples(track_path, segments, date_str)

                    # Transcribe diarized segments
                    with substep("Transcription", "✍️"):
                        segs = transcriber.transcribe_segments(track_path, segments)
                        ok(f"{len(segs)} transcribed segments")
                    all_segments.extend(segs)

        # Save
        with step(2, 2, "Writing output", "💾"):
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

    # Load server config
    servers_cfg = load_config(config.SERVERS_CONFIG)

    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🔊 {C_BOLD}Extract Speaker Samples{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")
    print(f"  🎵 Input:  {C_BOLD}{video_path.name}{C_RESET}")
    print(f"{C_MAGENTA}{'═' * 60}{C_RESET}")

    total_start = time.time()
    diarization = DiarizationPipeline(
        vad_url=servers_cfg.get_vad_url(),
        embedding_url=servers_cfg.get_embeddings_url(),
        voiceprints_dir=config.VOICEPRINTS_DIR,
    )

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        with step(1, 2, "Extracting audio", "🎵"):
            stream_indices = audio.probe_audio_tracks(video_path)
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                audio.extract_audio(video_path, out, stream_idx)
                track_files.append(out)
            ok(f"Extracted {len(track_files)} track(s)")

        with step(2, 2, "Diarizing & saving samples", "👥"):
            for track_num, track_path in enumerate(track_files, 1):
                print(f"\n    {C_BLUE}── Track {track_num} ──{C_RESET}")

                with substep("Voice activity detection", "🔍"):
                    segments = diarization.vad.detect(track_path)
                    if not segments:
                        warn(f"No speech in track {track_num}")
                        continue
                    ok(f"{len(segments)} speech segments")

                with substep("Speaker embeddings", "🔑"):
                    segments_with_emb = diarization.embeddings.extract_segments(
                        track_path, segments
                    )
                    ok(f"{len(segments_with_emb)} embeddings extracted")

                with substep("Speaker identification", "👥"):
                    segments = diarization.identifier.identify_segments(segments_with_emb)
                    speakers = {s.speaker for s in segments if s.speaker}
                    ok(f"{len(speakers)} speaker(s) identified")
                    for spk in sorted(speakers):
                        is_known = not spk.startswith("Unknown")
                        marker = f"{C_GREEN}✔" if is_known else f"{C_YELLOW}❓"
                        print(f"    {marker}{C_RESET} {spk}")

                save_unknown_samples(track_path, segments, date_str)

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
    print(f"  🔑 Voiceprints:{C_DIM} {config.VOICEPRINTS_DIR}{C_RESET}")
    print(f"  🎤 Samples:    {C_DIM}{config.SAMPLES_DIR}{C_RESET}")
    print(f"  🎤 Enrolled:   {C_DIM}{config.ENROLLED_SAMPLES_DIR}{C_RESET}")
    print(f"  📋 Logs:       {C_DIM}{config.LOGS_DIR}{C_RESET}")
    print(f"  ⚙️  Config:     {C_DIM}{config.SERVERS_CONFIG}{C_RESET}")
    print()


def main():
    colorama.just_fix_windows_console()

    try:
        audio.check_ffmpeg()
    except FFmpegNotFoundError:
        print(f"\n{C_RED}[ERROR] FFmpeg not found{C_RESET}\n")
        print("FFmpeg is required for audio processing. Install it:\n")
        print(f'  {C_CYAN}Windows:{C_RESET}  winget install "FFmpeg (Shared)"')
        print(f"  {C_CYAN}macOS:{C_RESET}    brew install ffmpeg")
        print(f"  {C_CYAN}Linux:{C_RESET}    sudo apt install ffmpeg")
        print("\nAfter installation, restart your terminal.\n")
        raise SystemExit(1)

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
    p.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Language")
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
