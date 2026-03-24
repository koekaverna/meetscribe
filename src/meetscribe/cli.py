#!/usr/bin/env python3
"""
MeetScribe - Meeting Transcription with Speaker Diarization

Supports video files with multiple audio tracks and audio files directly.
Per-track mode: diarize (default) or assign speaker name via --trackN.

Usage:
    meetscribe transcribe video.mp4 -o output.md
    meetscribe transcribe video.mp4 -o output.md --track1 "Host"
    meetscribe transcribe audio1.wav audio2.wav -o output.md --track1 "Name"
    meetscribe enroll "John" samples/*.wav
    meetscribe extract video.mp4 -o output_dir/
    meetscribe extract-samples meeting.mp4
    meetscribe list-speakers
    meetscribe info
    meetscribe web
    meetscribe team create my-team
    meetscribe user create admin --team default --admin
    meetscribe -t my-team enroll "John" samples/*.wav
"""

import argparse
import glob
import logging
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import colorama

from . import config
from .database import (
    count_voiceprints,
    create_team,
    create_user,
    delete_team,
    delete_user,
    get_db,
    get_team,
    list_teams,
    list_users,
    load_voiceprints,
    save_voiceprint,
)
from .log import StructuredFormatter
from .servers import AppConfig, load_config
from .team import TeamContext, resolve_team

# Enable ANSI colors on Windows
colorama.just_fix_windows_console()

# Ensure directories exist
config.ensure_dirs()

# === Logging setup (before heavy imports to capture their warnings) ===
_log_file = config.LOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(StructuredFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(StructuredFormatter("[%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])
logging.captureWarnings(True)
warnings.filterwarnings("default")

from .errors import ConfigurationError, SpeachesAPIError  # noqa: E402
from .pipeline import (  # noqa: E402 — capture warnings from heavy deps
    DiarizationPipeline,
    EmbeddingExtractor,
    SpeechSegment,
    Transcriber,
    audio,
    enroll_samples,
)
from .pipeline.audio import FFmpegNotFoundError  # noqa: E402 — capture warnings from heavy deps

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


# === Helpers ===
def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


@contextmanager
def step(num: int, total: int, desc: str, emoji: str = "\U0001f504") -> Generator[None, None, None]:
    """Context manager that prints step header and elapsed time on completion."""
    print(f"\n{C_CYAN}[{num}/{total}]{C_RESET} {emoji}  {C_BOLD}{desc}{C_RESET}")
    t = time.time()
    yield
    elapsed = time.time() - t
    print(f"  {C_GREEN}\u2705 Done {C_DIM}({_format_elapsed(elapsed)}){C_RESET}")


@contextmanager
def substep(desc: str, emoji: str = "\U0001f504") -> Generator[None, None, None]:
    """Context manager that prints sub-step and elapsed time."""
    print(f"    {emoji}  {desc}")
    t = time.time()
    yield
    elapsed = time.time() - t
    print(f"      {C_DIM}({_format_elapsed(elapsed)}){C_RESET}")


def ok(msg: str) -> None:
    """Print a success sub-status line."""
    print(f"  {C_GREEN}\u2714{C_RESET}  {msg}")


def warn(msg: str) -> None:
    """Print a warning sub-status line."""
    print(f"  {C_YELLOW}\u26a0\ufe0f{C_RESET}  {msg}")


def info(msg: str) -> None:
    """Print an info sub-status line."""
    print(f"  {C_DIM}\u2192{C_RESET}  {msg}")


def _print_team_header(team_ctx: TeamContext) -> None:
    """Print active team indicator if not default."""
    if team_ctx.name != "default":
        print(f"  {C_CYAN}\U0001f465 Team: {C_BOLD}{team_ctx.name}{C_RESET}")


def _create_diarization(cfg: AppConfig, team_ctx: TeamContext) -> DiarizationPipeline:
    """Create a DiarizationPipeline from config and team context."""
    voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)
    return DiarizationPipeline(
        vad_url=cfg.get_vad_url(),
        embedding_url=cfg.get_embeddings_url(),
        voiceprints=voiceprints,
        threshold=cfg.embeddings.threshold,
        vad_timeout=cfg.vad.timeout,
        embedding_timeout=cfg.embeddings.timeout,
        min_duration_ms=cfg.embeddings.min_duration_ms,
        unknown_cluster_threshold=cfg.embeddings.unknown_cluster_threshold,
        confident_gap=cfg.embeddings.confident_gap,
        min_threshold=cfg.embeddings.min_threshold,
        max_workers=cfg.embeddings.max_workers,
        embedding_model=cfg.embeddings.model,
        vad_min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
        vad_speech_pad_ms=cfg.vad.speech_pad_ms,
        vad_threshold=cfg.vad.threshold,
    )


def save_unknown_samples(
    audio_path: Path,
    segments: list[SpeechSegment],
    date_str: str,
    unknown_samples_dir: Path,
    min_duration_ms: int = 3000,
    max_duration_ms: int = 12000,
    max_samples: int = 10,
) -> None:
    """Save audio samples for unknown speakers using FFmpeg.

    Prefers segments in the 5-10s range (best for clean single-speaker embeddings).
    Filters out segments that are too short or too long.
    """
    ideal_ms = 7000

    unknown_speakers: dict[str, list[SpeechSegment]] = {}
    for seg in segments:
        if (
            seg.speaker
            and seg.speaker.startswith("Unknown")
            and min_duration_ms <= seg.duration_ms <= max_duration_ms
        ):
            unknown_speakers.setdefault(seg.speaker, []).append(seg)

    if not unknown_speakers:
        return

    for speaker_label, segs in unknown_speakers.items():
        cluster_dir = unknown_samples_dir / f"{date_str}-{speaker_label.replace(' ', '_').lower()}"
        # Sort by proximity to ideal duration (7s) — medium segments are cleanest
        segs.sort(key=lambda s: abs(s.duration_ms - ideal_ms))

        cluster_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segs[:max_samples]):
            duration_s = seg.duration_ms / 1000
            out_file = cluster_dir / f"sample_{i:02d}_{duration_s:.1f}s.wav"
            audio.extract_segment(audio_path, out_file, seg.start_ms, seg.end_ms)

        saved = segs[:max_samples]
        lo = saved[-1].duration_ms / 1000
        hi = saved[0].duration_ms / 1000
        info(f"Saved {len(saved)} samples for {speaker_label} ({lo:.1f}s—{hi:.1f}s)")


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


def cmd_enroll(args: argparse.Namespace, team_ctx: TeamContext) -> None:
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

    # Load config
    cfg = load_config(config.CONFIG_FILE)

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f399\ufe0f  {C_BOLD}Speaker Enrollment{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")
    _print_team_header(team_ctx)
    print(f"  \U0001f464 Name:    {C_CYAN}{args.name}{C_RESET}")
    print(f"  \U0001f4c1 Samples: {len(audio_files)} files")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")

    t = time.time()

    # Convert non-wav to wav in temp, collect all as wav_paths
    enrolled_dir = team_ctx.enrolled_samples_dir / args.name
    tmp_converted: list[Path] = []
    wav_paths: list[Path] = []
    for audio_file in audio_files:
        if audio_file.suffix.lower() == ".wav":
            wav_paths.append(audio_file)
        else:
            wav_dest = audio_file.with_suffix(".wav")
            audio.convert_to_wav(audio_file, wav_dest)
            wav_paths.append(wav_dest)
            tmp_converted.append(wav_dest)

    # Enroll: copy all samples, compute voiceprint from all
    try:
        with substep("Computing voiceprint", "\U0001f511"):
            extractor = EmbeddingExtractor(
                cfg.get_embeddings_url(),
                cfg.embeddings.timeout,
                cfg.embeddings.min_duration_ms,
                model=cfg.embeddings.model,
            )
            avg_embedding, total_count, new_count = enroll_samples(
                extractor, wav_paths, enrolled_dir
            )
            ok(f"{total_count} sample(s) in {enrolled_dir} ({new_count} new)")
            save_voiceprint(
                team_ctx.conn, team_ctx.id, args.name, avg_embedding, cfg.embeddings.model
            )
    finally:
        for tmp in tmp_converted:
            tmp.unlink(missing_ok=True)

    elapsed = time.time() - t
    print(
        f"\n  {C_GREEN}\u2705 Enrolled:{C_RESET} {C_BOLD}{args.name}{C_RESET}"
        f" {C_DIM}({_format_elapsed(elapsed)}){C_RESET}"
    )
    print(f"  \U0001f4be Samples:    {C_DIM}{enrolled_dir}{C_RESET}")
    print(f"  \U0001f511 Voiceprint: {C_DIM}DB (team: {team_ctx.name}){C_RESET}\n")


def cmd_list(args: argparse.Namespace, team_ctx: TeamContext) -> None:
    """List enrolled speakers."""
    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f465 {C_BOLD}Enrolled Speakers{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")
    _print_team_header(team_ctx)
    print()

    voiceprints = load_voiceprints(team_ctx.conn, team_ctx.id)

    if not voiceprints:
        warn("No speakers enrolled.")
        return

    for i, name in enumerate(sorted(voiceprints.keys()), 1):
        # Count samples if directory exists
        sample_dir = team_ctx.enrolled_samples_dir / name
        sample_count = len(list(sample_dir.glob("*.wav"))) if sample_dir.exists() else 0
        print(f"  {C_GREEN}\u2714{C_RESET} {i}. {name} ({sample_count} samples)")
    print()


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract audio tracks from a video file."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    output_dir = Path(args.output) if args.output else video_path.parent / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f3b5 {C_BOLD}Extract Audio Tracks{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f4f9 Input:  {C_BOLD}{video_path.name}{C_RESET}")
    print(f"  \U0001f4c2 Output: {C_DIM}{output_dir}{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")

    track_indices = audio.probe_audio_tracks(video_path)
    if not track_indices:
        warn("No audio tracks found")
        return

    ok(f"Found {len(track_indices)} audio track(s)")

    t = time.time()
    for i, stream_idx in enumerate(track_indices, 1):
        out_file = output_dir / f"track_{i}.wav"
        with substep(f"Track {i} (stream {stream_idx})", "\U0001f3b5"):
            audio.extract_audio(video_path, out_file, stream_idx)
            ok(f"\u2192 {out_file.name}")

    elapsed = time.time() - t
    print(
        f"\n  {C_GREEN}\u2705 Extracted {len(track_indices)} track(s)"
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


def cmd_transcribe(args: argparse.Namespace, extra_args: list[str], team_ctx: TeamContext) -> None:
    """Transcribe meeting with diarization using remote servers."""
    input_paths = _resolve_inputs(args.input)
    output_path = Path(args.output)

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    track_names = parse_track_args(extra_args)

    # Load config
    cfg = load_config(config.CONFIG_FILE)
    language = args.language or cfg.transcription.language

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

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f399\ufe0f  {C_BOLD}MeetScribe{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")
    _print_team_header(team_ctx)
    if is_single_video:
        print(f"  \U0001f4f9 Input:  {C_BOLD}{input_paths[0].name}{C_RESET}")
    else:
        print(f"  \U0001f3b5 Input:  {C_BOLD}{len(input_paths)} audio file(s){C_RESET}")
    if track_names:
        for tn, name in sorted(track_names.items()):
            print(f"  \U0001f3f7\ufe0f  Track {tn}: {C_CYAN}{name}{C_RESET}")
    print(f"  \U0001f4c4 Output: {C_DIM}{output_path}{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")

    total_start = time.time()

    # Create pipeline components
    diarization = _create_diarization(cfg, team_ctx)
    transcriber = Transcriber(
        cfg.get_transcription_urls(),
        language=language,
        timeout=cfg.transcription.timeout,
        model=cfg.transcription.model,
        max_gap_ms=cfg.transcription.max_gap_ms,
        max_chunk_ms=cfg.transcription.max_chunk_ms,
    )

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp:
        work_dir = Path(tmp)

        if is_single_video:
            stream_indices = audio.probe_audio_tracks(input_paths[0])
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                with substep(f"Track {i} (stream {stream_idx})", "\U0001f3b5"):
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
                    with substep(f"Converting {path.name}", "\U0001f3b5"):
                        audio.convert_to_wav(path, out)
                    track_files.append(out)

        num_tracks = len(track_files)

        # Process each track
        all_segments = []
        with step(1, 2, f"Processing {num_tracks} track(s)", "\U0001f3a4"):
            for track_num in range(1, num_tracks + 1):
                track_path = track_files[track_num - 1]
                speaker_name = track_names.get(track_num)

                print(f"\n    {C_BLUE}\u2500\u2500 Track {track_num} ", end="")
                if speaker_name:
                    print(f"({speaker_name}) \u2500\u2500{C_RESET}")
                else:
                    print(f"(diarize) \u2500\u2500{C_RESET}")

                if speaker_name:
                    # Named track: transcribe whole file, assign speaker
                    with substep("Transcription (named track)", "\u270d\ufe0f"):
                        segs = transcriber.transcribe_file(track_path, speaker=speaker_name)
                        ok(f"{len(segs)} transcribed segments")
                    all_segments.extend(segs)
                else:
                    # Diarize track: VAD -> embeddings -> identification
                    with substep("Voice activity detection", "\U0001f50d"):
                        segments = diarization.vad.detect(track_path)
                        if not segments:
                            warn(f"No speech in track {track_num}")
                            continue
                        ok(f"{len(segments)} speech segments")

                    with substep("Speaker embeddings", "\U0001f511"):
                        segments_with_emb = diarization.embeddings.extract_segments(
                            track_path, segments, diarization.max_workers
                        )
                        ok(f"{len(segments_with_emb)} embeddings extracted")

                    with substep("Speaker identification", "\U0001f465"):
                        segments = diarization.identifier.identify_segments(segments_with_emb)
                        speakers = {s.speaker for s in segments if s.speaker}
                        ok(f"{len(speakers)} speaker(s) identified")
                        for spk in sorted(speakers):
                            is_known = not spk.startswith("Unknown")
                            marker = f"{C_GREEN}\u2714" if is_known else f"{C_YELLOW}\u2753"
                            print(f"    {marker}{C_RESET} {spk}")

                    save_unknown_samples(
                        track_path, segments, date_str, team_ctx.unknown_samples_dir
                    )

                    # Transcribe diarized segments
                    with substep("Transcription", "\u270d\ufe0f"):
                        segs = transcriber.transcribe_segments(track_path, segments)
                        ok(f"{len(segs)} transcribed segments")
                    all_segments.extend(segs)

        # Save
        with step(2, 2, "Writing output", "\U0001f4be"):
            dialogue, total = merge_transcripts(all_segments)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(dialogue, encoding="utf-8")
            ok(f"{total} segments written")

    total_elapsed = time.time() - total_start
    print(f"\n{C_GREEN}{'=' * 60}{C_RESET}")
    elapsed_str = _format_elapsed(total_elapsed)
    print(f"  \U0001f389 {C_GREEN}{C_BOLD}Done!{C_RESET} {C_DIM}({elapsed_str}){C_RESET}")
    print(f"  \U0001f4c4 Output: {output_path}")
    print(f"  \U0001f4ca Total: {total} segments across {num_tracks} track(s)")
    print(f"{C_GREEN}{'=' * 60}{C_RESET}\n")


def cmd_extract_samples(args: argparse.Namespace, team_ctx: TeamContext) -> None:
    """Extract speaker samples without transcription (fast)."""
    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load config
    cfg = load_config(config.CONFIG_FILE)

    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f50a {C_BOLD}Extract Speaker Samples{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")
    _print_team_header(team_ctx)
    print(f"  \U0001f3b5 Input:  {C_BOLD}{video_path.name}{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}")

    total_start = time.time()
    diarization = _create_diarization(cfg, team_ctx)

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp:
        work_dir = Path(tmp)

        with step(1, 2, "Extracting audio", "\U0001f3b5"):
            stream_indices = audio.probe_audio_tracks(video_path)
            if not stream_indices:
                raise RuntimeError("No audio tracks found in video")
            track_files = []
            for i, stream_idx in enumerate(stream_indices, 1):
                out = work_dir / f"track_{i}.wav"
                audio.extract_audio(video_path, out, stream_idx)
                track_files.append(out)
            ok(f"Extracted {len(track_files)} track(s)")

        with step(2, 2, "Diarizing & saving samples", "\U0001f465"):
            for track_num, track_path in enumerate(track_files, 1):
                print(f"\n    {C_BLUE}\u2500\u2500 Track {track_num} \u2500\u2500{C_RESET}")

                with substep("Voice activity detection", "\U0001f50d"):
                    segments = diarization.vad.detect(track_path)
                    if not segments:
                        warn(f"No speech in track {track_num}")
                        continue
                    ok(f"{len(segments)} speech segments")

                with substep("Speaker embeddings", "\U0001f511"):
                    segments_with_emb = diarization.embeddings.extract_segments(
                        track_path, segments, diarization.max_workers
                    )
                    ok(f"{len(segments_with_emb)} embeddings extracted")

                with substep("Speaker identification", "\U0001f465"):
                    segments = diarization.identifier.identify_segments(segments_with_emb)
                    speakers = {s.speaker for s in segments if s.speaker}
                    ok(f"{len(speakers)} speaker(s) identified")
                    for spk in sorted(speakers):
                        is_known = not spk.startswith("Unknown")
                        marker = f"{C_GREEN}\u2714" if is_known else f"{C_YELLOW}\u2753"
                        print(f"    {marker}{C_RESET} {spk}")

                save_unknown_samples(track_path, segments, date_str, team_ctx.unknown_samples_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{C_GREEN}{'=' * 60}{C_RESET}")
    elapsed_str = _format_elapsed(total_elapsed)
    print(f"  \U0001f389 {C_GREEN}{C_BOLD}Done!{C_RESET} {C_DIM}({elapsed_str}){C_RESET}")
    print(f"  \U0001f4c2 Samples: {team_ctx.unknown_samples_dir}")
    print(f"{C_GREEN}{'=' * 60}{C_RESET}\n")


def cmd_info(args: argparse.Namespace, team_ctx: TeamContext) -> None:
    """Show data directories and configuration."""
    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \u2139\ufe0f  {C_BOLD}MeetScribe Configuration{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}\n")
    _print_team_header(team_ctx)
    print(f"  \U0001f4c2 Data:       {C_DIM}{config.DATA_DIR}{C_RESET}")
    print(f"  \U0001f5c3\ufe0f  Database:   {C_DIM}{config.DB_PATH}{C_RESET}")
    print(f"  \U0001f3a4 Enrolled:   {C_DIM}{team_ctx.enrolled_samples_dir}{C_RESET}")
    print(f"  \U0001f50a Unknown:    {C_DIM}{team_ctx.unknown_samples_dir}{C_RESET}")
    vp_count = count_voiceprints(team_ctx.conn, team_ctx.id)
    print(f"  \U0001f511 Voiceprints:{C_DIM} {vp_count} in DB (team: {team_ctx.name}){C_RESET}")
    print(f"  \U0001f5d1\ufe0f  Temp:       {C_DIM}{config.TMP_DIR}{C_RESET}")
    print(f"  \U0001f4cb Logs:       {C_DIM}{config.LOGS_DIR}{C_RESET}")
    print(f"  \u2699\ufe0f  Config:     {C_DIM}{config.CONFIG_FILE}{C_RESET}")
    print()


def cmd_web(args: argparse.Namespace) -> None:
    """Start the web UI server."""
    try:
        from .web.app import run
    except ImportError:
        print(
            f"\n  {C_RED}\u274c Error:{C_RESET} Web dependencies not installed.\n"
            f'  Install with: pip install -e ".[web]"\n'
        )
        raise SystemExit(1)
    cfg = load_config(config.CONFIG_FILE)
    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    run(host=host, port=port)


# === User management commands ===


def cmd_user_create(args: argparse.Namespace) -> None:
    """Create a new user."""
    import getpass

    from .web.services.auth import hash_password

    conn = get_db(config.DB_PATH)
    team = get_team(conn, args.team)
    if not team:
        print(f"\n  {C_RED}\u274c Error:{C_RESET} Team '{args.team}' not found.\n")
        raise SystemExit(1)

    password = getpass.getpass("Password: ")
    if len(password) < 4:
        print(f"\n  {C_RED}\u274c Error:{C_RESET} Password must be at least 4 characters.\n")
        raise SystemExit(1)
    password2 = getpass.getpass("Confirm: ")
    if password != password2:
        print(f"\n  {C_RED}\u274c Error:{C_RESET} Passwords do not match.\n")
        raise SystemExit(1)

    try:
        pw_hash = hash_password(password)
        is_admin = getattr(args, "admin", False)
        create_user(conn, args.username, pw_hash, team["id"], is_admin=is_admin)
        role = " (admin)" if is_admin else ""
        print(
            f"\n  {C_GREEN}\u2714{C_RESET}  User '{C_BOLD}{args.username}{C_RESET}'"
            f" created (team: {args.team}){role}\n"
        )
    except Exception as e:
        print(f"\n  {C_RED}\u274c Error:{C_RESET} {e}\n")
        raise SystemExit(1)
    finally:
        conn.close()


def cmd_user_list(args: argparse.Namespace) -> None:
    """List all users."""
    conn = get_db(config.DB_PATH)
    users = list_users(conn)
    conn.close()

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f464 {C_BOLD}Users{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}\n")

    if not users:
        warn("No users registered.")
    else:
        for u in users:
            print(
                f"  {C_GREEN}\u2714{C_RESET} {u['username']}"
                f" {C_DIM}(team: {u['team_name']}){C_RESET}"
            )
    print()


def cmd_user_delete(args: argparse.Namespace) -> None:
    """Delete a user."""
    if not args.yes:
        answer = input(f"Delete user '{args.username}'? [y/N] ")
        if answer.lower() != "y":
            print("Cancelled.")
            return

    conn = get_db(config.DB_PATH)
    deleted = delete_user(conn, args.username)
    conn.close()

    if deleted:
        print(f"\n  {C_GREEN}\u2714{C_RESET}  User '{args.username}' deleted.\n")
    else:
        print(f"\n  {C_YELLOW}\u26a0\ufe0f{C_RESET}  User '{args.username}' not found.\n")


# === Team management commands ===


def cmd_team_create(args: argparse.Namespace) -> None:
    """Create a new team."""
    conn = get_db(config.DB_PATH)
    team_id = create_team(conn, args.name, args.description)
    config.ensure_team_dirs(args.name)
    conn.close()
    name = args.name
    print(f"\n  {C_GREEN}\u2714{C_RESET}  Team '{C_BOLD}{name}{C_RESET}' created (id={team_id})")
    print(f"  Use: meetscribe -t {name} enroll ...\n")


def cmd_team_list(args: argparse.Namespace) -> None:
    """List all teams."""
    conn = get_db(config.DB_PATH)
    teams = list_teams(conn)

    print(f"\n{C_MAGENTA}{'=' * 60}{C_RESET}")
    print(f"  \U0001f465 {C_BOLD}Teams{C_RESET}")
    print(f"{C_MAGENTA}{'=' * 60}{C_RESET}\n")

    for team in teams:
        vp_count = count_voiceprints(conn, team["id"])
        desc = f" - {team['description']}" if team["description"] else ""
        name = team["name"]
        print(f"  {C_GREEN}\u2714{C_RESET} {name}{C_DIM}{desc}{C_RESET} ({vp_count} speakers)")
    print()
    conn.close()


def cmd_team_delete(args: argparse.Namespace) -> None:
    """Delete a team."""
    if not args.yes:
        answer = input(f"Delete team '{args.name}' and all its data? [y/N] ")
        if answer.lower() != "y":
            print("Cancelled.")
            return

    conn = get_db(config.DB_PATH)
    deleted = delete_team(conn, args.name)
    conn.close()

    if deleted:
        # Remove team directories
        team_dir = config.TEAMS_DIR / args.name
        if team_dir.exists():
            shutil.rmtree(team_dir)
        print(f"\n  {C_GREEN}\u2714{C_RESET}  Team '{args.name}' deleted.\n")
    else:
        print(f"\n  {C_YELLOW}\u26a0\ufe0f{C_RESET}  Team '{args.name}' not found.\n")


def main() -> None:
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
    parser.add_argument(
        "-t", "--team", default=None, help="Team profile to use (default: 'default')"
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
    p.add_argument("-l", "--language", default=None, help="Language (default: from config.yaml)")
    p.set_defaults(func=cmd_transcribe)

    # info
    p = subs.add_parser("info", help="Show configuration and data directories")
    p.set_defaults(func=cmd_info)

    # web
    p = subs.add_parser("web", help="Start web UI")
    p.add_argument("--host", default=None, help="Host to bind (default: from config.yaml)")
    p.add_argument("--port", type=int, default=None, help="Port (default: from config.yaml)")
    p.set_defaults(func=cmd_web)

    # team management
    team_parser = subs.add_parser("team", help="Manage team profiles")
    team_subs = team_parser.add_subparsers(dest="team_command", required=True)

    p = team_subs.add_parser("create", help="Create a new team")
    p.add_argument("name", help="Team name")
    p.add_argument("--description", default=None, help="Team description")
    p.set_defaults(func=cmd_team_create)

    p = team_subs.add_parser("list", help="List all teams")
    p.set_defaults(func=cmd_team_list)

    p = team_subs.add_parser("delete", help="Delete a team")
    p.add_argument("name", help="Team name")
    p.add_argument("--yes", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_team_delete)

    # user management
    user_parser = subs.add_parser("user", help="Manage users")
    user_subs = user_parser.add_subparsers(dest="user_command", required=True)

    p = user_subs.add_parser("create", help="Create a new user")
    p.add_argument("username", help="Username")
    p.add_argument("--team", required=True, help="Team name")
    p.add_argument("--admin", action="store_true", help="Grant admin privileges")
    p.set_defaults(func=cmd_user_create)

    p = user_subs.add_parser("list", help="List all users")
    p.set_defaults(func=cmd_user_list)

    p = user_subs.add_parser("delete", help="Delete a user")
    p.add_argument("username", help="Username")
    p.add_argument("--yes", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_user_delete)

    args, extra = parser.parse_known_args()

    # Commands that don't need team context
    NO_TEAM_COMMANDS = {
        cmd_team_create,
        cmd_team_list,
        cmd_team_delete,
        cmd_user_create,
        cmd_user_list,
        cmd_user_delete,
        cmd_extract,
        cmd_web,
    }

    try:
        if args.func in NO_TEAM_COMMANDS:
            if extra:
                parser.error(f"Unrecognized arguments: {' '.join(extra)}")
            args.func(args)
        elif args.func == cmd_transcribe:
            team_ctx = resolve_team(args.team)
            args.func(args, extra, team_ctx)
        else:
            if extra:
                parser.error(f"Unrecognized arguments: {' '.join(extra)}")
            team_ctx = resolve_team(args.team)
            args.func(args, team_ctx)
    except ConfigurationError as e:
        print(f"\n  {C_RED}\u274c Configuration error:{C_RESET} {e}")
        raise SystemExit(1)
    except SpeachesAPIError as e:
        print(f"\n  {C_RED}\u274c API error:{C_RESET} {e}")
        if e.detail:
            print(f"    {e.detail}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n  {C_RED}\u274c Error:{C_RESET} {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
