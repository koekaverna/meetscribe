"""Track upload and management routes."""

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from meetscribe import config
from meetscribe.pipeline import audio

from ..deps import get_current_user, get_session_for_user
from ..models import TrackConfig, TrackUploadResponse
from ..services.auth import AuthUser
from ..services.session import SessionService, get_session_service

logger = logging.getLogger(__name__)

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1 MB

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


async def _stream_to_file(upload: UploadFile, dest: Path, max_size: int) -> int:
    """Stream an UploadFile to disk in chunks. Returns bytes written."""
    total = 0
    oversized = False
    with open(dest, "wb") as f:
        while chunk := await upload.read(CHUNK_SIZE):
            total += len(chunk)
            if total > max_size:
                oversized = True
                break
            f.write(chunk)
    if oversized:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="File too large")
    return total


@router.post("/{session_id}/tracks", response_model=list[TrackUploadResponse])
async def upload_tracks(
    session_id: str,
    files: list[UploadFile] = File(...),
    user: AuthUser = Depends(get_current_user),
) -> list[TrackUploadResponse]:
    """Upload track files (video or audio)."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    tracks_dir = service._tracks_dir(session_id)

    responses = []

    for file in files:
        raw_name = file.filename or "unknown"
        # Sanitize: take basename only, replace unsafe chars
        filename = re.sub(r"[^\w.\-]", "_", Path(raw_name).name) or "upload"
        ext = Path(filename).suffix.lower()

        if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        # Stream upload to temp file in tracks dir (same filesystem → O(1) move)
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=ext, dir=tracks_dir)
        tmp_path = Path(tmp_name)
        try:
            os.close(tmp_fd)
            await _stream_to_file(file, tmp_path, config.MAX_UPLOAD_SIZE)

            if ext in VIDEO_EXTS:
                responses.extend(
                    await _process_video(service, session_id, filename, tmp_path, tracks_dir)
                )
            elif ext == ".wav":
                # WAV — move directly to session storage
                track = service.add_track(session_id, filename, tmp_path)
                tmp_path = None  # type: ignore[assignment]  # moved, don't unlink
                responses.append(
                    TrackUploadResponse(track_num=track.track_num, filename=track.filename)
                )
            else:
                # Non-WAV audio — convert then move
                wav_fd, wav_name = tempfile.mkstemp(suffix=".wav", dir=tracks_dir)
                wav_path = Path(wav_name)
                os.close(wav_fd)
                try:
                    await audio.convert_to_wav_async(tmp_path, wav_path)  # type: ignore[arg-type]
                    track = service.add_track(session_id, filename, wav_path)
                    wav_path = None  # type: ignore[assignment]  # moved
                    responses.append(
                        TrackUploadResponse(track_num=track.track_num, filename=track.filename)
                    )
                finally:
                    if wav_path is not None:
                        try:
                            wav_path.unlink(missing_ok=True)
                        except OSError:
                            logger.warning("Failed to clean up temp file: %s", wav_path)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", tmp_path)

    return responses


async def _process_video(
    service: SessionService, session_id: str, filename: str, video_path: Path, tracks_dir: Path
) -> list[TrackUploadResponse]:
    """Extract audio tracks from video, return responses."""
    stream_indices = await audio.probe_audio_tracks_async(video_path)
    if not stream_indices:
        raise HTTPException(status_code=400, detail=f"No audio tracks in {filename}")

    # Extract all tracks in parallel
    wav_paths: list[Path] = []
    tasks = []
    for stream_idx in stream_indices:
        wav_fd, wav_name = tempfile.mkstemp(suffix=".wav", dir=tracks_dir)
        wav_path = Path(wav_name)
        os.close(wav_fd)
        wav_paths.append(wav_path)
        tasks.append(audio.extract_audio_async(video_path, wav_path, stream_idx))

    try:
        await asyncio.gather(*tasks)
    except Exception:
        # Clean up all wav temp files on failure
        for wp in wav_paths:
            wp.unlink(missing_ok=True)
        raise

    responses = []
    stem = Path(filename).stem
    for i, wav_path in enumerate(wav_paths):
        try:
            track = service.add_track(session_id, f"{stem}_track{i + 1}.wav", wav_path)
            responses.append(
                TrackUploadResponse(track_num=track.track_num, filename=track.filename)
            )
        except Exception:
            # Clean up current and all remaining wav temp files
            for wp in wav_paths[i:]:
                wp.unlink(missing_ok=True)
            raise

    return responses


@router.get("/{session_id}/tracks", response_model=list[TrackConfig])
async def list_tracks(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> list[TrackConfig]:
    """List tracks in session."""
    state = get_session_for_user(session_id, user)
    return state.tracks


@router.get("/{session_id}/tracks/{track_num}/audio")
async def get_track_audio(
    session_id: str, track_num: int, user: AuthUser = Depends(get_current_user)
) -> FileResponse:
    """Stream track audio."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    path = service.get_track_path(session_id, track_num)
    if not path:
        raise HTTPException(status_code=404, detail="Track not found")
    return FileResponse(path, media_type="audio/wav")


@router.patch("/{session_id}/tracks/{track_num}")
async def update_track(
    session_id: str,
    track_num: int,
    speaker_name: str | None = None,
    diarize: bool = True,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, str]:
    """Update track configuration."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    if not service.update_track_config(session_id, track_num, speaker_name, diarize):
        raise HTTPException(status_code=404, detail="Track not found")
    return {"status": "updated"}


@router.delete("/{session_id}/tracks/{track_num}")
async def delete_track(
    session_id: str, track_num: int, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Delete a track."""
    get_session_for_user(session_id, user)
    service = get_session_service()
    if not service.remove_track(session_id, track_num):
        raise HTTPException(status_code=404, detail="Track not found")
    return {"status": "deleted"}
