"""Track upload and management routes."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..models import TrackConfig, TrackUploadResponse
from ..services.pipeline import convert_to_wav, extract_audio_track, probe_audio_tracks
from ..services.session import get_session_service

router = APIRouter()

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


@router.post("/{session_id}/tracks", response_model=list[TrackUploadResponse])
async def upload_tracks(session_id: str, files: list[UploadFile] = File(...)):
    """Upload track files (video or audio)."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    responses = []

    for file in files:
        filename = file.filename or "unknown"
        ext = Path(filename).suffix.lower()

        content = await file.read()

        if ext in VIDEO_EXTS:
            # Extract tracks from video
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                stream_indices = probe_audio_tracks(tmp_path)
                if not stream_indices:
                    raise HTTPException(status_code=400, detail=f"No audio tracks in {filename}")

                for i, stream_idx in enumerate(stream_indices):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                        wav_path = Path(wav_tmp.name)

                    extract_audio_track(tmp_path, wav_path, stream_idx)
                    wav_content = wav_path.read_bytes()
                    wav_path.unlink()

                    track = service.add_track(
                        session_id, f"{Path(filename).stem}_track{i + 1}.wav", wav_content
                    )
                    responses.append(
                        TrackUploadResponse(track_num=track.track_num, filename=track.filename)
                    )
            finally:
                tmp_path.unlink()

        elif ext in AUDIO_EXTS:
            if ext != ".wav":
                # Convert to WAV
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                    wav_path = Path(wav_tmp.name)

                try:
                    convert_to_wav(tmp_path, wav_path)
                    content = wav_path.read_bytes()
                finally:
                    tmp_path.unlink()
                    wav_path.unlink()

            track = service.add_track(session_id, filename, content)
            responses.append(
                TrackUploadResponse(track_num=track.track_num, filename=track.filename)
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    return responses


@router.get("/{session_id}/tracks", response_model=list[TrackConfig])
async def list_tracks(session_id: str):
    """List tracks in session."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.tracks


@router.get("/{session_id}/tracks/{track_num}/audio")
async def get_track_audio(session_id: str, track_num: int):
    """Stream track audio."""
    service = get_session_service()
    path = service.get_track_path(session_id, track_num)
    if not path:
        raise HTTPException(status_code=404, detail="Track not found")
    return FileResponse(path, media_type="audio/wav")


@router.patch("/{session_id}/tracks/{track_num}")
async def update_track(
    session_id: str, track_num: int, speaker_name: str | None = None, diarize: bool = True
):
    """Update track configuration."""
    service = get_session_service()
    if not service.update_track_config(session_id, track_num, speaker_name, diarize):
        raise HTTPException(status_code=404, detail="Track not found")
    return {"status": "updated"}


@router.delete("/{session_id}/tracks/{track_num}")
async def delete_track(session_id: str, track_num: int):
    """Delete a track."""
    service = get_session_service()
    if not service.remove_track(session_id, track_num):
        raise HTTPException(status_code=404, detail="Track not found")
    return {"status": "deleted"}
