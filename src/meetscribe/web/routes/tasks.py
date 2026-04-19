"""Long-running task routes with SSE progress."""

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import get_current_user, get_session_for_user
from ..models import SessionStatus, TranscribeOptions
from ..services.auth import AuthUser
from ..services.pipeline import get_pipeline_runner
from ..services.session import get_session_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active background threads for graceful shutdown
_active_threads: list[threading.Thread] = []
_threads_lock = threading.Lock()


def _register_thread(thread: threading.Thread) -> None:
    with _threads_lock:
        # Clean up finished threads
        _active_threads[:] = [t for t in _active_threads if t.is_alive()]
        _active_threads.append(thread)


def shutdown_threads(timeout: float = 30.0) -> None:
    """Wait for active background threads to finish. Called on app shutdown."""
    with _threads_lock:
        threads = list(_active_threads)
    for thread in threads:
        thread.join(timeout=timeout / max(len(threads), 1))
        if thread.is_alive():
            logger.warning("Thread did not finish within timeout", extra={"thread": thread.name})


def run_generator_with_queue(
    gen_func: Any, result_queue: queue.Queue[Any], *args: Any, **kwargs: Any
) -> None:
    """Run a generator in a thread and put results in a queue."""
    try:
        for item in gen_func(*args, **kwargs):
            result_queue.put(("item", item))
        result_queue.put(("done", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


async def stream_from_queue(
    result_queue: queue.Queue[Any],
    on_complete: Callable[..., Any] | None = None,
    on_error: Callable[..., Any] | None = None,
    check_interval: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Stream results from a queue as SSE events."""
    last_result = None

    while True:
        try:
            # Non-blocking check
            msg_type, data = result_queue.get_nowait()

            if msg_type == "error":
                logger.error("Task error", extra={"error": data})
                if on_error:
                    await on_error(data)
                yield f"data: {json.dumps({'error': data})}\n\n"
                break

            if msg_type == "done":
                if on_complete and last_result:
                    await on_complete(last_result)
                yield 'data: {"done": true}\n\n'
                break

            if msg_type == "item":
                last_result = data
                # Don't send large binary data over SSE
                item_to_send = {k: v for k, v in data.items() if k != "audio_bytes"}
                if "samples" in item_to_send:
                    item_to_send["samples"] = [
                        {k: v for k, v in s.items() if k != "audio_bytes"}
                        for s in item_to_send["samples"]
                    ]
                yield f"data: {json.dumps(item_to_send)}\n\n"

        except queue.Empty:
            # No message yet, wait a bit
            await asyncio.sleep(check_interval)


@router.post("/{session_id}/extract")
async def start_extraction(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Start sample extraction (returns immediately, use SSE endpoint for progress)."""
    state = get_session_for_user(session_id, user)
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")
    return {"status": "started"}


@router.get("/{session_id}/extract/stream")
async def stream_extraction(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    """Stream extraction progress via SSE."""
    state = get_session_for_user(session_id, user)
    service = get_session_service()

    # Get track paths and diarization config
    track_paths = []
    track_diarize = {}
    for track in state.tracks:
        path = service.get_track_path(session_id, track.track_num)
        if path:
            track_paths.append(path)
            # Only diarize if: diarize=True OR no speaker name assigned
            track_diarize[track.track_num] = track.diarize or not track.speaker_name

    if not track_paths:
        raise HTTPException(status_code=400, detail="No tracks found")

    runner = get_pipeline_runner(state.team_name)
    result_queue: queue.Queue = queue.Queue()

    # Store samples data for on_complete
    samples_data = []

    def extraction_wrapper() -> None:
        """Wrapper that captures samples data."""
        try:
            for item in runner.extract_samples(
                track_paths, track_diarize=track_diarize, progress_callback=True
            ):
                if "samples" in item:
                    samples_data.extend(item["samples"])
                result_queue.put(("item", item))
            result_queue.put(("done", None))
        except Exception as e:
            result_queue.put(("error", str(e)))

    # Start extraction in background thread
    thread = threading.Thread(
        target=extraction_wrapper, daemon=True, name=f"extract-{session_id[:8]}"
    )
    thread.start()
    _register_thread(thread)

    async def on_complete(final_result: Any) -> None:
        """Save samples when extraction completes."""
        for sample_info in samples_data:
            # Skip saving known samples - they're already in enrolled folder
            if sample_info.get("is_known", False):
                continue

            service.add_sample(
                session_id,
                sample_info["track_num"],
                sample_info["cluster_id"],
                sample_info["filename"],
                sample_info["duration_ms"],
                sample_info["audio_bytes"],
                is_known=sample_info.get("is_known", False),
                known_speaker_name=sample_info.get("known_speaker_name"),
            )

        # Update session status
        updated_state = service.get(session_id)
        if updated_state:
            updated_state.status = SessionStatus.EXTRACTED
            service.update(updated_state)

    async def on_error(error_msg: str) -> None:
        """Rollback status on extraction failure."""
        logger.error("Extraction failed", extra={"session_id": session_id, "error": error_msg})

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete, on_error=on_error),
        media_type="text/event-stream",
    )


@router.post("/{session_id}/enroll")
async def start_enrollment(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Start speaker enrollment (returns immediately, use SSE endpoint for progress)."""
    state = get_session_for_user(session_id, user)
    if not state.speakers:
        raise HTTPException(status_code=400, detail="No speakers defined")
    return {"status": "started"}


@router.get("/{session_id}/enroll/stream")
async def stream_enrollment(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    """Stream enrollment progress via SSE."""
    session_state = get_session_for_user(session_id, user)
    service = get_session_service()

    runner = get_pipeline_runner(session_state.team_name)
    result_queue: queue.Queue = queue.Queue()

    def enrollment_wrapper() -> None:
        """Run enrollment for all speakers."""
        try:
            for speaker in session_state.speakers:
                sample_paths = []
                for sample in session_state.samples:
                    if sample.speaker_id == speaker.id:
                        path = service.get_sample_path(session_id, sample.id)
                        if path:
                            sample_paths.append(path)

                if not sample_paths:
                    result_queue.put(
                        (
                            "item",
                            {"message": f"Skipping {speaker.name}: no samples"},
                        )
                    )
                    continue

                for item in runner.enroll_speaker(
                    speaker.name, sample_paths, progress_callback=True
                ):
                    result_queue.put(("item", item))

            result_queue.put(("done", None))
        except Exception as e:
            result_queue.put(("error", str(e)))

    thread = threading.Thread(
        target=enrollment_wrapper, daemon=True, name=f"enroll-{session_id[:8]}"
    )
    thread.start()
    _register_thread(thread)

    async def on_complete(final_result: Any) -> None:
        """Update status when done."""
        updated_state = service.get(session_id)
        if updated_state:
            updated_state.status = SessionStatus.ENROLLED
            service.update(updated_state)

    async def on_error(error_msg: str) -> None:
        """Log enrollment failure."""
        logger.error("Enrollment failed", extra={"session_id": session_id, "error": error_msg})

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete, on_error=on_error),
        media_type="text/event-stream",
    )


@router.post("/{session_id}/transcribe")
async def start_transcription(
    session_id: str, options: TranscribeOptions, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Start transcription."""
    state = get_session_for_user(session_id, user)
    service = get_session_service()
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")

    # Save options
    state.language = options.language
    service.update(state)

    return {"status": "started"}


@router.get("/{session_id}/transcribe/stream")
async def stream_transcription(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    """Stream transcription progress via SSE."""
    state = get_session_for_user(session_id, user)
    service = get_session_service()

    # Get track paths and speaker assignments
    track_paths = []
    track_speakers = {}
    for track in state.tracks:
        path = service.get_track_path(session_id, track.track_num)
        if path:
            track_paths.append(path)
            track_speakers[track.track_num] = track.speaker_name if not track.diarize else None

    if not track_paths:
        raise HTTPException(status_code=400, detail="No tracks found")

    runner = get_pipeline_runner(state.team_name)
    result_queue: queue.Queue = queue.Queue()
    transcript_data: dict = {"transcript": None, "segments": None}

    def transcription_wrapper() -> None:
        """Run transcription."""
        try:
            for item in runner.transcribe(
                track_paths,
                track_speakers,
                state.language,
                progress_callback=True,
                progress_queue=result_queue,
            ):
                if "transcript" in item:
                    transcript_data["transcript"] = item["transcript"]
                if "segments" in item:
                    transcript_data["segments"] = item["segments"]
                result_queue.put(("item", item))
            result_queue.put(("done", None))
        except Exception as e:
            logger.exception("Transcription failed")
            result_queue.put(("error", str(e)))

    thread = threading.Thread(
        target=transcription_wrapper, daemon=True, name=f"transcribe-{session_id[:8]}"
    )
    thread.start()
    _register_thread(thread)

    async def on_complete(final_result: Any) -> None:
        """Save transcript when done."""
        if transcript_data["transcript"]:
            service.set_transcript(session_id, transcript_data["transcript"])
        if transcript_data["segments"]:
            service.save_segments(session_id, transcript_data["segments"])

    async def on_error(error_msg: str) -> None:
        """Log transcription failure."""
        logger.error("Transcription failed", extra={"session_id": session_id, "error": error_msg})

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete, on_error=on_error),
        media_type="text/event-stream",
    )


@router.get("/{session_id}/transcript")
async def get_transcript(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    """Get the transcript."""
    state = get_session_for_user(session_id, user)
    if not state.transcript:
        raise HTTPException(status_code=404, detail="Transcript not available")
    return {"transcript": state.transcript}
