"""Long-running task routes with SSE progress."""

import asyncio
import json
import queue
import threading
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import SessionStatus, TranscribeOptions
from ..services.pipeline import get_pipeline_runner
from ..services.session import get_session_service

router = APIRouter()


def run_generator_with_queue(gen_func, result_queue: queue.Queue, *args, **kwargs):
    """Run a generator in a thread and put results in a queue."""
    try:
        for item in gen_func(*args, **kwargs):
            result_queue.put(("item", item))
        result_queue.put(("done", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


async def stream_from_queue(
    result_queue: queue.Queue,
    on_complete=None,
    check_interval: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Stream results from a queue as SSE events."""
    last_result = None

    while True:
        try:
            # Non-blocking check
            msg_type, data = result_queue.get_nowait()

            if msg_type == "error":
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
async def start_extraction(session_id: str):
    """Start sample extraction (returns immediately, use SSE endpoint for progress)."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")
    return {"status": "started"}


@router.get("/{session_id}/extract/stream")
async def stream_extraction(session_id: str):
    """Stream extraction progress via SSE."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

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

    runner = get_pipeline_runner()
    result_queue: queue.Queue = queue.Queue()

    # Store samples data for on_complete
    samples_data = []

    def extraction_wrapper():
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
    thread = threading.Thread(target=extraction_wrapper, daemon=True)
    thread.start()

    async def on_complete(final_result):
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

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete),
        media_type="text/event-stream",
    )


@router.post("/{session_id}/enroll")
async def start_enrollment(session_id: str):
    """Start speaker enrollment (returns immediately, use SSE endpoint for progress)."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    if not state.speakers:
        raise HTTPException(status_code=400, detail="No speakers defined")
    return {"status": "started"}


@router.get("/{session_id}/enroll/stream")
async def stream_enrollment(session_id: str):
    """Stream enrollment progress via SSE."""
    service = get_session_service()
    session_state = service.get(session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found")

    runner = get_pipeline_runner()
    result_queue: queue.Queue = queue.Queue()

    def enrollment_wrapper():
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
                    result_queue.put((
                        "item",
                        {"message": f"Skipping {speaker.name}: no samples"},
                    ))
                    continue

                for item in runner.enroll_speaker(
                    speaker.name, sample_paths, progress_callback=True
                ):
                    result_queue.put(("item", item))

            result_queue.put(("done", None))
        except Exception as e:
            result_queue.put(("error", str(e)))

    thread = threading.Thread(target=enrollment_wrapper, daemon=True)
    thread.start()

    async def on_complete(final_result):
        """Update status when done."""
        updated_state = service.get(session_id)
        if updated_state:
            updated_state.status = SessionStatus.ENROLLED
            service.update(updated_state)

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete),
        media_type="text/event-stream",
    )


@router.post("/{session_id}/transcribe")
async def start_transcription(session_id: str, options: TranscribeOptions):
    """Start transcription."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")

    # Save options
    state.whisper_model = options.model
    state.language = options.language
    service.update(state)

    return {"status": "started"}


@router.get("/{session_id}/transcribe/stream")
async def stream_transcription(session_id: str):
    """Stream transcription progress via SSE."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get track paths and speaker assignments
    track_paths = []
    track_speakers = {}
    for track in state.tracks:
        path = service.get_track_path(session_id, track.track_num)
        if path:
            track_paths.append(path)
            track_speakers[track.track_num] = (
                track.speaker_name if not track.diarize else None
            )

    if not track_paths:
        raise HTTPException(status_code=400, detail="No tracks found")

    runner = get_pipeline_runner()
    result_queue: queue.Queue = queue.Queue()
    transcript_data = {"transcript": None}

    def transcription_wrapper():
        """Run transcription."""
        import traceback
        try:
            for item in runner.transcribe(
                track_paths,
                track_speakers,
                state.whisper_model,
                state.language,
                progress_callback=True,
                progress_queue=result_queue,
            ):
                if "transcript" in item:
                    transcript_data["transcript"] = item["transcript"]
                result_queue.put(("item", item))
            result_queue.put(("done", None))
        except Exception as e:
            traceback.print_exc()
            result_queue.put(("error", str(e)))

    thread = threading.Thread(target=transcription_wrapper, daemon=True)
    thread.start()

    async def on_complete(final_result):
        """Save transcript when done."""
        if transcript_data["transcript"]:
            service.set_transcript(session_id, transcript_data["transcript"])

    return StreamingResponse(
        stream_from_queue(result_queue, on_complete=on_complete),
        media_type="text/event-stream",
    )


@router.get("/{session_id}/transcript")
async def get_transcript(session_id: str):
    """Get the transcript."""
    service = get_session_service()
    state = service.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    if not state.transcript:
        raise HTTPException(status_code=404, detail="Transcript not available")
    return {"transcript": state.transcript}
