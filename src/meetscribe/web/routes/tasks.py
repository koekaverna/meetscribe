"""Long-running task routes with SSE progress."""

import asyncio
import collections
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_COMPLETED_TASK_TTL = 60.0
_SSE_KEEPALIVE_INTERVAL = 5.0
_EVENT_LOG_MAX = 500


@dataclass
class _Subscriber:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None


@dataclass
class RunningTask:
    task_type: str
    session_id: str
    thread: threading.Thread | None = None
    on_complete: Callable[..., Any] | None = None
    on_error: Callable[..., Any] | None = None
    subscribers: list[_Subscriber] = field(default_factory=list)
    event_log: collections.deque[dict] = field(
        default_factory=lambda: collections.deque(maxlen=_EVENT_LOG_MAX)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    done_at: float | None = None
    error: str | None = None
    on_complete_ran: bool = False


_running_tasks: dict[tuple[str, str], RunningTask] = {}
_tasks_lock = threading.Lock()


def _prune_completed() -> None:
    now = time.monotonic()
    to_remove = [
        key
        for key, task in _running_tasks.items()
        if task.done and task.done_at and (now - task.done_at) > _COMPLETED_TASK_TTL
    ]
    for key in to_remove:
        del _running_tasks[key]


def _get_task(session_id: str, task_type: str) -> RunningTask | None:
    with _tasks_lock:
        _prune_completed()
        return _running_tasks.get((session_id, task_type))


def _register_task(task: RunningTask) -> RunningTask | None:
    """Register a new task. Returns existing running task if one exists, else None."""
    with _tasks_lock:
        _prune_completed()
        key = (task.session_id, task.task_type)
        existing = _running_tasks.get(key)
        if existing and not existing.done:
            return existing
        _running_tasks[key] = task
        return None


def shutdown_threads(timeout: float = 30.0) -> None:
    """Wait for active background threads to finish. Called on app shutdown."""
    with _tasks_lock:
        threads = [t.thread for t in _running_tasks.values() if t.thread and t.thread.is_alive()]
    for thread in threads:
        thread.join(timeout=timeout / max(len(threads), 1))
        if thread.is_alive():
            logger.warning("Thread did not finish within timeout", extra={"thread": thread.name})


# ---------------------------------------------------------------------------
# Event cleaning
# ---------------------------------------------------------------------------


def _clean_event(item: dict) -> dict:
    cleaned = {k: v for k, v in item.items() if k != "audio_bytes"}
    if "samples" in cleaned:
        cleaned["samples"] = [
            {k: v for k, v in s.items() if k != "audio_bytes"} for s in cleaned["samples"]
        ]
    return cleaned


# ---------------------------------------------------------------------------
# Background thread → asyncio bridge
# ---------------------------------------------------------------------------


def _deliver(subscribers: list[_Subscriber], msg: tuple[str, Any]) -> None:
    """Send a message to a pre-captured list of subscribers."""
    for sub in subscribers:
        loop = sub.loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(sub.queue.put_nowait, msg)
            except RuntimeError:
                pass


def _broadcast(task: RunningTask, msg: tuple[str, Any]) -> None:
    with task.lock:
        subs = list(task.subscribers)
    _deliver(subs, msg)


def _run_with_broadcast(task: RunningTask, gen: Generator[dict, None, None]) -> None:
    from meetscribe.database import close_db

    try:
        try:
            for item in gen:
                cleaned = _clean_event(item)
                with task.lock:
                    task.event_log.append(cleaned)
                    subs = list(task.subscribers)
                _deliver(subs, ("item", cleaned))

            with task.lock:
                task.done = True
                task.done_at = time.monotonic()
                subs = list(task.subscribers)
            _deliver(subs, ("done", None))

        except Exception as e:
            logger.exception("Task %s failed for session %s", task.task_type, task.session_id)
            with task.lock:
                task.done = True
                task.done_at = time.monotonic()
                task.error = str(e)
                subs = list(task.subscribers)
            _deliver(subs, ("error", str(e)))

        # Run callbacks
        with task.lock:
            if task.on_complete_ran:
                return
            task.on_complete_ran = True

        if task.error:
            if task.on_error:
                try:
                    task.on_error(task.error)
                except Exception:
                    logger.exception("on_error callback failed for %s", task.task_type)
        else:
            if task.on_complete:
                try:
                    task.on_complete()
                except Exception:
                    logger.exception("on_complete callback failed for %s", task.task_type)
    finally:
        close_db()


# ---------------------------------------------------------------------------
# SSE subscriber
# ---------------------------------------------------------------------------


async def _subscribe_to_task(task: RunningTask) -> AsyncGenerator[str, None]:
    sub = _Subscriber(loop=asyncio.get_running_loop())

    # Snapshot under lock — never yield while holding it
    with task.lock:
        replay_events = list(task.event_log)
        is_done = task.done
        if not is_done:
            task.subscribers.append(sub)

    for event in replay_events:
        yield f"data: {json.dumps(event)}\n\n"

    if is_done:
        if task.error:
            yield f"data: {json.dumps({'error': task.error})}\n\n"
        else:
            yield 'data: {"done": true}\n\n'
        return

    try:
        while True:
            try:
                msg_type, data = await asyncio.wait_for(
                    sub.queue.get(), timeout=_SSE_KEEPALIVE_INTERVAL
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue

            if msg_type == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"
                break

            if msg_type == "done":
                yield 'data: {"done": true}\n\n'
                break

            if msg_type == "item":
                yield f"data: {json.dumps(data)}\n\n"
    finally:
        with task.lock:
            if sub in task.subscribers:
                task.subscribers.remove(sub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_THRESHOLDS: dict[str, SessionStatus] = {
    "extract": SessionStatus.EXTRACTED,
    "enroll": SessionStatus.ENROLLED,
    "transcribe": SessionStatus.TRANSCRIBED,
}
_STATUS_ORDER = list(SessionStatus)


def _stream_or_404(session_id: str, task_type: str, user: AuthUser) -> StreamingResponse:
    state = get_session_for_user(session_id, user)

    task = _get_task(session_id, task_type)
    if task:
        return StreamingResponse(_subscribe_to_task(task), media_type="text/event-stream")

    threshold = _STATUS_THRESHOLDS[task_type]
    if _STATUS_ORDER.index(state.status) >= _STATUS_ORDER.index(threshold):

        async def _done_stream() -> AsyncGenerator[str, None]:
            yield 'data: {"done": true}\n\n'

        return StreamingResponse(_done_stream(), media_type="text/event-stream")

    raise HTTPException(status_code=404, detail="No task running")


def _start_task(
    session_id: str,
    task_type: str,
    gen: Generator[dict, None, None],
    on_complete: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Register and start a background task."""
    task = RunningTask(
        task_type=task_type,
        session_id=session_id,
        on_complete=on_complete,
        on_error=on_error,
    )
    if _register_task(task):
        return {"status": "already_running"}

    thread = threading.Thread(
        target=_run_with_broadcast,
        args=(task, gen),
        daemon=True,
        name=f"{task_type}-{session_id[:8]}",
    )
    task.thread = thread
    thread.start()
    return {"status": "started"}


# ===================================================================
# Extraction
# ===================================================================


@router.post("/{session_id}/extract")
def start_extraction(session_id: str, user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    state = get_session_for_user(session_id, user)
    service = get_session_service()
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")

    track_paths = []
    track_diarize = {}
    for track in state.tracks:
        path = service.get_track_path(session_id, track.track_num)
        if path:
            track_paths.append(path)
            track_diarize[track.track_num] = track.diarize or not track.speaker_name

    if not track_paths:
        raise HTTPException(status_code=400, detail="No tracks found")

    runner = get_pipeline_runner(state.team_name)
    samples_data: list[dict] = []

    def extraction_gen() -> Generator[dict, None, None]:
        for item in runner.extract_samples(
            track_paths, track_diarize=track_diarize, progress_callback=True
        ):
            if "samples" in item:
                samples_data.extend(item["samples"])
            yield item

    def on_complete() -> None:
        for sample_info in samples_data:
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
        updated_state = service.get(session_id)
        if updated_state:
            updated_state.status = SessionStatus.EXTRACTED
            service.update(updated_state)

    def on_error(error_msg: str) -> None:
        logger.error("Extraction failed", extra={"session_id": session_id, "error": error_msg})

    return _start_task(session_id, "extract", extraction_gen(), on_complete, on_error)


@router.get("/{session_id}/extract/stream")
async def stream_extraction(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    return _stream_or_404(session_id, "extract", user)


# ===================================================================
# Enrollment
# ===================================================================


@router.post("/{session_id}/enroll")
def start_enrollment(session_id: str, user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    session_state = get_session_for_user(session_id, user)
    service = get_session_service()
    if not session_state.speakers:
        raise HTTPException(status_code=400, detail="No speakers defined")

    runner = get_pipeline_runner(session_state.team_name)

    def enrollment_gen() -> Generator[dict, None, None]:
        for speaker in session_state.speakers:
            sample_paths = []
            for sample in session_state.samples:
                if sample.speaker_id == speaker.id:
                    path = service.get_sample_path(session_id, sample.id)
                    if path:
                        sample_paths.append(path)
            if not sample_paths:
                yield {"message": f"Skipping {speaker.name}: no samples"}
                continue
            yield from runner.enroll_speaker(speaker.name, sample_paths, progress_callback=True)

    def on_complete() -> None:
        updated_state = service.get(session_id)
        if updated_state:
            updated_state.status = SessionStatus.ENROLLED
            service.update(updated_state)

    def on_error(error_msg: str) -> None:
        logger.error("Enrollment failed", extra={"session_id": session_id, "error": error_msg})

    return _start_task(session_id, "enroll", enrollment_gen(), on_complete, on_error)


@router.get("/{session_id}/enroll/stream")
async def stream_enrollment(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    return _stream_or_404(session_id, "enroll", user)


# ===================================================================
# Transcription
# ===================================================================


@router.post("/{session_id}/transcribe")
def start_transcription(
    session_id: str, options: TranscribeOptions, user: AuthUser = Depends(get_current_user)
) -> dict[str, str]:
    state = get_session_for_user(session_id, user)
    service = get_session_service()
    if not state.tracks:
        raise HTTPException(status_code=400, detail="No tracks uploaded")

    state.language = options.language
    service.update(state)

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
    transcript_data: dict = {"transcript": None, "segments": None}

    def transcription_gen() -> Generator[dict, None, None]:
        for item in runner.transcribe(
            track_paths, track_speakers, state.language, progress_callback=True
        ):
            if "transcript" in item:
                transcript_data["transcript"] = item["transcript"]
            if "segments" in item:
                transcript_data["segments"] = item["segments"]
            yield item

    def on_complete() -> None:
        if transcript_data["transcript"]:
            service.set_transcript(session_id, transcript_data["transcript"])
        if transcript_data["segments"]:
            service.save_segments(session_id, transcript_data["segments"])

    def on_error(error_msg: str) -> None:
        logger.error("Transcription failed", extra={"session_id": session_id, "error": error_msg})

    return _start_task(session_id, "transcribe", transcription_gen(), on_complete, on_error)


@router.get("/{session_id}/transcribe/stream")
async def stream_transcription(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> StreamingResponse:
    return _stream_or_404(session_id, "transcribe", user)


# ===================================================================
# Task status & transcript
# ===================================================================


@router.get("/{session_id}/tasks/status")
def get_tasks_status(
    session_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, dict | None]:
    get_session_for_user(session_id, user)
    result: dict[str, dict | None] = {}
    for task_type in ("extract", "enroll", "transcribe"):
        task = _get_task(session_id, task_type)
        if task and not task.done:
            with task.lock:
                progress = task.event_log[-1] if task.event_log else {}
            result[task_type] = {"running": True, "progress": progress}
        else:
            result[task_type] = None
    return result


@router.get("/{session_id}/transcript")
def get_transcript(session_id: str, user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    state = get_session_for_user(session_id, user)
    if not state.transcript:
        raise HTTPException(status_code=404, detail="Transcript not available")
    return {"transcript": state.transcript}
