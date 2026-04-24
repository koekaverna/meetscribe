"""Tests for task registry, broadcast, and SSE subscriber."""

import asyncio
import json
import threading
import time

import pytest

from meetscribe.web.routes.tasks import (
    RunningTask,
    _clean_event,
    _get_task,
    _register_task,
    _run_with_broadcast,
    _running_tasks,
    _start_task,
    _subscribe_to_task,
    _tasks_lock,
)


@pytest.fixture(autouse=True)
def _clear_task_registry():
    """Ensure clean task registry for each test."""
    with _tasks_lock:
        _running_tasks.clear()
    yield
    with _tasks_lock:
        _running_tasks.clear()


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


class TestRegisterTask:
    def test_registers_new_task(self):
        task = RunningTask(task_type="extract", session_id="s1")
        assert _register_task(task) is None
        assert _get_task("s1", "extract") is task

    def test_returns_existing_running_task(self):
        task1 = RunningTask(task_type="extract", session_id="s1")
        _register_task(task1)

        task2 = RunningTask(task_type="extract", session_id="s1")
        existing = _register_task(task2)
        assert existing is task1
        assert _get_task("s1", "extract") is task1

    def test_replaces_completed_task(self):
        task1 = RunningTask(task_type="extract", session_id="s1")
        _register_task(task1)
        task1.done = True
        task1.done_at = time.monotonic()

        task2 = RunningTask(task_type="extract", session_id="s1")
        assert _register_task(task2) is None
        assert _get_task("s1", "extract") is task2

    def test_different_sessions_independent(self):
        t1 = RunningTask(task_type="extract", session_id="s1")
        t2 = RunningTask(task_type="extract", session_id="s2")
        _register_task(t1)
        _register_task(t2)
        assert _get_task("s1", "extract") is t1
        assert _get_task("s2", "extract") is t2

    def test_different_task_types_independent(self):
        t1 = RunningTask(task_type="extract", session_id="s1")
        t2 = RunningTask(task_type="transcribe", session_id="s1")
        _register_task(t1)
        _register_task(t2)
        assert _get_task("s1", "extract") is t1
        assert _get_task("s1", "transcribe") is t2


class TestGetTask:
    def test_returns_none_for_missing(self):
        assert _get_task("no-such", "extract") is None

    def test_prunes_expired_tasks(self):
        task = RunningTask(task_type="extract", session_id="s1")
        _register_task(task)
        task.done = True
        task.done_at = time.monotonic() - 120  # well past TTL

        assert _get_task("s1", "extract") is None


class TestStartTask:
    def test_returns_started(self):
        def gen():
            yield {"step": 1}

        result = _start_task("s1", "extract", gen())
        assert result == {"status": "started"}
        # Wait for thread to finish
        task = _get_task("s1", "extract")
        assert task is not None
        task.thread.join(timeout=2)

    def test_dedup_returns_already_running(self):
        event = threading.Event()

        def slow_gen():
            yield {"step": 1}
            event.wait(timeout=5)

        _start_task("s1", "extract", slow_gen())
        result = _start_task("s1", "extract", slow_gen())
        assert result == {"status": "already_running"}
        event.set()
        _get_task("s1", "extract").thread.join(timeout=2)


# ---------------------------------------------------------------------------
# _run_with_broadcast
# ---------------------------------------------------------------------------


class TestRunWithBroadcast:
    def test_success_sets_done(self):
        def gen():
            yield {"step": 1, "message": "Working..."}
            yield {"step": 2, "message": "Done"}

        task = RunningTask(task_type="test", session_id="s1")
        _run_with_broadcast(task, gen())

        assert task.done is True
        assert task.error is None
        assert len(task.event_log) == 2
        assert task.event_log[0]["step"] == 1
        assert task.event_log[1]["step"] == 2

    def test_error_sets_error(self):
        def gen():
            yield {"step": 1}
            raise ValueError("boom")

        task = RunningTask(task_type="test", session_id="s1")
        _run_with_broadcast(task, gen())

        assert task.done is True
        assert task.error == "boom"
        assert len(task.event_log) == 1

    def test_on_complete_called_on_success(self):
        called = []

        def gen():
            yield {"step": 1}

        task = RunningTask(
            task_type="test", session_id="s1", on_complete=lambda: called.append("ok")
        )
        _run_with_broadcast(task, gen())
        assert called == ["ok"]

    def test_on_error_called_on_failure(self):
        errors = []

        def gen():
            raise RuntimeError("fail")
            yield  # make it a generator

        task = RunningTask(
            task_type="test", session_id="s1", on_error=lambda msg: errors.append(msg)
        )
        _run_with_broadcast(task, gen())
        assert errors == ["fail"]

    def test_on_complete_not_called_on_error(self):
        called = []

        def gen():
            raise RuntimeError("fail")
            yield

        task = RunningTask(
            task_type="test",
            session_id="s1",
            on_complete=lambda: called.append("complete"),
            on_error=lambda msg: called.append(f"error:{msg}"),
        )
        _run_with_broadcast(task, gen())
        assert called == ["error:fail"]

    def test_on_complete_ran_only_once(self):
        count = []

        def gen():
            yield {"step": 1}

        task = RunningTask(task_type="test", session_id="s1", on_complete=lambda: count.append(1))
        _run_with_broadcast(task, gen())
        # Simulate second call (shouldn't happen, but guard works)
        task.done = False
        _run_with_broadcast(task, iter([]))
        assert len(count) == 1


# ---------------------------------------------------------------------------
# _clean_event
# ---------------------------------------------------------------------------


class TestCleanEvent:
    def test_strips_audio_bytes(self):
        event = {"step": 1, "audio_bytes": b"\x00" * 100, "message": "hi"}
        cleaned = _clean_event(event)
        assert "audio_bytes" not in cleaned
        assert cleaned["message"] == "hi"

    def test_strips_audio_from_samples(self):
        event = {
            "samples": [
                {"filename": "s.wav", "audio_bytes": b"\x00"},
                {"filename": "t.wav", "audio_bytes": b"\xff"},
            ]
        }
        cleaned = _clean_event(event)
        for s in cleaned["samples"]:
            assert "audio_bytes" not in s
            assert "filename" in s

    def test_passthrough_without_audio(self):
        event = {"step": 1, "message": "hello"}
        assert _clean_event(event) == event


# ---------------------------------------------------------------------------
# _subscribe_to_task (async)
# ---------------------------------------------------------------------------


class TestSubscribeToTask:
    def _collect(self, async_gen) -> list[dict]:
        """Run async generator and parse SSE data lines."""
        results = []

        async def _run():
            async for line in async_gen:
                if line.startswith("data: "):
                    results.append(json.loads(line[6:].strip()))

        asyncio.run(_run())
        return results

    def test_replay_completed_task(self):
        task = RunningTask(task_type="test", session_id="s1")
        task.event_log.append({"step": 1, "message": "A"})
        task.event_log.append({"step": 2, "message": "B"})
        task.done = True

        events = self._collect(_subscribe_to_task(task))
        assert events[0] == {"step": 1, "message": "A"}
        assert events[1] == {"step": 2, "message": "B"}
        assert events[2] == {"done": True}

    def test_replay_failed_task(self):
        task = RunningTask(task_type="test", session_id="s1")
        task.event_log.append({"step": 1})
        task.done = True
        task.error = "something broke"

        events = self._collect(_subscribe_to_task(task))
        assert events[0] == {"step": 1}
        assert events[1] == {"error": "something broke"}

    def test_live_events_then_done(self):
        task = RunningTask(task_type="test", session_id="s1")
        _register_task(task)

        async def _run():
            results = []
            gen = _subscribe_to_task(task)

            # Push events from another thread after subscriber is attached
            def _push():
                time.sleep(0.05)
                from meetscribe.web.routes.tasks import _broadcast

                with task.lock:
                    task.event_log.append({"step": 1})
                _broadcast(task, ("item", {"step": 1}))

                time.sleep(0.05)
                with task.lock:
                    task.done = True
                    task.done_at = time.monotonic()
                _broadcast(task, ("done", None))

            t = threading.Thread(target=_push, daemon=True)
            t.start()

            async for line in gen:
                if line.startswith("data: "):
                    results.append(json.loads(line[6:].strip()))

            t.join(timeout=2)
            return results

        events = asyncio.run(_run())
        assert {"step": 1} in events
        assert {"done": True} in events

    def test_subscriber_cleaned_up_after_disconnect(self):
        task = RunningTask(task_type="test", session_id="s1")
        task.done = True
        task.event_log.append({"step": 1})

        self._collect(_subscribe_to_task(task))
        assert len(task.subscribers) == 0
