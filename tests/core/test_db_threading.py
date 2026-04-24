"""Tests for connection-per-thread DB access and shutdown cleanup."""

import sqlite3
import threading
from pathlib import Path

import pytest

from meetscribe.database import close_all_db, close_db, get_db, init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "thread_test.db"
    init_db(path)
    yield path
    close_db()


class TestConnectionPerThread:
    def test_same_thread_returns_same_connection(self, db_path):
        conn1 = get_db()
        conn2 = get_db()
        assert conn1 is conn2

    def test_different_threads_get_different_connections(self, db_path):
        main_conn = get_db()
        other_conn = None

        def worker():
            nonlocal other_conn
            other_conn = get_db()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert other_conn is not None
        assert other_conn is not main_conn

    def test_thread_connection_works(self, db_path):
        """Each thread's connection can read/write independently."""
        results = []

        def worker(value: int):
            conn = get_db()
            conn.execute(
                "INSERT INTO teams (name, description) VALUES (?, ?)",
                (f"thread-team-{value}", f"desc-{value}"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT name FROM teams WHERE name = ?", (f"thread-team-{value}",)
            ).fetchone()
            results.append(row["name"])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["thread-team-0", "thread-team-1", "thread-team-2"]

    def test_get_db_before_init_raises(self, tmp_path):
        """get_db() without init_db() raises RuntimeError."""
        import meetscribe.database as db_mod

        old_path = db_mod._db_path
        db_mod._db_path = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                # Need a fresh thread so there's no cached connection
                error = None

                def worker():
                    nonlocal error
                    try:
                        # Clear thread-local in case it's cached
                        if hasattr(db_mod._local, "conn"):
                            del db_mod._local.conn
                        get_db()
                    except RuntimeError as e:
                        error = e

                t = threading.Thread(target=worker)
                t.start()
                t.join()
                if error:
                    raise error
        finally:
            db_mod._db_path = old_path


class TestCloseAllDb:
    def test_closes_all_thread_connections(self, db_path):
        conns: list[sqlite3.Connection] = []

        def worker():
            conns.append(get_db())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Main thread connection too
        conns.append(get_db())
        assert len(conns) == 4

        close_all_db()

        # All connections should be closed — executing on them should fail
        for conn in conns:
            with pytest.raises(Exception):
                conn.execute("SELECT 1")

    def test_close_db_clears_thread_local(self, db_path):
        conn1 = get_db()
        close_db()
        conn2 = get_db()
        assert conn1 is not conn2
