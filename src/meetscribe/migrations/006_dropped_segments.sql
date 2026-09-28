-- Segments removed from the transcript (hallucination filter, joint duplicates)
-- and chunks the server failed to transcribe. Kept for review and restore.

CREATE TABLE IF NOT EXISTS session_dropped_segments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_num      INTEGER NOT NULL,
    start_ms       INTEGER NOT NULL,
    end_ms         INTEGER NOT NULL,
    speaker        TEXT,
    text           TEXT NOT NULL,
    reason         TEXT NOT NULL,
    no_speech_prob REAL,
    avg_logprob    REAL
);

CREATE INDEX IF NOT EXISTS idx_session_dropped_segments_session
    ON session_dropped_segments(session_id);
