-- Initial schema: teams, voiceprints, users, auth, sessions.

CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

CREATE TABLE IF NOT EXISTS voiceprints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id     INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(team_id, name)
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    team_id       INTEGER NOT NULL REFERENCES teams(id),
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    team_id    INTEGER NOT NULL REFERENCES teams(id),
    status     TEXT NOT NULL DEFAULT 'created',
    language   TEXT NOT NULL DEFAULT 'ru',
    transcript TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_tracks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_num    INTEGER NOT NULL,
    filename     TEXT NOT NULL,
    speaker_name TEXT,
    diarize      INTEGER NOT NULL DEFAULT 1,
    UNIQUE(session_id, track_num)
);

CREATE TABLE IF NOT EXISTS session_speakers (
    id         TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    PRIMARY KEY (session_id, id)
);

CREATE TABLE IF NOT EXISTS session_samples (
    id                 TEXT NOT NULL,
    session_id         TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_num          INTEGER NOT NULL,
    cluster_id         INTEGER NOT NULL,
    filename           TEXT NOT NULL,
    duration_ms        INTEGER NOT NULL,
    speaker_id         TEXT,
    is_known           INTEGER NOT NULL DEFAULT 0,
    known_speaker_name TEXT,
    PRIMARY KEY (session_id, id)
);
