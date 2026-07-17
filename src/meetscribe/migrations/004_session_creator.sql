-- Session archive: record who created each session (NULL for pre-existing rows)

ALTER TABLE sessions ADD COLUMN creator_id INTEGER REFERENCES users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_team_created ON sessions(team_id, created_at);
