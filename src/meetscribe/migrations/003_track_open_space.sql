-- Open-space mic filtering: per-track flag to keep only the assigned speaker

ALTER TABLE session_tracks ADD COLUMN open_space INTEGER NOT NULL DEFAULT 0;
