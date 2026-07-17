-- Two-tier admin model: superadmins manage the instance (teams, all users,
-- status/disk/errors), regular admins manage only their own team's users.
ALTER TABLE users ADD COLUMN is_superadmin INTEGER NOT NULL DEFAULT 0;

-- Existing admins predate the two-tier model and were created by the instance
-- owner via CLI; promote them so upgrades don't lock anyone out of the panel.
UPDATE users SET is_superadmin = 1 WHERE is_admin = 1;
