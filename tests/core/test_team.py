"""Tests for team.py — team context resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from meetscribe.database import close_db, create_team, get_db, init_db
from meetscribe.team import resolve_team


class TestResolveTeam:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        self.db_path = tmp_path / "test.db"
        self.teams_dir = tmp_path / "teams"
        self.teams_dir.mkdir()
        init_db(self.db_path)
        yield
        close_db()

    def _resolve(self, name=None):
        with (
            patch("meetscribe.team.config.DB_PATH", self.db_path),
            patch("meetscribe.team.config.TEAMS_DIR", self.teams_dir),
            patch(
                "meetscribe.team.config.get_team_samples_dir",
                lambda n: self.teams_dir / n / "samples",
            ),
            patch(
                "meetscribe.team.config.get_team_enrolled_dir",
                lambda n: self.teams_dir / n / "samples" / "enrolled",
            ),
            patch(
                "meetscribe.team.config.get_team_unknown_dir",
                lambda n: self.teams_dir / n / "samples" / "unknown",
            ),
            patch("meetscribe.team.config.ensure_team_dirs", lambda n: None),
        ):
            return resolve_team(name)

    def test_default_team_auto_created(self):
        ctx = self._resolve()
        assert ctx.name == "default"
        assert ctx.id > 0

    def test_existing_team_resolved(self):
        create_team(get_db(), "my-team")
        ctx = self._resolve("my-team")
        assert ctx.name == "my-team"

    def test_nonexistent_team_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self._resolve("nonexistent")

    def test_none_resolves_to_default(self):
        ctx = self._resolve(None)
        assert ctx.name == "default"

    def test_explicit_default_string(self):
        """Passing "default" explicitly should work the same as None."""
        ctx = self._resolve("default")
        assert ctx.name == "default"
        assert ctx.id > 0
