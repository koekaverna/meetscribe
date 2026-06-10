"""First-run regression test: importing the CLI must not require existing dirs."""

import os
import subprocess
import sys
from pathlib import Path


def test_cli_import_creates_missing_dirs(tmp_path: Path):
    """`import meetscribe.cli` sets up file logging at module level; on a fresh
    machine the data/logs directories don't exist yet and must be created
    before the FileHandler opens the log file."""
    data_dir = tmp_path / "fresh-data-dir"
    assert not data_dir.exists()

    env = os.environ.copy()
    env["MEETSCRIBE_DATA_DIR"] = str(data_dir)
    env["MEETSCRIBE_TMP_DIR"] = str(tmp_path / "fresh-tmp-dir")

    result = subprocess.run(
        [sys.executable, "-c", "import meetscribe.cli"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert (data_dir / "logs").is_dir()
