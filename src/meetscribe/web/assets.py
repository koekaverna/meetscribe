"""Static asset versioning: content-hash URLs for cache busting."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"

# rel_path -> (mtime, digest); mtime key re-hashes edited files without a restart
_hash_cache: dict[str, tuple[float, str]] = {}


def static_url(path: str) -> str:
    """Return a versioned URL for a file under static/: /static/{path}?v={md5[:8]}."""
    file = STATIC_DIR / path
    try:
        mtime = file.stat().st_mtime
    except OSError:
        logger.warning("static_url: file not found, serving unversioned: %s", file)
        return f"/static/{path}"
    cached = _hash_cache.get(path)
    if cached is None or cached[0] != mtime:
        digest = hashlib.md5(file.read_bytes(), usedforsecurity=False).hexdigest()[:8]
        _hash_cache[path] = (mtime, digest)
    return f"/static/{path}?v={_hash_cache[path][1]}"
