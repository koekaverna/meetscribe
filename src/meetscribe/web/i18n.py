"""Runtime i18n support: catalog loading, language resolution, translation.

Catalog files live at ``locales/<lang>/*.json`` as flat JSON ``{"ns.key": "text"}``.
All files for a language are merged into one dict, cached per language. The active
language for the current request is held in a ContextVar so ``t()`` and the Jinja
globals can read it without threading it through every call.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).parent / "locales"

SUPPORTED: tuple[str, ...] = ("en", "ru")
DEFAULT = "en"

# Lazily-populated per-language merged catalogs: {"en": {...}, "ru": {...}}.
_catalogs: dict[str, dict[str, str]] = {}

# Active language for the current request/task.
_current_lang: ContextVar[str] = ContextVar("current_lang", default=DEFAULT)


def _load_lang(lang: str) -> dict[str, str]:
    """Merge every ``locales/<lang>/*.json`` file into a single flat dict."""
    merged: dict[str, str] = {}
    lang_dir = LOCALES_DIR / lang
    for path in sorted(lang_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load locale file %s", path)
            continue
        if isinstance(data, dict):
            merged.update({str(k): str(v) for k, v in data.items()})
    return merged


def load_catalogs() -> None:
    """(Re)load all supported-language catalogs into the module cache."""
    _catalogs.clear()
    for lang in SUPPORTED:
        _catalogs[lang] = _load_lang(lang)


def _catalog_for(lang: str) -> dict[str, str]:
    """Return the merged catalog for ``lang``, loading lazily on first use."""
    if not _catalogs:
        load_catalogs()
    return _catalogs.get(lang, _catalogs.get(DEFAULT, {}))


def resolve_lang(request: object) -> str:
    """Resolve the request language: cookie, then Accept-Language, then default.

    ``request`` is a Starlette ``Request`` but typed loosely to avoid a hard
    import here (this module is imported early by the app).
    """
    cookies = getattr(request, "cookies", {}) or {}
    cookie_lang = cookies.get("lang")
    if cookie_lang in SUPPORTED:
        return str(cookie_lang)

    headers = getattr(request, "headers", {}) or {}
    accept = headers.get("accept-language", "") if headers else ""
    if accept:
        # Rough parse: honour the first tag whose primary subtag we support.
        for part in accept.split(","):
            tag = part.split(";", 1)[0].strip().lower()
            primary = tag.split("-", 1)[0]
            if primary in SUPPORTED:
                return primary

    return DEFAULT


def set_current_lang(lang: str) -> None:
    """Set the active language for the current request (falls back to default)."""
    _current_lang.set(lang if lang in SUPPORTED else DEFAULT)


def current_lang() -> str:
    """Return the active language for the current request."""
    return _current_lang.get()


def i18n_catalog() -> dict[str, str]:
    """Return the full merged catalog for the active language (for JS injection)."""
    return _catalog_for(current_lang())


def t(key: str, **kwargs: object) -> str:
    """Translate ``key`` for the active language.

    Falls back to the English catalog, then to the raw key. When ``kwargs`` are
    given, ``str.format`` is applied defensively — a malformed template returns
    the raw string rather than raising.
    """
    lang = current_lang()
    text = _catalog_for(lang).get(key)
    if text is None and lang != DEFAULT:
        text = _catalog_for(DEFAULT).get(key)
    if text is None:
        text = key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text
