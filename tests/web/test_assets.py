"""Tests for static asset versioning (static_url) and cache headers."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meetscribe.web import assets
from meetscribe.web.assets import static_url


@pytest.fixture
def static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(assets, "STATIC_DIR", tmp_path)
    monkeypatch.setattr(assets, "_hash_cache", {})
    return tmp_path


class TestStaticUrl:
    def test_versioned_url_format(self, static_dir: Path) -> None:
        (static_dir / "app.js").write_text("console.log(1)")
        url = static_url("app.js")
        prefix, _, version = url.partition("?v=")
        assert prefix == "/static/app.js"
        assert len(version) == 8
        int(version, 16)  # hex digest

    def test_hash_is_stable(self, static_dir: Path) -> None:
        (static_dir / "app.js").write_text("console.log(1)")
        assert static_url("app.js") == static_url("app.js")

    def test_hash_changes_on_content_change(self, static_dir: Path) -> None:
        f = static_dir / "app.js"
        f.write_text("one")
        url1 = static_url("app.js")
        f.write_text("two")
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 1))
        url2 = static_url("app.js")
        assert url1 != url2

    def test_missing_file_returns_unversioned(self, static_dir: Path) -> None:
        assert static_url("missing.css") == "/static/missing.css"


class TestCacheHeaders:
    def test_versioned_static_is_immutable(self, client: TestClient) -> None:
        resp = client.get("/static/js/shell.js?v=abc123")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"

    def test_unversioned_static_has_no_long_cache(self, client: TestClient) -> None:
        resp = client.get("/static/js/shell.js")
        assert resp.status_code == 200
        assert "immutable" not in resp.headers.get("cache-control", "")

    def test_html_page_is_no_cache(self, client: TestClient) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-cache"

    def test_api_response_untouched(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "cache-control" not in resp.headers
