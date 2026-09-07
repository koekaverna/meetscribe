"""End-to-end tests for the i18n layer: locale resolution, catalog rendering, toggle."""

import pytest
from fastapi.testclient import TestClient

from meetscribe.web import i18n


@pytest.fixture(autouse=True)
def _reset_lang():
    """Keep the module-level contextvar from leaking the active lang across tests."""
    yield
    i18n.set_current_lang(i18n.DEFAULT)


class _FakeRequest:
    """Minimal stand-in for resolve_lang() — only cookies/headers are read."""

    def __init__(self, accept_language: str) -> None:
        self.cookies: dict[str, str] = {}
        self.headers = {"accept-language": accept_language}


def _csrf_token(client: TestClient) -> str:
    """Get the CSRF token from the cookie after a GET request."""
    client.get("/login")
    token = client.cookies.get("meetscribe_csrf")
    assert token, "CSRF cookie not set after GET /login"
    return token


class TestCatalogRuntime:
    def test_ru_catalog_differs_from_en(self) -> None:
        i18n.load_catalogs()
        i18n.set_current_lang("en")
        assert i18n.t("nav.sessions") == "Sessions"
        i18n.set_current_lang("ru")
        assert i18n.t("nav.sessions") == "Сессии"

    def test_missing_key_returns_key(self) -> None:
        i18n.set_current_lang("ru")
        assert i18n.t("does.not.exist") == "does.not.exist"

    def test_interpolation(self) -> None:
        i18n.load_catalogs()
        i18n.set_current_lang("en")
        # step1.track_label = "Track {n}"
        assert i18n.t("step1.track_label", n=3) == "Track 3"

    def test_full_key_parity_en_ru(self) -> None:
        i18n.load_catalogs()
        en = i18n._catalog_for("en")
        ru = i18n._catalog_for("ru")
        assert set(en) == set(ru)
        assert len(en) > 100


class TestLoginPageLocale:
    def test_login_default_english(self, client: TestClient) -> None:
        html = client.get("/login").text
        assert '<html lang="en">' in html
        assert "Войти" not in html

    def test_login_russian_via_cookie(self, client: TestClient) -> None:
        client.cookies.set("lang", "ru")
        html = client.get("/login").text
        assert '<html lang="ru">' in html
        assert "Войти" in html

    def test_catalog_injected_into_page(self, client: TestClient) -> None:
        html = client.get("/login").text
        assert "window.__I18N__" in html
        assert "window.__LANG__" in html


class TestShellLocale:
    def test_index_russian_subtitle(self, auth_client: TestClient) -> None:
        auth_client.cookies.set("lang", "ru")
        html = auth_client.get("/").text
        assert '<html lang="ru">' in html
        assert "Транскрибация встреч с диаризацией спикеров" in html

    def test_index_english_default(self, auth_client: TestClient) -> None:
        auth_client.cookies.set("lang", "en")
        html = auth_client.get("/").text
        assert "Meeting transcription with speaker diarization" in html

    def test_step6_editing_controls_russian(self, auth_client: TestClient) -> None:
        auth_client.cookies.set("lang", "ru")
        html = auth_client.get("/step/6").text
        assert "Вставить в начало" in html
        assert "Текст пропущенной фразы" in html
        assert "Insert at start" not in html


class TestLangToggle:
    def test_sets_cookie(self, client: TestClient) -> None:
        token = _csrf_token(client)
        resp = client.post("/api/lang", data={"lang": "ru", "csrf_token": token})
        assert resp.status_code == 204
        assert client.cookies.get("lang") == "ru"

    def test_rejects_unknown_lang(self, client: TestClient) -> None:
        token = _csrf_token(client)
        resp = client.post("/api/lang", data={"lang": "xx", "csrf_token": token})
        assert resp.status_code == 400

    def test_requires_csrf(self, client: TestClient) -> None:
        client.get("/login")  # seed csrf cookie
        resp = client.post("/api/lang", data={"lang": "ru", "csrf_token": "bad"})
        assert resp.status_code == 403


class TestAcceptLanguage:
    def test_highest_q_wins(self) -> None:
        assert i18n.resolve_lang(_FakeRequest("en;q=0.1, ru;q=1")) == "ru"

    def test_zero_q_ignored(self) -> None:
        assert i18n.resolve_lang(_FakeRequest("ru;q=0, en")) == "en"

    def test_unsupported_falls_back_to_default(self) -> None:
        assert i18n.resolve_lang(_FakeRequest("fr-FR, de;q=0.8")) == "en"
