# Test Coverage Analysis

**Date:** 2026-03-25
**Overall Coverage:** 39% (1003/2562 statements covered)
**Tests:** 204 passing across 14 test files

## Current Coverage by Module

| Module | Coverage | Notes |
|--------|----------|-------|
| `pipeline/` (core) | 93-100% | Well tested |
| `config.py` | 91% | Well tested |
| `database.py` | 91% | Well tested |
| `errors.py` | 95% | Well tested |
| `team.py` | 85% | Minor gaps |
| `web/models.py` | 100% | Well tested |
| `web/services/session.py` | 78% | Partial coverage |
| **`cli.py`** | **0%** | **574 statements untested** |
| **`log.py`** | **0%** | **19 statements untested** |
| **`web/app.py`** | **0%** | **109 statements untested** |
| **`web/deps.py`** | **0%** | **28 statements untested** |
| **`web/routes/*`** | **0%** | **282 statements untested** |
| **`web/services/auth.py`** | **0%** | **90 statements untested** |
| **`web/services/pipeline.py`** | **0%** | **165 statements untested** |

## Priority Recommendations

### 1. Web Auth Service (`web/services/auth.py`) — High Priority

**Why:** Security-critical code at 0% coverage. Handles password hashing, session tokens, login, and registration.

**What to test:**
- `hash_password` / `verify_password` round-trip correctness
- `verify_password` with malformed hashes (edge cases, tampered data)
- `AuthService.register` — happy path, duplicate username, nonexistent team
- `AuthService.login` — valid credentials, invalid password, nonexistent user
- `AuthService.verify_session` — valid token, expired token, invalid token
- `AuthService.logout` — session deletion
- Singleton initialization (`init_auth_service` / `get_auth_service`)

**Estimated effort:** Medium. Uses the existing `db` fixture; no HTTP mocking needed.

### 2. Web Routes (`web/routes/`) — High Priority

**Why:** 282 statements at 0%. These are the API endpoints that handle user input, file uploads, CSRF validation, and session lifecycle.

**What to test (using FastAPI `TestClient`):**
- **`auth.py`**: Login/register/logout flows, CSRF enforcement, error responses
- **`tracks.py`**: File upload validation, audio extraction triggering
- **`samples.py`**: Speaker sample CRUD operations
- **`session.py`**: Session creation and retrieval, team isolation
- **`speakers.py`**: Speaker management endpoints
- **`tasks.py`**: Async task submission, status polling, cancellation

**Estimated effort:** Medium-high. Requires a `TestClient` fixture with authenticated sessions. Recommend creating a `web_conftest.py` or extending `conftest.py` with a factory that sets up the FastAPI app with an in-memory database.

### 3. CLI (`cli.py`) — Medium Priority

**Why:** 574 statements at 0% — the single largest untested file and the primary user interface for the non-web workflow.

**What to test:**
- Argument parsing and validation for each subcommand
- Config loading and error handling
- Pipeline orchestration (mock the pipeline stages)
- Output formatting (transcript generation)
- Error handling paths (missing files, invalid configs, API failures)

**Estimated effort:** Medium-high. Use `click.testing.CliRunner` (or equivalent for the CLI framework used). Mock external calls (HTTP, FFmpeg) to keep tests fast and deterministic.

### 4. FastAPI App & Middleware (`web/app.py`, `web/deps.py`) — Medium Priority

**Why:** 137 statements at 0%. Contains CSRF middleware, auth middleware, and dependency injection — all security-relevant.

**What to test:**
- CSRF cookie middleware sets cookie on first request
- CSRF token validation in `verify_csrf` (matching, missing, mismatched)
- Auth middleware redirects unauthenticated users on page routes
- Auth middleware passes through public paths (`/auth`, `/static`, `/health`)
- Auth middleware lets API routes through (delegates to dependency)
- `get_current_user` raises 401 when no cookie / expired session
- `get_session_for_user` enforces team isolation (returns 404 for wrong team)
- Health endpoint returns `{"status": "ok"}`

**Estimated effort:** Medium. TestClient-based.

### 5. Pipeline Service (`web/services/pipeline.py`) — Medium Priority

**Why:** 165 statements at 0%. Orchestrates the entire processing pipeline for the web UI — coordinates VAD, embeddings, clustering, and transcription.

**What to test:**
- Pipeline step orchestration (mock individual pipeline stages)
- Error propagation from pipeline stages
- Session state updates during processing
- Concurrent task handling

**Estimated effort:** Medium. Requires mocking HTTP calls to Speaches API servers.

### 6. Logging (`log.py`) — Low Priority

**Why:** 19 statements at 0%. Small but testable.

**What to test:**
- `apply_log_level` sets correct level on root logger
- `apply_log_level` with invalid level string falls back to INFO
- `StructuredFormatter.format` with no extras returns base message
- `StructuredFormatter.format` appends key=value pairs for extras

**Estimated effort:** Low. Pure unit tests, no fixtures needed.

### 7. Minor Gaps in Existing Tests

- **`team.py` (85%)**: Lines 35-38 uncovered — likely an edge case in team operations
- **`web/services/session.py` (78%)**: Several uncovered branches around session state transitions (lines 122-127, 228-230, 296-298, 338-341, 405-425, 435-443)
- **`errors.py` (95%)**: Line 67 — the `is_transient` path for `status_code == 429`
- **`config.py` (91%)**: Lines around environment variable loading and defaults

## Suggested Implementation Order

1. **Auth service tests** — highest security impact, lowest effort
2. **`web/deps.py` tests** — small file, security-critical (CSRF + auth dependencies)
3. **`log.py` tests** — quick win, easy to implement
4. **Web route tests** — build `TestClient` fixture, then cover routes incrementally
5. **App middleware tests** — reuses `TestClient` fixture from step 4
6. **Pipeline service tests** — requires more mocking infrastructure
7. **CLI tests** — largest effort, but important for the non-web workflow

## Infrastructure Recommendations

- **Add a `tests/web/` directory** with its own `conftest.py` providing a `TestClient` fixture, an authenticated client, and helpers for creating test sessions
- **Add `httpx` mocking** (e.g., `respx` or `pytest-httpx`) to test Speaches API interactions in pipeline service and CLI tests
- **Set a coverage threshold** in CI (e.g., `--cov-fail-under=60`) and increase it incrementally as tests are added
- **Expand mutation testing** (`mutmut` is already configured) to cover `web/services/auth.py` after adding tests — mutation testing on security code catches subtle correctness bugs that line coverage misses
