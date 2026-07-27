# MeetScribe — Roadmap

> v0.5.13 → v1.1 | 8 phases | App (Web + Desktop) · CLI removed at Phase 4

## Current State

MeetScribe — self-hosted app (web + desktop) for meeting transcription with speaker diarization. **The CLI is being removed in favor of the app.** No further CLI work: its features move to the app (web + desktop) and, for automation, the REST API. Once the web app covers everything (Phase 3) and server-side first-run setup replaces the CLI bootstrap (Phase 4), the argparse CLI is deleted.

**What works:**
- Pipeline: server-side diarization (Speaches `/v1/audio/diarization`, VBx clustering) → cluster embeddings → local voiceprint matching → transcription
- Open-space mode: per-track filter that keeps only the target speaker's speech (drops other voices before STT)
- Whisper hallucination filtering; thread-safe DB access with task persistence
- CLI (legacy, slated for removal): transcribe, enroll, list-speakers, delete-speaker, extract, extract-samples, info, web, team/user admin — superseded by the app UI + first-run setup, deleted in Phase 4
- Web UI: FastAPI + Jinja2 + Alpine.js, 6-step workflow, auth, team scoping, SSE progress
- Page-scoped frontend: thin shell + per-page Alpine components (`workflowPage`, `sessionsPage`), x-if mounting, SSE teardown in `destroy()`
- Session archive (`/sessions`): paginated list (status, speakers, duration, preview, creator), sort, open-in-workflow resume, single + bulk delete with files
- Access model: non-admins see/delete only their own sessions (`creator_id`), admins the whole team; lazy session creation (no empty sessions from just opening the app)
- Admin panel (`/admin`): two-tier access — superadmin (all teams, team CRUD, Speaches status, disk usage, error log) vs team admin (own team's users only); user create/delete, password reset, grant/revoke admin; registration page removed
- Transcript playback (web): structured segments in DB, global player, multi-track sync + per-track mute, active segment/track highlighting, click-to-play
- DB: SQLite, 10 tables, numbered migrations, multi-team
- ~5400 lines, 28 modules, Python 3.12+
- Tests restructured (`core/`, `pipeline/`, `web/`), CI (ruff/mypy/pytest/bandit, 3.12+3.13 matrix), mutation testing (mutmut)

**Unique positioning:**
- Self-hosted, privacy-first — data never leaves your infrastructure (no cloud competitor offers this)
- One app, two surfaces — the same web UI in the browser and embedded in the Electron desktop client (separate repo), which adds dual-channel meeting recording
- Hybrid identification — enrolled voiceprints + auto-clustering unknown speakers
- Pluggable backend — Speaches API, swappable models

## Competitive Landscape

| Solution | Type | Key Features |
|----------|------|--------------|
| Otter.ai | Cloud | Real-time, 99+ languages, collaborative notes, series tracking |
| Fireflies.ai | Cloud | 6000+ integrations, sentiment analysis, conversation intelligence |
| Fathom | Cloud | Free tier, privacy-first, action items |
| MeetGeek | Cloud | Best summaries, action items with assignee/deadline, team analytics |
| tl;dv | Cloud | Free recordings, moment sharing, timestamp navigation |
| Grain | Cloud | Deal intelligence, collaborative annotation |
| WhisperX | OSS | Whisper + Pyannote, word-level timestamps, batch-only |
| pyannote | OSS | Diarization models, ~11-19% DER, no transcription |
| NeMo | OSS | NVIDIA, GPU-optimized diarization |

**Common commercial features missing in MeetScribe:**
- AI summaries and action items
- Transcript archive search
- Platform integrations (Zoom, Teams, Meet)
- Speaker analytics (talk time, engagement)
- Real-time transcription
- Multiple export formats
- Transcript editing

---

## Phase 1: Foundation & Hardening (v0.4) — ✅ done

> Solid foundation

**Goal:** Reliability for daily use. No new features — only confidence in existing ones.

### Tests

**Rules (still apply to all new tests):**
- Every assert checks a **specific value**, not `is not None`, not `isinstance`
- Mock only external dependencies (httpx, filesystem), not the object under test
- Test **behavior**, not implementation — no coupling to internal methods

**Done:**
- [x] Fixtures (`tests/conftest.py`): in-memory SQLite + migrations, Speaches API mock, 16kHz mono WAV, embedding vectors, minimal config
- [x] Unit tests: `merge_close_segments` / `merge_by_proximity` (`pipeline/models.py`), `SpeakerIdentifier` identify/identify_segments/`_find_nearest_labeled`, `enroll_samples`, `load_config` + `AppConfig.validate` (incl. **defaults match config.yaml**)
- [x] Functional tests: migrations idempotency, team/voiceprint/auth-session CRUD, **voiceprint stores `embeddings.model`**, session lifecycle + file cleanup, **web enrollment copies samples to disk**
- [x] Integration tests (mock HTTP): diarization end-to-end, embedding extraction + short-segment filtering, transcriber merge→slice→transcribe with offset timestamps
- [x] Test suite restructured into `tests/core/`, `tests/pipeline/`, `tests/web/`
- [x] Mutation testing infra (mutmut) configured over `pipeline/`, `database.py`, `config.py`, `team.py`

**Intentionally NOT tested:** trivial getters/duration calcs, numpy/werkzeug wrappers (`cosine_similarity`, `compute_voiceprint`, `hash_password`), low-level `_slice_wav` binary, CLI subcommands (manual), model presence on server (needs running server).

### Other hardening — done

- [x] Typed exceptions: `SpeachesAPIError`, `PipelineError`, `ConfigurationError` + HTTP retry for transient failures
- [x] DB migration versioning: `schema_version` table + numbered migrations (replaces `CREATE TABLE IF NOT EXISTS`)
- [x] Structured logging with stage timings (diarization / embedding / transcription)
- [x] CI: ruff check + format, mypy, pytest --cov, bandit, Python 3.12 + 3.13 matrix

### Remaining

- [ ] Docker: build image + verify migrations + CLI entry point (smoke test in CI)
- [ ] Round out mutation-test gaps: `merge_by_proximity` boundaries, DB PRAGMA/return-value checks, `load_config` error messages, `get_data_dir`/`get_tmp_dir` platform overrides, diarization param forwarding

---

## Phase 2: Transcript Storage & Playback (v0.5) — ✅ done

> Interactive transcript

**Goal:** Persist structured transcripts and tie audio playback to segments.

### Done

- [x] Table `session_segments`: session_id, track_num, start_ms, end_ms, speaker, text, sort_order
- [x] Structured segments stored in DB alongside markdown
- [x] Global audio player: play/pause, seekable progress bar, time display
- [x] Synchronized multi-track playback, per-track mute
- [x] Active segment highlighting during playback
- [x] Active track highlighting (which track the current segment belongs to)
- [x] Click segment to play from that point

### Remaining (transcript access)

- [x] Listing + viewing stored transcripts → delivered by the session archive (v0.5.5)
- [ ] Programmatic access → REST API in Phase 6

> Transcript **editing** also moved to Phase 3 (Web UI Maturity) — it belongs with the broader web push.

---

## Phase 3: Web UI Maturity (v0.6)

> From workflow to application — **in progress** (session list + frontend architecture shipped in v0.5.5)

**Goal:** Turn the linear 6-step workflow into a full application: a meeting archive, participant management, and in-place transcript editing for non-technical users.

### Meeting / session list — ✅ done (v0.5.5)

- [x] Session list page (`/sessions`): date, duration, speakers, status badge, transcript preview, creator, track count
- [x] Click → opens the session in the workflow (transcribed → step 6 transcript view with playback)
- [x] Pagination, sort by date/duration (rowid tie-breakers for stable OFFSET paging)
- [x] Per-user history: `creator_id` (migration 004) + access model — non-admins see/delete only their own sessions, admins the whole team (All team / Mine toggle); enforced in `get_session_for_user`, i.e. across the whole session API
- [x] Resume interrupted sessions: click opens the workflow at the step matching session status
- [x] Delete sessions with their files: per-row + bulk delete with checkbox selection (`POST /api/session/bulk-delete`)
- [x] Extras: lazy session creation (no empty sessions from just opening the app; also fixed a double-`init()` bug that created an orphan session per page load), dead `cleanup_old_sessions`/`SESSION_TTL` removed (TTL purge is incompatible with a permanent archive)

### Participant management (speakers dashboard) — ✅ done (v0.5.7)

- [x] Enrolled speakers list with sample playback (`/speakers`, admin-only like the admin panel: name, model, sample count, total duration, enrollment date)
- [x] Play / delete individual samples to curate voiceprint quality — deleting a sample recomputes the voiceprint from the remaining ones; the last sample can't be deleted (delete the speaker instead)
- [x] Delete / rename speakers — delete now also removes the enrolled samples directory; rename moves it and rejects collisions
- [x] Voiceprint quality indicator (sample count + total duration tiers). Embedding spread skipped: per-sample embeddings aren't persisted (only the averaged voiceprint), and recomputing them via the API on every dashboard load is too expensive

### Transcript editing

- [ ] Inline text editing per segment
- [ ] Speaker reassignment per segment
- [ ] Delete segment
- [ ] Merge adjacent segments
- [ ] Split segment
- [ ] Regenerate markdown after edits
- [ ] Speaker color coding in the viewer

### Frontend architecture (page-scoped lifecycle) — ✅ done (v0.5.5)

> Prerequisite for multiple pages (session list, admin) without full reloads.

- [x] Partial updates instead of full page reload (path-based routing `/` ↔ `/sessions`, history API, same template served on both paths)
- [x] Split the monolithic `app()` on `<body>` into a thin shell (`shell.js`: routing, page mounting) + per-page Alpine components (`workflowPage`, `sessionsPage`), gated by `x-if`; `adminPage` slots in later
- [x] SSE streams owned by `workflowPage`, torn down via Alpine `destroy()` on unmount (`x-if` flip / keyed `x-for` re-mount), no `unload`/`beforeunload`
- [x] New session = bump `workflowKey` → old `workflowPage` unmounts → `destroy()` closes its streams; navigating to another page does the same
- [x] Removed the interim `_closeTaskStreams()` poke
- **Gotcha for future pages:** shell state is deliberately named `activePage` — page components shadow same-named properties via the Alpine scope chain (sessionsPage's pagination `page` silently swallowed shell writes)

### Admin panel — ✅ done (v0.5.6)

- [x] User and team management (was CLI-only) — the last thing that *required* the CLI
  - Two-tier access: **superadmin** (instance owner — all teams, team CRUD) vs **team admin** (own team's users only); `is_superadmin` flag (migration 005, existing admins promoted), bootstrap via `meetscribe user create --superadmin`
  - Password reset (invalidates the user's sessions; resetting your own logs you out) and grant/revoke admin toggle
  - Guards: self-delete, last-admin, last-superadmin, team admins can't touch superadmins, team deletion blocked while it has users/sessions
  - Registration page removed — users are created in the panel (or via CLI)
- [x] Speaches server status (health ping with latency; non-2xx counts as down)
- [x] Disk usage (data dir total, sessions, team samples)
- [x] Recent error log (streamed tail of the newest log file)

### CLI retirement (prep)

> The app (web + desktop) becomes the only interface. No more CLI work — the CLI is **deleted in Phase 4** once the app + first-run setup cover everything it did. This phase just makes that deletion safe.

- [ ] Confirm the web app covers every CLI feature (transcribe, enroll, extract, samples, team/user admin) — the precondition for deletion
- [ ] Interim: print a "moved to the app" notice on the remaining CLI subcommands
- [ ] Everything new (search, stats, export, batch, webhooks, dictionary) lands in the app UI + REST API — never as CLI commands
- [ ] README/docs stop documenting CLI as a user interface, point to the app

### Files

- Shipped (v0.5.5): `migrations/004_session_creator.sql`, `web/static/js/{shell,workflow-page,sessions-page}.js` (replaces `app.js`), `web/templates/pages/sessions.html`; session list API lives in `web/routes/session.py` (no separate dashboard router needed)
- Shipped (v0.5.6): `migrations/005_superadmin.sql`, `web/routes/admin.py`, `web/static/js/admin-page.js`, `web/templates/pages/admin.html`; `register.html` + `/auth/register` removed
- Remaining: `web/routes/transcript.py`, `cli.py` (interim "moved to app" notices only)

---

## Phase 4: Desktop App — Electron client (v0.7)

> Thin client + dual-channel recording

**Goal:** Ship a desktop app that **records meeting audio directly** (mic + system audio as two tracks), embeds the existing web UI and uploads recordings through the regular API — removing the manual "record elsewhere → upload" step. The app is a thin **Electron client in a separate repo ([meetscribe-client](https://github.com/koekaverna/meetscribe-client))** talking to a local or remote server; this repo's stack stays Python-only.

> **Pivot from pywebview (2026-07):** the capture spike resolved the open question against pywebview. System-audio loopback is the make-or-break feature, and the only path that works on all three OSes without user-installed drivers is Chromium's `getDisplayMedia` loopback — which requires controlling the display-media handler, i.e. Electron. System webviews can't do it (WKWebView/WebKitGTK have no system-audio capture; Chromium hides PulseAudio monitor sources from `enumerateDevices`), and native Python capture has no macOS story: CATap — the API the ecosystem converged on (Chromium, OBS-adjacent tools) — has no Python bindings, and ScreenCaptureKit via PyObjC costs the Screen Recording permission plus Sequoia's monthly re-approval nag.

### App shell — ✅ done (client v0.1)

- [x] Electron window: 320px control panel + embedded web UI (`WebContentsView`, `persist:meetscribe` partition — login survives restarts); existing cookie auth reused as-is
- [x] Settings: server URL, mic device, system-audio toggle, keep-local-copies

### Audio recording — ✅ done (client v0.1)

- [x] Microphone capture with start/stop and a level meter (pause deferred)
- [x] System-audio loopback on all three OSes via one Chromium code path (`setDisplayMediaRequestHandler` + `audio: 'loopback'`): WASAPI (Windows), CoreAudio tap (macOS 14.2+, ScreenCaptureKit fallback for 13.0–14.1), PulseLoopbackManager (Linux, PulseAudio + PipeWire)
- [x] Reliability hardening: silence watchdog + hot reconnect (`MediaStreamAudioDestinationNode` swap survives Windows output-device switches), crash recovery via per-recording manifests (3 s chunks streamed to disk), device-loss auto-stop
- [x] Mic = track 1, system audio = track 2 → uploaded into `POST /api/session/{id}/tracks` (webm/opus ~40 MB/h; server extracts to WAV) — reuses named-track diarization and open-space filtering unchanged
- [x] Upload retry without duplicate tracks (server session id persisted in the manifest; only missing files re-sent)

### Remaining

- [ ] Real-device capture testing: Windows host, macOS 14.2+ (TCC prompt), Linux desktop (PipeWire) — WSL can't verify capture
- [ ] Pause/resume, tray icon (stretch)
- [ ] Distributable builds per OS (electron-builder config ready; macOS signing/notarization open)
- [ ] "Loopback produces non-silence" smoke check on every Electron bump (the backend already regressed silently once, in 39.0.0-beta.4)

### CLI removal

> Decoupled from the desktop app (a thin client doesn't manage the server). The server itself takes over the CLI's last jobs.

- [ ] Web first-run setup: when no users exist, the web app bootstraps the initial superadmin (replaces `team`/`user create`)
- [ ] Server entry point: `meetscribe web` shrinks to the only remaining command (or plain `uvicorn`/Docker CMD)
- [ ] Delete the argparse subcommands (`transcribe`, `enroll`, `extract`, `extract-samples`, `list-speakers`, `delete-speaker`, `team`, `user`, `info`) and their tests
- [ ] Gut `cli.py` to the launcher only; drop the `[project.scripts]` subcommand surface

### Files

- Client lives in a separate repo: `meetscribe-client` (Electron pinned exactly, plain JS, no build step)
- Modified here: `web/` (first-run bootstrap), `cli.py` (gutted to launcher), `pyproject.toml` (entry points)
- Removed: argparse subcommands + their tests under `tests/`

---

## Phase 5: Search & Analytics (v0.8)

> Organizational memory

**Goal:** Meeting archive as a searchable knowledge base, plus export and speaker metrics.

### Full-text search

- [ ] SQLite FTS5 on `session_segments`
- [ ] App: search bar with result highlighting
- [ ] Filters: date, speaker, team
- [ ] REST API: `GET /api/v1/search?q=…` — segments with context, date, speaker (automation; formalized in Phase 6)

### Speaker analytics

| Metric | Description |
|--------|-------------|
| Talk time | Total speaking time |
| Turn count | Number of turns |
| Avg turn duration | Average turn length |

- [ ] Computed during transcription, stored in `speaker_stats`
- [ ] App: per-session analytics panel, bar charts (CSS-only, no JS frameworks)
- [ ] REST API exposes the same stats as JSON

### Export

| Format | Description |
|--------|-------------|
| Markdown | Current format (default) |
| SRT | Subtitles with timecodes |
| VTT | WebVTT subtitles |
| JSON | Structured (segments with speaker, start, end, text) |
| TXT | Plain text (speaker: text, no timecodes) |

- [ ] App: export menu on the result page — pick format (srt/vtt/json/txt/md), download
- [ ] REST API: `GET /api/v1/sessions/{id}/export?format=…` — same formats for automation (replaces CLI export and `--json` piping)

### Batch processing

- [ ] App: multi-file / bulk upload with a processing queue (replaces `meetscribe transcribe *.mp4`)
- [ ] REST API: submit multiple files programmatically
- [ ] Parallel processing with configurable concurrency
- [ ] Progress summary on completion

### Files

- New: `pipeline/analytics.py`, `pipeline/export.py`, `web/routes/search.py`
- Modified: `database.py` (FTS5 + `speaker_stats`), `web/routes/dashboard.py`, `web/routes/api.py`

---

## Phase 6: Real-Time & Integrations (v0.9)

> Live meetings

**Goal:** Real-time transcription. Technically the most complex phase, but the strongest differentiator. Builds on the desktop app's capture (Phase 4).

### WebSocket audio streaming

- [ ] Endpoint: `ws://.../v1/stream` — receive PCM chunks
- [ ] Server-side VAD/diarization on the stream
- [ ] Buffering → embeddings → transcription in near-real-time
- [ ] Push transcript updates back via WebSocket

### Live recording → stream

- [ ] Desktop app / browser `MediaRecorder` → WebSocket (the client already pumps 3 s opus chunks through IPC — redirect the sink to a WebSocket)
- [ ] Live transcript display in the UI
- [ ] System audio capture (reuses Phase 4 loopback)

### REST API formalization

- [ ] OpenAPI documentation (FastAPI auto-generated)
- [ ] API key authentication (separate from cookie auth)
- [ ] Versioned API: `/api/v1/`

### Webhook notifications

- [ ] POST to Slack/Teams/any URL on events
- [ ] Events: `transcribe.complete`, `action_items.extracted`
- [ ] App: webhook management in the admin panel (add URL + select events)

### Meeting bot (stretch goal)

- [ ] Plugin interface: `MeetingBotPlugin` with `join()`, `record()`, `leave()`
- [ ] First candidate: SIP/VoIP

### Files

- New: `web/routes/stream.py`, `pipeline/realtime.py`, `integrations/`
- Modified: `web/app.py`, `pipeline/diarization.py`

---

## Phase 7: Enterprise (v1.0)

> Production Grade

**Goal:** Production-ready for organizational deployment.

### Security

- [ ] Rate limiting on auth endpoints
- [ ] RBAC: viewer / editor / admin
- [ ] Audit log: who, what, when
- [ ] Upload validation (magic bytes)

### Observability

- [ ] Prometheus `/metrics`: latency, pipeline duration, queue depth
- [ ] Structured JSON logs for log aggregation
- [ ] `StructuredFormatter` — switch to namespace key (`_ctx`) instead of denylist
- [ ] Health check with Speaches API connectivity verification

### Deployment

- [ ] Helm chart for Kubernetes
- [ ] Docker Compose profiles (`--profile gpu`)
- [ ] Automatic SQLite backup
- [ ] Multi-worker uvicorn with file locking

### Plugin system

- [ ] `meetscribe.plugins` entry point
- [ ] Refactor summaries / action items into plugins
- [ ] Interface: `TranscriptPlugin.process(segments) -> dict`

### Documentation

- [ ] User guide (installation, configuration, app walkthrough)
- [ ] API reference (auto-generated + guides)
- [ ] Deployment guide (Docker, bare metal, Kubernetes)

---

## Phase 8: Transcript Intelligence — LLM (v1.1)

> More than text — **done last, by request**

**Goal:** LLM post-processing. Highest user-visible value, but deliberately sequenced after the app, archive, and analytics are solid.

**Why:** Every commercial competitor has AI summaries. Organizations that can't use cloud tools need this locally.

### LLM integration

```yaml
# config.yaml
llm:
  url: "http://localhost:11434/v1"   # Ollama, vLLM, llama.cpp
  model: "llama3.1"
  timeout: 120
  max_tokens: 4096
```

- [ ] `src/meetscribe/pipeline/llm.py` — OpenAI-compatible client
- [ ] Graceful degradation: if LLM not configured — skip post-processing

### Meeting summary

- [ ] Executive summary
- [ ] Key discussions
- [ ] Decisions made
- [ ] Chunked processing for long transcripts (split by speaker turns)
- [ ] Output: markdown sections at the end of transcript

### Action items

- [ ] Task + assignee (from speaker name) + deadline (if mentioned)
- [ ] Format: `- [ ] @Speaker: task description`
- [ ] App: toggle summary / action-items on a session (post-process on demand)

### Custom dictionary (no LLM required)

- [ ] User-defined word list for correcting common ASR typos (names, jargon, abbreviations)
- [ ] Applied as post-processing after transcription
- [ ] App: dictionary editor in settings; also a YAML config section

### Files

- New: `pipeline/llm.py`, `pipeline/dictionary.py`
- Modified: `database.py`, `config.py`, `web/routes/tasks.py`, `web/routes/admin.py`

---

## Summary

| Phase | Version | Theme | Status | Key Outcome |
|-------|---------|-------|--------|-------------|
| 1 | v0.4 | Foundation & Hardening | ✅ done | Tests, CI, mutation testing, reliability |
| 2 | v0.5 | Storage & Playback | ✅ done | Segment storage, multi-track sync playback |
| 3 | v0.6 | Web UI Maturity | in progress | ✅ Session list + frontend architecture (v0.5.5), admin panel (v0.5.6), speakers dashboard (v0.5.7); next: transcript editing |
| 4 | v0.7 | Desktop (Electron client) | in progress | Thin client + dual-channel recording (separate repo, v0.1 done); **CLI removed** |
| 5 | v0.8 | Search & Analytics | planned | Full-text search, speaker stats, export |
| 6 | v0.9 | Real-time & Integrations | planned | WebSocket streaming, webhooks, API |
| 7 | v1.0 | Enterprise | planned | RBAC, metrics, plugins, Helm |
| 8 | v1.1 | Intelligence (LLM) | last | Summaries, action items, dictionary |

## Intentionally NOT doing

- **pywebview / embedded-server desktop** — rejected after the 2026-07 capture spike: system webviews can't record loopback audio, and native Python capture has no viable macOS path (CATap has no Python bindings; ScreenCaptureKit via PyObjC costs Screen Recording permission + Sequoia's monthly nag). Desktop is a thin Electron client in a separate repo; this repo stays Python-only with no Node runtime
- **Mobile app** — responsive web is sufficient
- **Video recording/playback** — record/transcribe audio only; no video capture
- **Training custom ASR models** — pluggable backend already supports model swapping
