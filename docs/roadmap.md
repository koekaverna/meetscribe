# MeetScribe — Roadmap

> v0.5.4 → v1.1 | 8 phases | App (Web + Desktop) · CLI removed at Phase 4

## Current State

MeetScribe — self-hosted app (web + desktop) for meeting transcription with speaker diarization. **The CLI is being removed in favor of the app.** No further CLI work: its features move to the app (web + desktop) and, for automation, the REST API. Once the web app covers everything (Phase 3) and the desktop app's first-run setup replaces launch + bootstrap (Phase 4), the argparse CLI is deleted.

**What works:**
- Pipeline: server-side diarization (Speaches `/v1/audio/diarization`, VBx clustering) → cluster embeddings → local voiceprint matching → transcription
- Open-space mode: per-track filter that keeps only the target speaker's speech (drops other voices before STT)
- Whisper hallucination filtering; thread-safe DB access with task persistence
- CLI (legacy, slated for removal): transcribe, enroll, list-speakers, delete-speaker, extract, extract-samples, info, web, team/user admin — superseded by the app UI + first-run setup, deleted in Phase 4
- Web UI: FastAPI + Jinja2 + Alpine.js, 6-step workflow, auth, team scoping, SSE progress
- Transcript playback (web): structured segments in DB, global player, multi-track sync + per-track mute, active segment/track highlighting, click-to-play
- DB: SQLite, 10 tables, numbered migrations, multi-team
- ~5400 lines, 28 modules, Python 3.12+
- Tests restructured (`core/`, `pipeline/`, `web/`), CI (ruff/mypy/pytest/bandit, 3.12+3.13 matrix), mutation testing (mutmut)

**Unique positioning:**
- Self-hosted, privacy-first — data never leaves your infrastructure (no cloud competitor offers this)
- One app, two surfaces — same FastAPI app served in the browser and wrapped natively via pywebview
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

- [ ] Listing + viewing stored transcripts → delivered by the app's session list in Phase 3 (**not** CLI `list`/`show` — the CLI is being removed)
- [ ] Programmatic access → REST API in Phase 6

> Transcript **editing** also moved to Phase 3 (Web UI Maturity) — it belongs with the broader web push.

---

## Phase 3: Web UI Maturity (v0.6)

> From workflow to application — **next priority**

**Goal:** Turn the linear 6-step workflow into a full application: a meeting archive, participant management, and in-place transcript editing for non-technical users.

### Meeting / session list

- [ ] Session list page: date, duration, speakers, status badge, summary preview
- [ ] Click → full transcript with speaker timeline
- [ ] Pagination, sort by date/duration
- [ ] Per-user history
- [ ] Resume interrupted sessions
- [ ] Delete old sessions with their files

### Participant management (speakers dashboard)

- [ ] Enrolled speakers list with sample playback
- [ ] Play / delete individual samples to curate voiceprint quality
- [ ] Delete / rename speakers
- [ ] Voiceprint quality indicator (sample count, embedding spread)

### Transcript editing

- [ ] Inline text editing per segment
- [ ] Speaker reassignment per segment
- [ ] Delete segment
- [ ] Merge adjacent segments
- [ ] Split segment
- [ ] Regenerate markdown after edits
- [ ] Speaker color coding in the viewer

### Frontend architecture (page-scoped lifecycle)

> Prerequisite for multiple pages (session list, admin) without full reloads.

- [ ] Partial updates instead of full page reload
- [ ] Split the monolithic `app()` on `<body>` into a thin shell (auth, routing, current page) + per-page Alpine components (`workflowPage`, `sessionsPage`, `adminPage`), gated by `x-if`
- [ ] SSE streams owned by the page that uses them, not the long-lived root — torn down via Alpine `destroy()` on unmount (`x-if` flip / `:key="session.id"` re-mount), **not** `unload`/`beforeunload` (breaks the back/forward cache; the browser already closes EventSource on real navigation)
- [ ] New session = change `:key` → old `workflowPage` unmounts → `destroy()` closes its streams; navigating to another page does the same
- [ ] Remove the interim `_closeTaskStreams()` poke in `startNewSession` once streams move into `workflowPage`
- **Why:** streams currently live on the root component, which never unmounts on session switch — so there is no lifecycle event to detach them, and a stale stream can leak into the next session's UI state

### Admin panel

- [ ] User and team management (currently CLI-only) — the last thing that *requires* the CLI; moving it here is what lets the CLI be removed
- [ ] Speaches server status
- [ ] Disk usage
- [ ] Recent error log

### CLI retirement (prep)

> The app (web + desktop) becomes the only interface. No more CLI work — the CLI is **deleted in Phase 4** once the app + first-run setup cover everything it did. This phase just makes that deletion safe.

- [ ] Confirm the web app covers every CLI feature (transcribe, enroll, extract, samples, team/user admin) — the precondition for deletion
- [ ] Interim: print a "moved to the app" notice on the remaining CLI subcommands
- [ ] Everything new (search, stats, export, batch, webhooks, dictionary) lands in the app UI + REST API — never as CLI commands
- [ ] README/docs stop documenting CLI as a user interface, point to the app

### Files

- New: `web/routes/dashboard.py`, `web/routes/admin.py`, `web/routes/transcript.py`, `migrations/004_*.sql`
- Modified: all `web/templates/`, `web/static/js/app.js`, `web/services/session.py`, `database.py`, `cli.py` (interim "moved to app" notices only)

---

## Phase 4: Desktop App — pywebview (v0.7)

> Native window + in-app recording

**Goal:** Ship a desktop app that wraps the existing web UI in a native window and can **record meeting audio directly**, removing the manual "record elsewhere → upload" step. pywebview (system webview + Python backend) keeps the stack Python-first — no Electron/Node runtime.

### App shell

- [ ] pywebview window hosting the embedded FastAPI app (local server lifecycle managed by the app)
- [ ] Packaging into a single distributable (PyInstaller) per OS
- [ ] First-run setup: Speaches endpoint, data dir, credentials
- [ ] System tray + window state (stretch)

### Audio recording

- [ ] Microphone capture with start/stop/pause and a level meter
- [ ] System-audio / loopback capture (meeting audio from other participants) — platform-specific (WASAPI loopback on Windows, monitor source on Linux/PulseAudio, ScreenCaptureKit/aggregate device on macOS)
- [ ] Record → save WAV → feed into the existing extract/diarize/transcribe pipeline
- [ ] Multi-source capture mapped to tracks (mic = track 1, system audio = track 2) to reuse named-track diarization
- **Open question:** capture inside the webview via `getUserMedia`/`MediaRecorder` vs. native Python capture (`sounddevice`/`soundcard`). Native gives reliable loopback + device selection; decide during spike.

### CLI removal

> The native launch + first-run setup take over the CLI's last jobs (starting the server, bootstrapping the first admin). The argparse CLI is now deleted — this is the "app transition".

- [ ] First-run setup creates the initial admin (replaces `team`/`user create` bootstrap)
- [ ] Single launch entry point: desktop binary for users, `uvicorn`/Docker CMD for server deploys (replaces `meetscribe web`/`app`)
- [ ] Delete the argparse subcommands (`transcribe`, `enroll`, `extract`, `extract-samples`, `list-speakers`, `delete-speaker`, `team`, `user`, `info`) and their tests
- [ ] Gut `cli.py` to the launcher only; drop the `[project.scripts]` subcommand surface

### Files

- New: `desktop/` (pywebview entry, packaging spec), `pipeline/record.py`
- Modified: `web/app.py` (embeddable server), `cli.py` (gutted to launcher), `pyproject.toml` (entry points)
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

- [ ] Desktop app / browser `MediaRecorder` → WebSocket
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
| 3 | v0.6 | Web UI Maturity | next | Session list, participant mgmt, transcript editing, admin |
| 4 | v0.7 | Desktop (pywebview) | planned | Native app + in-app recording; **CLI removed** |
| 5 | v0.8 | Search & Analytics | planned | Full-text search, speaker stats, export |
| 6 | v0.9 | Real-time & Integrations | planned | WebSocket streaming, webhooks, API |
| 7 | v1.0 | Enterprise | planned | RBAC, metrics, plugins, Helm |
| 8 | v1.1 | Intelligence (LLM) | last | Summaries, action items, dictionary |

## Intentionally NOT doing

- **Electron desktop app** — desktop is delivered via pywebview (Phase 4): system webview + Python backend, no Node runtime
- **Mobile app** — responsive web is sufficient
- **Video recording/playback** — record/transcribe audio only; no video capture
- **Training custom ASR models** — pluggable backend already supports model swapping
