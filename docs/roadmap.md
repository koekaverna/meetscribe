# MeetScribe — Roadmap

> v0.3.2 → v1.0 | 7 phases | CLI + Web

## Current State

MeetScribe — CLI + Web tool for meeting transcription with speaker diarization.

**What works:**
- Pipeline: VAD → Embeddings → Clustering → Identification → Transcription (remote Speaches API)
- CLI: transcribe, enroll, extract, extract-samples, list-speakers, info, web, team, user
- Web UI: FastAPI + Jinja2, 6-step workflow, auth, team scoping, SSE progress
- DB: SQLite, 8 tables, migrations, multi-team
- ~4900 lines, 29 modules

**Unique positioning:**
- CLI-first — no competitors in this niche
- Self-hosted / privacy-first — data never leaves your infrastructure
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

## Phase 1: Hardening (v0.4)

> Solid foundation

**Goal:** Reliability for daily use. No new features — only confidence in existing ones.

### Tests

#### Rules

- Every assert checks a **specific value**, not `is not None`, not `isinstance`
- Mock only external dependencies (httpx, filesystem), not the object under test
- Test **behavior**, not implementation — no coupling to internal methods

#### Fixtures (`tests/conftest.py`)

- [x] In-memory SQLite with migrations
- [x] Speaches API mock (httpx responses for VAD, embeddings, transcriptions)
- [x] 16kHz mono WAV file (short, in-memory)
- [x] Embedding vectors (normalized float lists)
- [x] Minimal config.yaml (temp file)

#### Unit tests (pure functions)

**`tests/test_models.py`** — `pipeline/models.py`:
- [x] `merge_close_segments()` — empty, single, merge same speaker, skip different speakers, max_chunk_ms limit, gap > max_gap_ms
- [x] `merge_by_proximity()` — same but ignoring speaker, speaker preserved from first

**`tests/test_clustering.py`** — `pipeline/clustering.py`:
- [x] `cluster_embeddings()` — 2 voices → 2 clusters, 1 voice → 1 cluster, thresholds affect cluster count

**`tests/test_identification.py`** — `pipeline/embeddings.py` (local logic):
- [x] `SpeakerIdentifier.identify()` — direct match (sim >= threshold), confident gap accept, below min_threshold reject, no voiceprints → None
- [x] `SpeakerIdentifier._find_nearest_labeled()` — left/right neighbor, both sides, no neighbors
- [x] `SpeakerIdentifier.identify_segments()` — all known, all unknown → clusters, mixed, short segments inherit from neighbor
- [x] `enroll_samples()` — copies files to enrolled_dir, includes previously enrolled samples, returns correct count

**`tests/test_config.py`** — `config.py`:
- [x] `load_config()` — file not found, empty file, valid YAML, defaults when sections missing
- [x] `AppConfig.validate()` — empty servers list, nonexistent server name
- [x] **Defaults match config.yaml** — parse both, compare values (git history: 3 bugs from drift)

#### Functional tests (DB + filesystem)

**`tests/test_database.py`** — `database.py` with in-memory SQLite:
- [x] Migrations: idempotency (repeated `get_db()`)
- [x] `_validate_team_name()` — valid/invalid names
- [x] Team: create + delete, "default" protected from deletion, cascade to voiceprints
- [x] Voiceprint: save (insert + upsert), load by team, delete
- [x] **Voiceprint model metadata** — save stores `embeddings.model`, not `transcription.model` (git: commit 4242478)
- [x] Auth sessions: create, get with expiry check, delete_expired
- [x] `ensure_default_team()` — idempotency

**`tests/test_team.py`** — `team.py`:
- [x] `resolve_team()` — auto-create "default", resolve existing team, error on nonexistent

**`tests/test_session.py`** — `web/services/session.py` (DB + temp dirs):
- [x] Lifecycle: create → add track → configure → delete (file cleanup)
- [x] Track: renumbering on middle track deletion
- [x] Sample: add → move between speakers → delete (file cleanup)
- [x] Rollback: error on add_track → file not left behind
- [x] **Enrollment copies samples to disk** (git: commit 4242478 — web didn't copy)

#### Integration tests (mock HTTP)

**`tests/test_pipeline_integration.py`** — full flow with Speaches mocks:
- [x] VAD: mock → specific segments with expected start_ms/end_ms; error mock → raise
- [x] Embeddings: `extract_segments()` — parallel extraction, short segment filtering (< min_duration_ms)
- [x] Transcriber: `transcribe_segments()` — merge → slice → transcribe → verify offset timestamps and speaker assignment
- [x] Diarization: VAD → Embeddings → Clustering → Identification — end-to-end, verify final speaker labels
- [x] `_find_speaker()` — overlap calculation, no overlap → "Unknown"

#### Intentionally NOT tested

- `SpeechSegment.duration_ms` — trivial `end_ms - start_ms`
- `cosine_similarity()`, `compute_voiceprint()` — would test numpy
- `hash_password()` / `verify_password()` — would test werkzeug
- `_slice_wav()` — low-level binary, validation = Speaches accepted the audio
- Trivial DB getters (`get_team`, `get_user_by_id`, `list_users`) — single SELECT
- Auth service flow (register/login/logout) — thin wrapper
- CLI subcommands — heavy for unit tests, covered by manual testing
- `_format_elapsed()` — CLI formatting, breakage is visually obvious
- Model presence on server — impossible without a running server

#### Phase 1.1: Mutation testing

**Unit tests:**

- [ ] `merge_by_proximity`: split preserves field values, duration calc, max_chunk boundary, single segment
- [ ] `database`: PRAGMA WAL/FK verification, parent dir creation, schema_version fallback, delete return values
- [ ] `load_config`: error message content, partial YAML defaults match dataclass, log_level default
- [ ] `get_data_dir` / `get_tmp_dir`: env var overrides, platform-specific defaults (mock sys.platform)
- [ ] `resolve_team`: default name, directory creation, error message, correct team id

**Integration tests (mock HTTP):**

- [ ] `diarization.diarize`: VAD/embedding params forwarded, empty VAD → empty result, short segments → "Unknown"
- [ ] `transcriber`: model/language in request, merge before STT, round-robin across servers
- [ ] `audio` (`@pytest.mark.integration`): probe track count, extract valid WAV, segment duration, FFmpegNotFoundError

### CI (GitHub Actions)

- [x] ruff check + ruff format --check
- [x] mypy
- [x] pytest --cov
- [x] bandit
- [x] Matrix: Python 3.12 + 3.13
- [ ] Docker: build image + verify migrations + CLI entry point (smoke test)

### Typed exceptions

- [x] `SpeachesAPIError` — remote API errors
- [x] `PipelineError` — processing errors
- [x] `ConfigurationError` — invalid configuration
- [x] HTTP retry for transient failures (httpx retries)

### DB migration versioning

- [x] `schema_version` table + numbered migrations
- [x] Replace `CREATE TABLE IF NOT EXISTS` (doesn't support ALTER)

### Structured logging

- [x] Timings: VAD took X ms, embedding extraction took Y ms, transcription took Z ms

### Files

- New: `tests/`, `.github/workflows/ci.yml`
- Modified: `database.py`, `transcriber.py`, `vad.py`, `embeddings.py`

---

## Phase 2: Transcript Playback & Editing (v0.5)

> Interactive transcript

**Goal:** Audio playback tied to transcript segments, inline editing. Foundation for all future transcript features.

### Transcript storage

- [x] Table `session_segments`: session_id, track_num, start_ms, end_ms, speaker, text, sort_order
- [ ] CLI: `meetscribe list` — transcription list with metadata
- [ ] CLI: `meetscribe show <id>` — view

### Transcript playback (web)

- [x] Structured segments stored in DB alongside markdown
- [x] Global audio player: play/pause, seekable progress bar, time display
- [x] Synchronized multi-track playback, per-track mute
- [x] Active segment highlighting during playback
- [x] Active track highlighting (which track the current segment belongs to)
- [x] Click segment to play from that point

### Transcript editing (web)

- [ ] Inline text editing per segment
- [ ] Speaker reassignment per segment
- [ ] Delete segment
- [ ] Merge adjacent segments
- [ ] Split segment
- [ ] Regenerate markdown after edits

### Files

- New: `migrations/002_transcript_segments.sql`
- Modified: `pipeline/models.py`, `web/models.py`, `web/services/pipeline.py`, `web/services/session.py`, `web/routes/tasks.py`, `web/static/js/app.js`, `web/templates/steps/step6_transcribe.html`

---

## Phase 3: Transcript Intelligence (v0.6)

> More than text

**Goal:** LLM post-processing and export. Most user-visible value gain.

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

### Custom dictionary

- [ ] User-defined word list for correcting common ASR typos (names, jargon, abbreviations)
- [ ] Applied as post-processing after transcription
- [ ] CLI: `meetscribe dict add "Speaches" "Speeches"` or YAML config section

### Action items

- [ ] Task + assignee (from speaker name) + deadline (if mentioned)
- [ ] Format: `- [ ] @Speaker: task description`
- [ ] CLI: `meetscribe transcribe --summarize --action-items`

### Export

| Format | Description |
|--------|-------------|
| Markdown | Current format (default) |
| SRT | Subtitles with timecodes |
| VTT | WebVTT subtitles |
| JSON | Structured (segments with speaker, start, end, text) |
| TXT | Plain text (speaker: text, no timecodes) |

- [ ] CLI: `meetscribe export <session-id> --format srt|vtt|json|txt|md`
- [ ] Web: download buttons on the result page

### Files

- New: `pipeline/llm.py`, `pipeline/export.py`
- Modified: `database.py`, `cli.py`, `config.py`, `web/routes/tasks.py`

---

## Phase 4: Search & Analytics (v0.7)

> Organizational memory

**Goal:** Meeting archive as a searchable knowledge base.

### Full-text search

- [ ] SQLite FTS5 on `session_segments`
- [ ] CLI: `meetscribe search "quarterly revenue"` — segments with context, date, speaker
- [ ] Web: search bar with result highlighting
- [ ] Filters: date, speaker, team

### Speaker analytics

| Metric | Description |
|--------|-------------|
| Talk time | Total speaking time |
| Turn count | Number of turns |
| Avg turn duration | Average turn length |

- [ ] Computed during transcription, stored in `speaker_stats`
- [ ] CLI: `meetscribe stats <session-id>`
- [ ] Web: bar charts (CSS-only, no JS frameworks)

### Meeting dashboard (web)

- [ ] Meeting list: date, duration, speakers, summary preview
- [ ] Click → full transcript with speaker timeline
- [ ] Pagination, sort by date/duration

### CLI: machine-readable output

- [ ] `--json` for all list commands
- [ ] Pipe-friendly: `meetscribe search "topic" --json | jq '.segments[].text'`

### Batch processing

- [ ] `meetscribe transcribe *.mp4 -o output_dir/`
- [ ] Parallel processing with configurable concurrency
- [ ] Progress summary on completion

### Files

- New: `pipeline/analytics.py`, `web/routes/dashboard.py`
- Modified: `database.py` (FTS5), `cli.py`

---

## Phase 5: Web UI Maturity (v0.8)

> From workflow to application

**Goal:** Full-featured web interface for non-technical users.

### Transcript viewer

- [ ] Speaker color coding

### Frontend improvements

- [ ] Partial updates instead of full page reload
- [ ] SSE for real-time progress
- [ ] Tech choice TBD (HTMX, Alpine.js, or lightweight framework)

### Speaker management (dashboard)

- [ ] Enrolled speakers list with sample playback
- [ ] Play / delete individual samples to curate voiceprint quality
- [ ] Delete / rename speakers
- [ ] Voiceprint quality indicator (sample count, embedding spread)

### Session management

- [ ] Resume interrupted sessions
- [ ] Per-user history with status badges
- [ ] Delete old sessions with files

### Admin panel

- [ ] User and team management (currently CLI-only)
- [ ] Speaches server status
- [ ] Disk usage
- [ ] Recent error log

### Files

- New: `web/routes/admin.py`, `web/routes/transcript.py`
- Modified: all `web/templates/`, `web/services/session.py`

---

## Phase 6: Real-Time & Integrations (v0.9)

> Live meetings

**Goal:** Real-time transcription. Technically the most complex phase, but the strongest differentiator.

### WebSocket audio streaming

- [ ] Endpoint: `ws://.../v1/stream` — receive PCM chunks
- [ ] Server-side VAD on the stream
- [ ] Buffering → embeddings → transcription in near-real-time
- [ ] Push transcript updates back via WebSocket

### Browser recording

- [ ] MediaRecorder API → WebSocket
- [ ] Live transcript display in web UI
- [ ] System audio capture (screen sharing audio)

### REST API formalization

- [ ] OpenAPI documentation (FastAPI auto-generated)
- [ ] API key authentication (separate from cookie auth)
- [ ] Versioned API: `/api/v1/`

### Webhook notifications

- [ ] POST to Slack/Teams/any URL on events
- [ ] Events: `transcribe.complete`, `action_items.extracted`
- [ ] CLI: `meetscribe webhook add <url> --events transcribe.complete`

### Meeting bot (stretch goal)

- [ ] Plugin interface: `MeetingBotPlugin` with `join()`, `record()`, `leave()`
- [ ] First candidate: SIP/VoIP

### Files

- New: `web/routes/stream.py`, `pipeline/realtime.py`, `integrations/`
- Modified: `web/app.py`, `pipeline/vad.py`

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

- [ ] User guide (installation, configuration, CLI reference)
- [ ] API reference (auto-generated + guides)
- [ ] Deployment guide (Docker, bare metal, Kubernetes)

---

## Summary

| Phase | Version | Theme | Key Outcome |
|-------|---------|-------|-------------|
| 1 | v0.4 | Hardening | Tests, CI, reliability |
| 2 | v0.5 | Playback & Editing | Segment playback, multi-track sync, transcript editing |
| 3 | v0.6 | Intelligence | LLM summaries, action items, export |
| 4 | v0.7 | Search | Full-text search, analytics, dashboard |
| 5 | v0.8 | Web UI | Speaker color coding, admin panel, session management |
| 6 | v0.9 | Real-time | WebSocket streaming, webhooks, API |
| 7 | v1.0 | Enterprise | RBAC, metrics, plugins, Helm |

## Intentionally NOT doing

- **Desktop app (Electron)** — CLI + web covers all use cases
- **Mobile app** — responsive web is sufficient
- **Video recording/playback** — focus on audio
- **Training custom ASR models** — pluggable backend already supports model swapping
