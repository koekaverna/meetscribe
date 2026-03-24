# MeetScribe — Roadmap

> v0.3.2 → v1.0 | 6 фаз | CLI + Web

## Текущее состояние

MeetScribe — CLI + Web инструмент для транскрибации встреч с диаризацией спикеров.

**Что работает:**
- Пайплайн: VAD → Embeddings → Clustering → Identification → Transcription (remote Speaches API)
- CLI: transcribe, enroll, extract, extract-samples, list-speakers, info, web, team, user
- Web UI: FastAPI + Jinja2, 6-step workflow, auth, team scoping, SSE progress
- БД: SQLite, 8 таблиц, миграции, multi-team
- ~4900 строк, 29 модулей

**Уникальное позиционирование:**
- CLI-first — нет конкурентов в этой нише
- Self-hosted / privacy-first — данные не покидают инфраструктуру
- Гибридная идентификация — enrolled voiceprints + авто-кластеризация unknown
- Pluggable backend — Speaches API, сменяемые модели

## Конкурентный ландшафт

| Решение | Тип | Ключевые фичи |
|---------|-----|----------------|
| Otter.ai | Cloud | Real-time, 99+ языков, collaborative notes, series tracking |
| Fireflies.ai | Cloud | 6000+ интеграций, sentiment analysis, conversation intelligence |
| Fathom | Cloud | Бесплатный tier, privacy-first, action items |
| MeetGeek | Cloud | Лучшие саммари, action items с assignee/deadline, team analytics |
| tl;dv | Cloud | Бесплатные записи, moment sharing, timestamp navigation |
| Grain | Cloud | Deal intelligence, collaborative annotation |
| WhisperX | OSS | Whisper + Pyannote, word-level timestamps, batch-only |
| pyannote | OSS | Diarization models, ~11-19% DER, без транскрипции |
| NeMo | OSS | NVIDIA, GPU-оптимизированная диаризация |

**Общие фичи коммерческих решений, отсутствующие в MeetScribe:**
- AI-саммари и action items
- Поиск по архиву транскриптов
- Интеграции с платформами (Zoom, Teams, Meet)
- Аналитика спикеров (talk time, engagement)
- Real-time транскрибация
- Множественные форматы экспорта
- Редактирование транскриптов

---

## Фаза 1: Hardening (v0.4)

> Надёжная основа

**Цель:** Надёжность для ежедневного использования. Без новых фич — только уверенность в существующих.

### Тесты

#### Правила

- Каждый assert проверяет **конкретное значение**, не `is not None`, не `isinstance`
- Мокаем только внешние зависимости (httpx, filesystem), не тестируемый объект
- Тестируем **поведение**, не реализацию — не завязываемся на внутренние методы

#### Фикстуры (`tests/conftest.py`)

- [x] In-memory SQLite с миграциями
- [x] Мок Speaches API (httpx responses для VAD, embeddings, transcriptions)
- [x] WAV-файл 16kHz mono (короткий, в памяти)
- [x] Embedding-векторы (нормализованные float-списки)
- [x] Минимальный config.yaml (temp file)

#### Unit-тесты (чистые функции)

**`tests/test_models.py`** — `pipeline/models.py`:
- [x] `merge_close_segments()` — empty, single, merge same speaker, skip different speakers, max_chunk_ms limit, gap > max_gap_ms
- [x] `merge_by_proximity()` — то же но без учёта спикера, speaker preserved from first

**`tests/test_clustering.py`** — `pipeline/clustering.py`:
- [x] `cluster_embeddings()` — 2 голоса → 2 кластера, 1 голос → 1 кластер, пороги влияют на кол-во кластеров

**`tests/test_identification.py`** — `pipeline/embeddings.py` (локальная логика):
- [x] `SpeakerIdentifier.identify()` — direct match (sim ≥ threshold), confident gap accept, below min_threshold reject, no voiceprints → None
- [x] `SpeakerIdentifier._find_nearest_labeled()` — left/right neighbor, both sides, no neighbors
- [x] `SpeakerIdentifier.identify_segments()` — all known, all unknown → clusters, mixed, short segments inherit from neighbor
- [x] `enroll_samples()` — копирует файлы в enrolled_dir, включает ранее зарегистрированные семплы, возвращает correct count

**`tests/test_config.py`** — `servers.py`:
- [x] `load_config()` — file not found, empty file, валидный YAML, дефолты при отсутствии секций
- [x] `AppConfig.validate()` — пустой servers list, несуществующий server name
- [x] **Дефолты совпадают с config.yaml** — парсим оба, сравниваем значения (git-история: 3 бага из-за расхождений)

#### Функциональные тесты (БД + файловая система)

**`tests/test_database.py`** — `database.py` с in-memory SQLite:
- [x] Миграции: идемпотентность (повторный `get_db()`)
- [x] `_validate_team_name()` — валидные/невалидные имена
- [x] Team: create + delete, защита "default" от удаления, cascade к voiceprints
- [x] Voiceprint: save (insert + upsert), load по team, delete
- [x] **Voiceprint model metadata** — save хранит именно `embeddings.model`, не `transcription.model` (git: commit 4242478)
- [x] Auth sessions: create, get с проверкой expiry, delete_expired
- [x] `ensure_default_team()` — идемпотентность

**`tests/test_team.py`** — `team.py`:
- [x] `resolve_team()` — авто-создание "default", резолв существующей команды, ошибка на несуществующую

**`tests/test_session.py`** — `web/services/session.py` (БД + temp dirs):
- [x] Lifecycle: create → add track → configure → delete (cleanup файлов)
- [x] Track: renumbering при удалении средней дорожки
- [x] Sample: add → move между спикерами → delete (cleanup файлов)
- [x] Rollback: ошибка при add_track → файл не остаётся
- [x] **Enrollment копирует семплы на диск** (git: commit 4242478 — web не копировал)

#### Интеграционные тесты (мок HTTP)

**`tests/test_pipeline_integration.py`** — полный flow с моками Speaches:
- [x] VAD: мок → конкретные сегменты с ожидаемыми start_ms/end_ms; мок ошибки → raise
- [x] Embeddings: `extract_segments()` — параллельное извлечение, фильтрация коротких сегментов (< min_duration_ms)
- [x] Transcriber: `transcribe_segments()` — merge → slice → transcribe → проверяем offset timestamps и speaker assignment
- [x] Diarization: VAD → Embeddings → Clustering → Identification — end-to-end, проверяем финальные speaker labels
- [x] `_find_speaker()` — overlap calculation, no overlap → "Unknown"

#### Осознанно НЕ тестируем

- `SpeechSegment.duration_ms` — тривиальное `end_ms - start_ms`
- `cosine_similarity()`, `compute_voiceprint()` — тестировали бы numpy
- `hash_password()` / `verify_password()` — тестировали бы werkzeug
- `_slice_wav()` — low-level binary, валидация = Speaches принял аудио
- `config.py` path resolution — простой if/else по платформе
- Тривиальные DB getters (`get_team`, `get_user_by_id`, `list_users`) — один SELECT
- Auth service flow (register/login/logout) — тонкая обвязка
- `pipeline/audio.py` — FFmpeg subprocess, зависит от системного бинарника
- CLI subcommands — тяжёлые для unit-тестов, покрываются ручным тестированием
- `_format_elapsed()` — CLI-форматирование, сломается — увидим глазами
- Проверка что модель есть на сервере — невозможно без запущенного сервера
- Проверка аргументов HTTP-запроса — нет надёжного способа без e2e

### CI (GitHub Actions)

- [x] ruff check + ruff format --check
- [x] mypy
- [x] pytest --cov
- [x] bandit
- [x] Matrix: Python 3.12 + 3.13
- [ ] Docker: build image + verify migrations + CLI entry point (smoke test)

### Типизированные ошибки

- [x] `SpeachesAPIError` — ошибки удалённого API
- [x] `PipelineError` — ошибки обработки
- [x] `ConfigurationError` — невалидная конфигурация
- [x] HTTP retry для transient failures (httpx retries)

### Версионирование миграций БД

- [x] Таблица `schema_version` + нумерованные миграции
- [x] Замена текущего `CREATE TABLE IF NOT EXISTS` (не поддерживает ALTER)

### Structured logging

- [x] Тайминги: VAD took X ms, embedding extraction took Y ms, transcription took Z ms

### Файлы

- Новые: `tests/`, `.github/workflows/ci.yml`
- Изменяемые: `database.py`, `transcriber.py`, `vad.py`, `embeddings.py`

---

## Фаза 2: Transcript Intelligence (v0.5)

> Больше, чем текст

**Цель:** LLM-постобработка и экспорт. Самый заметный для пользователя прирост ценности.

**Почему:** Каждый коммерческий конкурент имеет AI-саммари. Организации, которые не могут использовать облачные инструменты, нуждаются в этом локально.

### LLM-интеграция

```yaml
# config.yaml
llm:
  url: "http://localhost:11434/v1"   # Ollama, vLLM, llama.cpp
  model: "llama3.1"
  timeout: 120
  max_tokens: 4096
```

- [ ] `src/meetscribe/pipeline/llm.py` — OpenAI-compatible клиент
- [ ] Graceful degradation: если LLM не настроен — пропускаем пост-обработку

### Саммари встречи

- [ ] Executive summary
- [ ] Ключевые обсуждения
- [ ] Принятые решения
- [ ] Chunked processing для длинных транскриптов (split по speaker turns)
- [ ] Результат: markdown-секции в конце транскрипта

### Action items

- [ ] Задача + ответственный (из имени спикера) + дедлайн (если упомянут)
- [ ] Формат: `- [ ] @Speaker: описание задачи`
- [ ] CLI: `meetscribe transcribe --summarize --action-items`

### Экспорт

| Формат | Описание |
|--------|----------|
| Markdown | Текущий формат (по умолчанию) |
| SRT | Субтитры с таймкодами |
| VTT | WebVTT субтитры |
| JSON | Структурированный (segments с speaker, start, end, text) |
| TXT | Plain text (speaker: текст, без таймкодов) |

- [ ] CLI: `meetscribe export <session-id> --format srt|vtt|json|txt|md`
- [ ] Web: кнопки скачивания на странице результата

### Хранение транскриптов

- [ ] Таблица `transcript_segments`: session_id, start_ms, end_ms, speaker, text
- [ ] CLI: `meetscribe list` — список транскрибаций с метаданными
- [ ] CLI: `meetscribe show <id>` — просмотр

### Файлы

- Новые: `pipeline/llm.py`, `pipeline/export.py`
- Изменяемые: `database.py`, `cli.py`, `servers.py`, `web/routes/tasks.py`

---

## Фаза 3: Поиск и аналитика (v0.6)

> Организационная память

**Цель:** Архив встреч как поисковая база знаний.

### Полнотекстовый поиск

- [ ] SQLite FTS5 по `transcript_segments`
- [ ] CLI: `meetscribe search "quarterly revenue"` — сегменты с контекстом, датой, спикером
- [ ] Web: поисковая строка с подсветкой результатов
- [ ] Фильтры: дата, спикер, команда

### Аналитика спикеров

| Метрика | Описание |
|---------|----------|
| Talk time | Общее время речи спикера |
| Turn count | Число реплик |
| Avg turn duration | Средняя длительность реплики |

- [ ] Вычисление при транскрибации, хранение в `speaker_stats`
- [ ] CLI: `meetscribe stats <session-id>`
- [ ] Web: bar charts (CSS-only, без JS-фреймворков)

### Дашборд встреч (web)

- [ ] Список встреч: дата, длительность, спикеры, превью саммари
- [ ] Клик → полный транскрипт с timeline спикеров
- [ ] Пагинация, сортировка по дате/длительности

### CLI: machine-readable output

- [ ] `--json` для всех list-команд
- [ ] Pipe-friendly: `meetscribe search "topic" --json | jq '.segments[].text'`

### Batch processing

- [ ] `meetscribe transcribe *.mp4 -o output_dir/`
- [ ] Параллельная обработка с настраиваемой конкурентностью
- [ ] Прогресс-сводка по завершении

### Файлы

- Новые: `pipeline/analytics.py`, `web/routes/dashboard.py`
- Изменяемые: `database.py` (FTS5), `cli.py`

---

## Фаза 4: Web UI Maturity (v0.7)

> От воркфлоу к приложению

**Цель:** Полноценный веб-интерфейс для нетехнических пользователей. HTMX — без build step, без JS-фреймворка.

### Просмотр и редактирование транскрипта

- [ ] Цветовая маркировка спикеров
- [ ] Timestamps и copy-to-clipboard
- [ ] Inline editing: клик на сегмент → исправление текста или переназначение спикера
- [ ] Re-export после редактирования

### HTMX-миграция

- [ ] Partial updates вместо полной перезагрузки страницы
- [ ] SSE через HTMX для real-time progress
- [ ] Jinja2 остаётся, добавляются `hx-*` атрибуты
- [ ] Без build step, без node_modules — один JS-файл

### Управление спикерами

- [ ] Список enrolled спикеров с проигрыванием семплов
- [ ] Удаление / переименование
- [ ] Визуальный индикатор качества voiceprint (кол-во семплов, разброс эмбеддингов)

### Управление сессиями

- [ ] Возобновление прерванных сессий
- [ ] История по пользователю со статус-бейджами
- [ ] Удаление старых сессий с файлами

### Админ-панель

- [ ] Управление пользователями и командами (сейчас только CLI)
- [ ] Статус серверов Speaches
- [ ] Использование диска
- [ ] Лог последних ошибок

### Файлы

- Новые: `web/routes/admin.py`, `web/routes/transcript.py`
- Изменяемые: все `web/templates/`, `web/services/session.py`

---

## Фаза 5: Real-Time и интеграции (v0.8)

> Живые встречи

**Цель:** Транскрибация в реальном времени. Технически самая сложная фаза, но самый мощный дифференциатор.

### WebSocket audio streaming

- [ ] Endpoint: `ws://.../v1/stream` — приём PCM-чанков
- [ ] Server-side VAD на потоке
- [ ] Буферизация → embeddings → transcription в near-real-time
- [ ] Push обновлений транскрипта обратно через WebSocket

### Запись из браузера

- [ ] MediaRecorder API → WebSocket
- [ ] Live-отображение транскрипта в web UI
- [ ] Захват системного аудио (screen sharing audio)

### REST API формализация

- [ ] OpenAPI документация (FastAPI auto-generated)
- [ ] API key аутентификация (отдельно от cookie auth)
- [ ] Версионированный API: `/api/v1/`

### Webhook-нотификации

- [ ] POST на Slack/Teams/любой URL при событиях
- [ ] Events: `transcribe.complete`, `action_items.extracted`
- [ ] CLI: `meetscribe webhook add <url> --events transcribe.complete`

### Meeting bot (stretch goal)

- [ ] Plugin-интерфейс: `MeetingBotPlugin` с `join()`, `record()`, `leave()`
- [ ] Первый кандидат: SIP/VoIP

### Файлы

- Новые: `web/routes/stream.py`, `pipeline/realtime.py`, `integrations/`
- Изменяемые: `web/app.py`, `pipeline/vad.py`

---

## Фаза 6: Enterprise (v1.0)

> Production Grade

**Цель:** Готовность к production-деплою в организациях.

### Безопасность

- [ ] Rate limiting на auth endpoints
- [ ] RBAC: viewer / editor / admin
- [ ] Audit log: кто, что, когда
- [ ] Валидация загрузок (magic bytes)

### Observability

- [ ] Prometheus `/metrics`: latency, pipeline duration, queue depth
- [ ] Structured JSON logs для log aggregation
- [ ] `StructuredFormatter` — перейти на namespace key (`_ctx`) вместо denylist
- [ ] Health check с проверкой Speaches API connectivity

### Deployment

- [ ] Helm chart для Kubernetes
- [ ] Docker Compose profiles (`--profile gpu`)
- [ ] Автоматический backup SQLite
- [ ] Multi-worker uvicorn с file locking

### Plugin system

- [ ] `meetscribe.plugins` entry point
- [ ] Рефакторинг саммари / action items в плагины
- [ ] Интерфейс: `TranscriptPlugin.process(segments) -> dict`

### Документация

- [ ] User guide (установка, конфигурация, CLI reference)
- [ ] API reference (auto-generated + guides)
- [ ] Deployment guide (Docker, bare metal, Kubernetes)

---

## Сводная таблица

| Фаза | Версия | Тема | Ключевой результат |
|------|--------|------|--------------------|
| 1 | v0.4 | Hardening | Тесты, CI, надёжность |
| 2 | v0.5 | Intelligence | LLM-саммари, action items, экспорт |
| 3 | v0.6 | Search | Полнотекстовый поиск, аналитика, дашборд |
| 4 | v0.7 | Web UI | HTMX, редактор транскриптов, админка |
| 5 | v0.8 | Real-time | WebSocket streaming, webhooks, API |
| 6 | v1.0 | Enterprise | RBAC, metrics, plugins, Helm |

## Осознанно НЕ делаем

- **Desktop app (Electron)** — CLI + web покрывает все кейсы
- **Mobile app** — responsive web достаточно
- **Видеозапись/воспроизведение** — фокус на аудио
- **Тренировка собственных ASR-моделей** — pluggable backend уже поддерживает смену моделей
