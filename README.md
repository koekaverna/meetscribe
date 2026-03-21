# MeetScribe

Meeting transcription with speaker diarization using remote [speaches](https://github.com/speaches-ai/speaches) API servers. Offloads VAD, speaker embeddings, and transcription to GPU servers — the client stays lightweight with no ML dependencies.

## Features

- **Remote processing**: VAD, speaker embeddings, and transcription via speaches API (OpenAI-compatible)
- **Web UI**: Browser-based interface with step-by-step workflow
- **User authentication**: Login/password auth with team-scoped access
- **Multi-track processing**: Handle video files with multiple audio tracks or individual audio files
- **Speaker enrollment**: Register speakers with voice samples for automatic identification
- **Speaker diarization**: Automatically separate and identify speakers without enrollment
- **Multi-team support**: Separate speaker databases and sessions per team
- **Parallel transcription**: Distribute chunks across multiple servers
- **Flexible input**: Video files, audio files, directories, or glob patterns

## Installation

### Requirements

- Python 3.12+
- FFmpeg (for audio extraction and conversion)
- One or more [speaches](https://github.com/speaches-ai/speaches) API servers

### FFmpeg

FFmpeg is required for processing audio and video files.

**Windows:**
```bash
winget install "FFmpeg (Shared)"
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

After installation, restart your terminal to update PATH.

### Python package

```bash
uv venv
uv pip install -e ".[web]"
```

The `[web]` extra installs FastAPI, Uvicorn, Jinja2 and other web dependencies. Omit it for CLI-only usage.

## Configuration

MeetScribe uses two configuration files:

| File | Purpose |
|------|---------|
| `.env` | Data directory path and environment settings |
| `data/config.yaml` | Servers, pipeline parameters, web UI settings |

### Quick start

```bash
# 1. Set up environment
cp .env.example .env

# 2. Set up config
cp config.example.yaml data/config.yaml
# Edit data/config.yaml — set your server URL

# 3. Initialize database and create admin user
meetscribe team create default
meetscribe user create admin --team default --admin
```

### `.env`

Controls where MeetScribe stores its data. See [.env.example](.env.example).

| Variable | Description | Default |
|----------|-------------|---------|
| `MEETSCRIBE_DATA_DIR` | Root directory for DB, logs, sessions, samples | Platform-specific (see below) |
| `MEETSCRIBE_TMP_DIR` | Temp files directory | `DATA_DIR/tmp` |
| `MEETSCRIBE_MAX_UPLOAD_SIZE` | Max upload size in bytes | `4294967296` (4 GB) |

Default data directory without `MEETSCRIBE_DATA_DIR`:
- **Windows:** `%LOCALAPPDATA%/meetscribe`
- **macOS:** `~/Library/Application Support/meetscribe`
- **Linux:** `~/.local/share/meetscribe`

Setting `MEETSCRIBE_DATA_DIR=./data` keeps everything in the project directory — convenient for development and debugging.

### `config.yaml`

Located at `MEETSCRIBE_DATA_DIR/config.yaml` (by default `./data/config.yaml`). All application settings in one file. See [config.example.yaml](config.example.yaml) for a fully documented example.

**Sections:**

- **`servers`** — List of speaches API servers (URL + name)
- **`vad`** — Voice Activity Detection: server, timeout
- **`embeddings`** — Speaker embeddings: server, timeout, identification thresholds, clustering parameters
- **`transcription`** — Speech-to-text: servers, model, language, timeout, segment merging
- **`web`** — Web UI: host, port, session TTL

## Web UI

Start the web interface:

```bash
meetscribe web
meetscribe web --host 0.0.0.0 --port 8080
```

Host and port can also be set in `config.yaml` under the `web` section. CLI arguments take priority.

### First-time setup

Create an admin user via CLI before using the web UI:

```bash
meetscribe user create admin --team default --admin
```

The admin can then register other users through the web UI at `/register`.

### Workflow

The web UI guides you through a 6-step process:

1. **Upload** — Upload video or audio files
2. **Configure** — Assign speakers to tracks or enable auto-diarization
3. **Extract** — Extract speaker samples via VAD + embeddings
4. **Samples** — Review and organize extracted speaker samples
5. **Enroll** — Register speakers from samples
6. **Transcribe** — Generate transcript with speaker attribution

### Access control

- Each user belongs to a team
- Sessions are visible only to users in the same team
- Only admin users can register new users (in their own team)
- Authentication uses HttpOnly cookies (works with SSE streaming)

## Teams

MeetScribe supports multiple teams, each with its own set of enrolled speakers, voice samples, and sessions. This enables separate speaker databases for different projects, clients, or departments.

All commands accept `-t/--team` flag to specify the team (defaults to `default`):

```bash
meetscribe -t sales enroll "John Doe" ./samples/john/
meetscribe -t sales transcribe meeting.mp4 -o output.md
meetscribe -t sales list-speakers
```

### Team management

```bash
meetscribe team create sales
meetscribe team list
meetscribe team delete sales
```

Team data is stored in `teams/<name>/samples/` under the data directory. Voiceprints are stored in a shared SQLite database (`meetscribe.db`), scoped per team.

## User management

```bash
# Create an admin user
meetscribe user create admin --team default --admin

# Create a regular user
meetscribe user create john --team sales

# List all users
meetscribe user list

# Delete a user
meetscribe user delete john
```

## CLI Commands

### `meetscribe transcribe`

Transcribe a meeting with speaker diarization:

```bash
meetscribe transcribe meeting.mp4 -o output.md --track1 "Host"
meetscribe transcribe path/to/tracks/ -o output.md --track1 "Host"
meetscribe transcribe track1.wav track2.wav -o output.md --track1 "Host"
```

Tracks without a `--trackN` assignment are diarized automatically.

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --team` | Team to use for speaker identification | `default` |
| `-o, --output` | Output file or directory | required |
| `-l, --language` | Language code (overrides config.yaml) | from config |
| `--trackN` | Assign speaker name to track N | diarize |

### `meetscribe enroll`

Register known speakers for automatic identification:

```bash
meetscribe enroll "John Doe" ./samples/john/
meetscribe -t my-team enroll "John Doe" ./samples/john/
```

### `meetscribe extract`

Extract audio tracks from a video file:

```bash
meetscribe extract meeting.mp4 -o output_dir/
```

### `meetscribe extract-samples`

Extract audio samples from unknown speakers for later enrollment:

```bash
meetscribe extract-samples meeting.mp4
```

### `meetscribe list-speakers`

Show enrolled speakers:

```bash
meetscribe list-speakers
meetscribe -t my-team list-speakers
```

### `meetscribe web`

Start the web UI server:

```bash
meetscribe web
meetscribe web --host 0.0.0.0 --port 8080
```

### `meetscribe info`

Display data directories, configuration, and settings:

```bash
meetscribe info
```

## Audio Input

MeetScribe supports multiple input formats:
- **Video files** (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`): audio tracks are extracted automatically
- **Audio files** (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`): used directly as tracks
- **Directories**: all audio files in the directory are used as tracks
- **Glob patterns**: matched audio files are used as tracks

For video files with multiple audio tracks (e.g., track 1 = host, track 2 = guests), use `--trackN` to assign speaker names.

## Development

```bash
uv venv
uv pip install -e ".[dev,web]"

pytest
ruff check src/
ruff format src/
```

## License

MIT
