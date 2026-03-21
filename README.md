# MeetScribe

Meeting transcription with speaker diarization using remote [speaches](https://github.com/speaches-ai/speaches) API servers. Offloads VAD, speaker embeddings, and transcription to GPU servers — the client stays lightweight with no ML dependencies.

## Features

- **Remote processing**: VAD, speaker embeddings, and transcription via speaches API (OpenAI-compatible)
- **Multi-track processing**: Handle video files with multiple audio tracks or individual audio files
- **Speaker enrollment**: Register speakers with voice samples for automatic identification
- **Speaker diarization**: Automatically separate and identify speakers without enrollment
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
uv pip install -e .
```

## Server Configuration

MeetScribe requires at least one speaches API server for processing. Copy `servers.example.yaml` to your data directory as `servers.yaml`:

- **Windows:** `%LOCALAPPDATA%/meetscribe/servers.yaml`
- **macOS:** `~/Library/Application Support/meetscribe/servers.yaml`
- **Linux:** `~/.local/share/meetscribe/servers.yaml`

Run `meetscribe info` to see the exact path.

```yaml
servers:
  - url: http://192.168.1.100:8000
    name: "GPU-1"

vad:
  server: "GPU-1"

embeddings:
  server: "GPU-1"

transcribe:
  servers:
    - "GPU-1"
```

Multiple transcription servers enable parallel processing of audio chunks.

## Commands

### `meetscribe transcribe`

Transcribe a meeting with speaker diarization. Accepts a video file (extracts tracks automatically), audio files, a directory of audio files, or glob patterns:

```bash
# From video (extracts audio tracks automatically)
meetscribe transcribe meeting.mp4 -o output.md --track1 "Host"

# From a directory of audio files
meetscribe transcribe path/to/tracks/ -o output.md --track1 "Host"

# From individual audio files
meetscribe transcribe track1.wav track2.wav -o output.md --track1 "Host"

# From glob pattern
meetscribe transcribe path/to/tracks/*.wav -o output.md
```

Tracks without a `--trackN` assignment are diarized automatically.

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file or directory | required |
| `-l, --language` | Language code | ru |
| `--trackN` | Assign speaker name to track N (e.g. `--track1 "Name"`) | diarize |

### `meetscribe enroll`

Register known speakers for automatic identification:

```bash
meetscribe enroll "John Doe" ./samples/john/
meetscribe enroll "Jane Smith" recording.wav
```

### `meetscribe extract`

Extract audio tracks from a video file:

```bash
meetscribe extract meeting.mp4
meetscribe extract meeting.mp4 -o output_dir/
```

### `meetscribe extract-samples`

Extract audio samples from unknown speakers for later enrollment:

```bash
meetscribe extract-samples meeting.mp4
meetscribe extract-samples meeting.mp4 --max-speakers 5 --threshold 0.6
```

### `meetscribe list-speakers`

Show enrolled speakers:

```bash
meetscribe list-speakers
```

### `meetscribe info`

Display data directories, server configuration, and settings:

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
# Setup
uv venv
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
ruff format src/
```

## License

MIT
