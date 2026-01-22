# MeetScribe

Meeting transcription with speaker diarization. Combines OpenAI Whisper for transcription, SpeechBrain for voice activity detection and speaker embeddings, and spectral clustering for speaker separation.

## Features

- **Two-track processing**: Separate host and guests audio tracks for better accuracy
- **Speaker enrollment**: Register speakers with voice samples for automatic identification
- **Speaker diarization**: Automatically separate and identify speakers without enrollment
- **Configurable**: Whisper model size, language, identification threshold

## Installation

Requires Python 3.12+.

```bash
uv venv
uv pip install -e .
```

## Commands

### `meetscribe transcribe`

Transcribe a meeting with speaker identification:

```bash
meetscribe transcribe meeting.mp4 --host "John Doe" --output ./notes/
```

| Option | Description | Default |
|--------|-------------|---------|
| `-H, --host` | Name of the host | required |
| `-o, --output` | Output file or directory | required |
| `-m, --model` | Whisper model size | medium |
| `-l, --language` | Language code | ru |
| `--max-speakers` | Maximum speakers to detect | 10 |
| `--threshold` | Identification threshold | 0.7 |

### `meetscribe enroll`

Register known speakers for automatic identification:

```bash
meetscribe enroll "John Doe" ./samples/john/
meetscribe enroll "Jane Smith" recording.wav
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
```

### `meetscribe info`

Display data directories and settings:

```bash
meetscribe info
```

## Audio Format

MeetScribe expects video files with two audio tracks:
1. **Track 1**: Host audio (e.g., from local microphone)
2. **Track 2**: Guests audio (e.g., from remote participants)

## Development

```bash
# Setup
uv venv
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
pytest check src/
ruff format src/
```

## License

MIT
