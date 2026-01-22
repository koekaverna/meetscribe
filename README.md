# MeetScribe

Meeting transcription with speaker diarization. Combines OpenAI Whisper for transcription, SpeechBrain for voice activity detection and speaker embeddings, and spectral clustering for speaker separation.

## Features

- **Two-track processing**: Separate host and guests audio tracks for better accuracy
- **Speaker enrollment**: Register speakers with voice samples for automatic identification
- **Speaker diarization**: Automatically separate and identify speakers without enrollment
- **Configurable**: Whisper model size, language, identification threshold

## Installation

Requires Python 3.12+ and CUDA-compatible GPU (recommended).

```bash
# Using uv (recommended)
uv pip install meetscribe

# Or with pip
pip install meetscribe
```

### CUDA Support

For CUDA acceleration, install PyTorch with CUDA support first:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install meetscribe
```

## Usage

### Enroll Speakers

Register known speakers for automatic identification:

```bash
# From a directory of audio samples
meetscribe enroll "John Doe" ./samples/john/

# From a single file
meetscribe enroll "Jane Smith" recording.wav
```

### Transcribe Meeting

Transcribe a meeting with speaker identification:

```bash
meetscribe transcribe meeting.mp4 --host "John Doe" --output ./notes/
```

Options:
- `-H, --host`: Name of the host (required)
- `-o, --output`: Output file or directory (required)
- `-m, --model`: Whisper model size (default: medium)
- `-l, --language`: Language code (default: ru)
- `--max-speakers`: Maximum number of speakers to detect (default: 10)
- `--threshold`: Speaker identification confidence threshold (default: 0.7)

### Extract Samples

Extract audio samples from unknown speakers for later enrollment:

```bash
meetscribe extract-samples meeting.mp4
```

### List Speakers

Show enrolled speakers:

```bash
meetscribe list-speakers
```

### Show Configuration

Display data directories and settings:

```bash
meetscribe info
```

## Audio Format

MeetScribe expects video files with two audio tracks:
1. **Track 1**: Host audio (e.g., from local microphone)
2. **Track 2**: Guests audio (e.g., from remote participants)

This separation improves transcription accuracy by isolating speakers.

## Data Directories

MeetScribe stores data in platform-specific directories:

| Platform | Data | Cache |
|----------|------|-------|
| Windows | `%LOCALAPPDATA%\meetscribe` | `%LOCALAPPDATA%\meetscribe\cache` |
| macOS | `~/Library/Application Support/meetscribe` | `~/Library/Caches/meetscribe` |
| Linux | `~/.local/share/meetscribe` | `~/.cache/meetscribe` |

## Pipeline

1. **Audio extraction**: Extract separate tracks from video using FFmpeg
2. **VAD**: Voice Activity Detection using SpeechBrain CRDNN
3. **Embeddings**: Speaker embeddings using ECAPA-TDNN
4. **Clustering**: Spectral clustering to group segments by speaker
5. **Identification**: Match clusters to enrolled speakers
6. **Transcription**: Whisper transcription with speaker labels

## Models

- **VAD**: [speechbrain/vad-crdnn-libriparty](https://huggingface.co/speechbrain/vad-crdnn-libriparty)
- **Embeddings**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Transcription**: [OpenAI Whisper](https://github.com/openai/whisper)

## License

MIT
