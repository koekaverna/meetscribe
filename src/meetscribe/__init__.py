"""MeetScribe - Meeting transcription with speaker diarization."""

__version__ = "0.1.0"

from .pipeline import (
    VADProcessor,
    EmbeddingExtractor,
    SpectralClusterer,
    SpeakerIdentifier,
    Transcriber,
)

__all__ = [
    "__version__",
    "VADProcessor",
    "EmbeddingExtractor",
    "SpectralClusterer",
    "SpeakerIdentifier",
    "Transcriber",
]
