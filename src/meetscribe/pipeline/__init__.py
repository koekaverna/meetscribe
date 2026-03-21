"""Pipeline modules for meeting processing."""

from . import audio
from .diarization import DiarizationPipeline
from .embeddings import EmbeddingExtractor, SpeakerIdentifier, save_voiceprint
from .models import SpeechSegment, TranscriptSegment, merge_by_proximity, merge_close_segments
from .transcriber import Transcriber
from .vad import VoiceActivityDetector

__all__ = [
    "audio",
    "DiarizationPipeline",
    "EmbeddingExtractor",
    "SpeakerIdentifier",
    "SpeechSegment",
    "TranscriptSegment",
    "Transcriber",
    "VoiceActivityDetector",
    "merge_by_proximity",
    "merge_close_segments",
    "save_voiceprint",
]
