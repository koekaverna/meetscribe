"""Pipeline modules for meeting processing."""

from . import audio
from .diarization import DiarizationPipeline
from .embeddings import EmbeddingExtractor, SpeakerIdentifier, compute_voiceprint, enroll_samples
from .models import SpeechSegment, TranscriptSegment, merge_by_proximity, merge_close_segments
from .transcriber import Transcriber
from .vad import VoiceActivityDetector

__all__ = [
    "audio",
    "DiarizationPipeline",
    "EmbeddingExtractor",
    "compute_voiceprint",
    "enroll_samples",
    "SpeakerIdentifier",
    "SpeechSegment",
    "TranscriptSegment",
    "Transcriber",
    "VoiceActivityDetector",
    "merge_by_proximity",
    "merge_close_segments",
]
