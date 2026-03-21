"""Speaker diarization pipeline: VAD -> embeddings -> identification."""

import logging
from pathlib import Path

from .embeddings import EmbeddingExtractor, SpeakerIdentifier
from .models import SpeechSegment
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class DiarizationPipeline:
    """Decomposed speaker diarization pipeline.

    Steps:
        1. VAD (remote) — detect speech segments
        2. Speaker embeddings (remote) — extract embedding per segment
        3. Identification (local) — cosine similarity with stored voiceprints
    """

    def __init__(
        self,
        vad_url: str,
        embedding_url: str,
        voiceprints: dict[str, list[float]],
        threshold: float,
        vad_timeout: float,
        embedding_timeout: float,
        min_duration_ms: int,
        unknown_cluster_threshold: float,
        confident_gap: float,
        min_threshold: float,
        max_workers: int,
        embedding_model: str,
    ):
        self.vad = VoiceActivityDetector(vad_url, vad_timeout)
        self.embeddings = EmbeddingExtractor(
            embedding_url, embedding_timeout, min_duration_ms, model=embedding_model
        )
        self.identifier = SpeakerIdentifier(
            voiceprints, threshold, confident_gap, min_threshold, unknown_cluster_threshold
        )
        self.max_workers = max_workers

    def diarize(self, audio_path: Path) -> list[SpeechSegment]:
        """Run full diarization pipeline on an audio file.

        Args:
            audio_path: Path to audio file (WAV).

        Returns:
            List of SpeechSegment with speaker labels assigned.
        """
        # 1. VAD
        segments = self.vad.detect(audio_path)
        if not segments:
            return []

        logger.info("VAD: %d speech segments", len(segments))

        # 2. Extract embeddings
        segments_with_embeddings = self.embeddings.extract_segments(
            audio_path, segments, self.max_workers
        )
        logger.info("Embeddings: extracted for %d segments", len(segments_with_embeddings))

        # 3. Identify speakers
        labeled = self.identifier.identify_segments(segments_with_embeddings)
        speakers = {s.speaker for s in labeled if s.speaker}
        logger.info("Identification: %d speakers found", len(speakers))

        return labeled
