"""Speaker diarization pipeline: VAD -> embeddings -> identification."""

import logging
import time
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
        vad_min_silence_duration_ms: int = 1200,
        vad_speech_pad_ms: int = 30,
        vad_threshold: float = 0.5,
    ):
        self.vad = VoiceActivityDetector(
            vad_url,
            vad_timeout,
            min_silence_duration_ms=vad_min_silence_duration_ms,
            speech_pad_ms=vad_speech_pad_ms,
            threshold=vad_threshold,
        )
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
        t0 = time.perf_counter()

        # 1. VAD
        segments = self.vad.detect(audio_path)
        if not segments:
            return []

        # 2. Extract embeddings
        segments_with_embeddings = self.embeddings.extract_segments(
            audio_path, segments, self.max_workers
        )

        # 3. Identify speakers
        labeled = self.identifier.identify_segments(segments_with_embeddings)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        audio_duration_ms = labeled[-1].end_ms if labeled else 0
        speakers = {s.speaker for s in labeled if s.speaker}
        rtf = elapsed_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        logger.info(
            "Diarization completed",
            extra={
                "file": audio_path.name,
                "segments": len(labeled),
                "speakers": len(speakers),
                "audio_duration_ms": audio_duration_ms,
                "elapsed_ms": round(elapsed_ms),
                "rtf": round(rtf, 2),
            },
        )

        return labeled
