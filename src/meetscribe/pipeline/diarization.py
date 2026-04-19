"""Speaker diarization via speaches /v1/audio/diarization + local speaker identification."""

import logging
import time
import wave
from pathlib import Path

import httpx

from meetscribe.errors import SpeachesAPIError, speaches_retry

from .embeddings import EmbeddingExtractor, SpeakerIdentifier, slice_wav
from .models import SpeechSegment

logger = logging.getLogger(__name__)


class DiarizationPipeline:
    """Speaker diarization using speaches server-side diarization.

    Steps:
        1. Upload audio to /v1/audio/diarization (segmentation + VBx clustering)
        2. For each cluster, extract embedding from representative segment
        3. Match cluster embeddings against stored voiceprints
    """

    def __init__(
        self,
        diarization_url: str,
        embedding_url: str,
        voiceprints: dict[str, list[float]],
        threshold: float,
        confident_gap: float,
        min_threshold: float,
        diarization_timeout: float,
        embedding_timeout: float,
        min_duration_ms: int,
        embedding_model: str,
        diarization_model: str = "fedirz/segmentation_community_1",
    ):
        self.diarization_url = diarization_url.rstrip("/")
        self.diarization_timeout = diarization_timeout
        self.diarization_model = diarization_model

        self.embeddings = EmbeddingExtractor(
            embedding_url, embedding_timeout, min_duration_ms, model=embedding_model
        )
        self.identifier = SpeakerIdentifier(voiceprints, threshold, confident_gap, min_threshold)

    def diarize(self, audio_path: Path) -> list[SpeechSegment]:
        """Run full diarization pipeline on an audio file.

        Args:
            audio_path: Path to audio file (WAV).

        Returns:
            List of SpeechSegment with speaker labels assigned.
        """
        t0 = time.perf_counter()

        # 1. Call speaches diarization API
        raw_segments = self._call_diarization(audio_path)
        if not raw_segments:
            return []

        # 2. Convert to SpeechSegments (seconds -> ms)
        segments = [
            SpeechSegment(
                start_ms=int(s["start"] * 1000),
                end_ms=int(s["end"] * 1000),
                speaker=s["speaker"],
            )
            for s in raw_segments
        ]

        # 3. Build cluster -> known speaker mapping
        speaker_map = self._map_clusters_to_speakers(audio_path, segments)

        # 4. Apply mapping
        for seg in segments:
            if seg.speaker is not None:
                seg.speaker = speaker_map.get(seg.speaker, seg.speaker)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        speakers = {s.speaker for s in segments if s.speaker}
        speech_duration_ms = segments[-1].end_ms if segments else 0
        speech_rtf = elapsed_ms / speech_duration_ms if speech_duration_ms > 0 else 0
        logger.info(
            "Diarization completed",
            extra={
                "file": audio_path.name,
                "segments": len(segments),
                "speakers": len(speakers),
                "speech_duration_ms": speech_duration_ms,
                "elapsed_ms": round(elapsed_ms),
                "speech_rtf": round(speech_rtf, 2),
            },
        )

        return segments

    @speaches_retry
    def _call_diarization(self, audio_path: Path) -> list[dict]:
        """Call speaches /v1/audio/diarization endpoint."""
        endpoint = f"{self.diarization_url}/v1/audio/diarization"
        try:
            with open(audio_path, "rb") as f:
                response = httpx.post(
                    endpoint,
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={"model": self.diarization_model},
                    timeout=self.diarization_timeout,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SpeachesAPIError(
                f"Diarization failed: {e.response.status_code}",
                status_code=e.response.status_code,
                endpoint=endpoint,
                detail=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise SpeachesAPIError(
                f"Diarization connection error: {e}",
                endpoint=endpoint,
            ) from e

        result: dict[str, list[dict]] = response.json()
        return result.get("segments", [])

    def _map_clusters_to_speakers(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
    ) -> dict[str, str]:
        """Map cluster labels to known speaker names or Unknown-N labels.

        For each cluster:
          1. Pick the longest segment (within 3-12s) as representative
          2. Extract its embedding
          3. Try to match against voiceprints
          4. If no match, assign Unknown-N label
        """
        if not segments:
            return {}

        # Group segments by cluster label
        clusters: dict[str, list[SpeechSegment]] = {}
        for seg in segments:
            if seg.speaker is not None:
                clusters.setdefault(seg.speaker, []).append(seg)

        # Load audio once for slicing
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            raw_frames = wf.readframes(wf.getnframes())

        speaker_map: dict[str, str] = {}
        unknown_counter = 0

        for cluster_label, cluster_segs in clusters.items():
            # Pick best representative: longest segment in 3-12s range
            candidates = [s for s in cluster_segs if 3000 <= s.duration_ms <= 12000]
            if not candidates:
                # Fallback: longest segment above min_duration_ms
                candidates = [
                    s for s in cluster_segs if s.duration_ms >= self.embeddings.min_duration_ms
                ]
            if not candidates:
                # No suitable segment — can't extract embedding
                unknown_counter += 1
                speaker_map[cluster_label] = f"Unknown-{unknown_counter}"
                logger.debug(
                    "No suitable segment for cluster",
                    extra={"cluster": cluster_label, "assigned": speaker_map[cluster_label]},
                )
                continue

            candidates.sort(key=lambda s: s.duration_ms, reverse=True)
            representative = candidates[0]

            # Extract embedding from representative segment
            try:
                wav_bytes = slice_wav(raw_frames, sample_rate, sample_width, representative)
                embedding = self.embeddings.extract(wav_bytes)
            except Exception:
                logger.warning(
                    "Failed to extract embedding for cluster",
                    extra={"cluster": cluster_label},
                    exc_info=True,
                )
                unknown_counter += 1
                speaker_map[cluster_label] = f"Unknown-{unknown_counter}"
                continue

            # Try to match against voiceprints
            name, sim = self.identifier.identify(embedding)
            if name is not None:
                speaker_map[cluster_label] = name
                logger.info(
                    "Cluster matched to known speaker",
                    extra={
                        "cluster": cluster_label,
                        "speaker": name,
                        "similarity": round(sim, 3),
                    },
                )
            else:
                unknown_counter += 1
                speaker_map[cluster_label] = f"Unknown-{unknown_counter}"
                logger.info(
                    "Cluster is unknown",
                    extra={
                        "cluster": cluster_label,
                        "assigned": speaker_map[cluster_label],
                        "best_similarity": round(sim, 3),
                    },
                )

        return speaker_map
