"""Remote speaker embedding extraction and local speaker identification."""

import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from .audio import extract_segment
from .models import SpeechSegment

logger = logging.getLogger(__name__)

# Minimum segment duration for reliable embedding extraction.
# Segments shorter than this get None embedding and inherit speaker from neighbors.
MIN_EMBEDDING_DURATION_MS = 1500

# Stricter threshold for clustering unknown speakers together.
UNKNOWN_CLUSTER_THRESHOLD = 0.7


class EmbeddingExtractor:
    """Speaker embedding extraction via speaches POST /v1/audio/speech/embedding."""

    DEFAULT_MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

    def __init__(self, server_url: str, timeout: float = 60.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def extract(self, audio_bytes: bytes, filename: str = "audio.wav") -> list[float]:
        """Extract speaker embedding from audio bytes.

        Args:
            audio_bytes: WAV audio data.
            filename: Filename for the multipart upload.

        Returns:
            Embedding vector as list of floats.
        """
        response = httpx.post(
            f"{self.server_url}/v1/audio/speech/embedding",
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": self.DEFAULT_MODEL},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.error("Embedding failed: %s %s", response.status_code, response.text)
        response.raise_for_status()

        result = response.json()
        return result["data"][0]["embedding"]

    def extract_from_file(self, audio_path: Path) -> list[float]:
        """Extract speaker embedding from an audio file."""
        return self.extract(audio_path.read_bytes(), audio_path.name)

    def _extract_one(
        self, audio_path: Path, seg: SpeechSegment
    ) -> tuple[SpeechSegment, list[float] | None]:
        """Extract embedding for a single segment (used by thread pool)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = Path(tmp.name)

        try:
            extract_segment(audio_path, chunk_path, seg.start_ms, seg.end_ms)
            filename = f"seg_{seg.start_ms}_{seg.end_ms}.wav"
            embedding = self.extract(chunk_path.read_bytes(), filename)
            return seg, embedding
        except Exception:
            logger.warning(
                "Failed to extract embedding for segment %d-%dms",
                seg.start_ms,
                seg.end_ms,
                exc_info=True,
            )
            return seg, None
        finally:
            chunk_path.unlink(missing_ok=True)

    def extract_segments(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
        max_workers: int = 4,
    ) -> list[tuple[SpeechSegment, list[float] | None]]:
        """Extract embeddings for each speech segment in parallel.

        Cuts audio per segment via FFmpeg, then calls the embedding API.
        Segments shorter than MIN_EMBEDDING_DURATION_MS get None embedding.

        Returns:
            List of (segment, embedding_or_None) tuples in original order.
        """
        # Separate short segments (skip) from ones needing extraction
        ordered: dict[int, tuple[SpeechSegment, list[float] | None]] = {}
        to_extract: list[tuple[int, SpeechSegment]] = []

        for i, seg in enumerate(segments):
            if seg.duration_ms < MIN_EMBEDDING_DURATION_MS:
                logger.debug(
                    "Skipping short segment %d-%dms (%dms)",
                    seg.start_ms,
                    seg.end_ms,
                    seg.duration_ms,
                )
                ordered[i] = (seg, None)
            else:
                to_extract.append((i, seg))

        pbar = tqdm(total=len(segments), desc="  Embeddings", unit="seg", leave=False)
        pbar.update(len(ordered))  # short segments already done

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._extract_one, audio_path, seg): idx for idx, seg in to_extract
            }
            for future in as_completed(futures):
                idx = futures[future]
                ordered[idx] = future.result()
                pbar.update(1)

        pbar.close()
        return [ordered[i] for i in range(len(segments))]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SpeakerIdentifier:
    """Identify speakers by comparing embeddings against stored voiceprints."""

    def __init__(self, voiceprints: dict[str, list[float]], threshold: float = 0.6):
        self.voiceprints = voiceprints
        self.threshold = threshold
        logger.info("Loaded %d voiceprints (threshold=%.2f)", len(self.voiceprints), threshold)

    # If best match is below threshold but the gap between 1st and 2nd
    # candidate exceeds this value, accept the match anyway (confident gap).
    CONFIDENT_GAP = 0.2
    # Absolute minimum similarity even with a confident gap.
    MIN_THRESHOLD = 0.45

    def identify(self, embedding: list[float]) -> tuple[str | None, float]:
        """Find the best matching known speaker.

        Uses two criteria:
        1. Direct match: best_sim >= threshold
        2. Confident gap: best_sim >= MIN_THRESHOLD and gap to 2nd >= CONFIDENT_GAP

        Returns:
            (speaker_name, similarity) if matched, else (None, best_sim).
        """
        if not self.voiceprints:
            return None, -1.0

        scores = [(cosine_similarity(embedding, vp), name) for name, vp in self.voiceprints.items()]
        scores.sort(reverse=True)

        best_sim, best_name = scores[0]
        second_sim = scores[1][0] if len(scores) > 1 else -1.0

        if best_sim >= self.threshold:
            return best_name, best_sim

        # Confident gap: large margin over 2nd candidate
        gap = best_sim - second_sim
        if best_sim >= self.MIN_THRESHOLD and gap >= self.CONFIDENT_GAP:
            logger.debug(
                "Accepted %s via confident gap: sim=%.3f, gap=%.3f",
                best_name,
                best_sim,
                gap,
            )
            return best_name, best_sim

        return None, best_sim

    def identify_segments(
        self,
        segments_with_embeddings: list[tuple[SpeechSegment, list[float] | None]],
    ) -> list[SpeechSegment]:
        """Assign speaker labels to segments.

        Known speakers are matched by voiceprint similarity.
        Unknown speakers are clustered greedily so the same voice gets a consistent label.
        Segments with no embedding (too short) inherit speaker from the nearest
        identified neighbor by time proximity.
        """
        unknown_clusters: list[list[float]] = []  # centroid per cluster
        result: list[SpeechSegment] = []
        # Indices of segments with None embedding — resolved after first pass
        deferred: list[int] = []

        for seg, embedding in segments_with_embeddings:
            idx = len(result)
            result.append(seg)

            if embedding is None:
                seg.speaker = None  # resolved in second pass
                deferred.append(idx)
                continue

            name, sim = self.identify(embedding)
            if name is not None:
                seg.speaker = name
                logger.debug(
                    "Segment %d-%dms -> %s (sim=%.3f)",
                    seg.start_ms,
                    seg.end_ms,
                    name,
                    sim,
                )
            else:
                # Cluster unknown speakers
                cluster_idx = self._assign_unknown_cluster(embedding, unknown_clusters)
                seg.speaker = f"Unknown-{cluster_idx + 1}"
                logger.debug(
                    "Segment %d-%dms -> Unknown-%d (best_known_sim=%.3f)",
                    seg.start_ms,
                    seg.end_ms,
                    cluster_idx + 1,
                    sim,
                )

        # Second pass: assign deferred (short) segments to nearest neighbor
        for idx in deferred:
            seg = result[idx]
            neighbor = self._find_nearest_labeled(idx, result)
            seg.speaker = neighbor or "Unknown"
            logger.debug(
                "Segment %d-%dms -> %s (inherited from neighbor)",
                seg.start_ms,
                seg.end_ms,
                seg.speaker,
            )

        return result

    def _assign_unknown_cluster(self, embedding: list[float], clusters: list[list[float]]) -> int:
        """Assign embedding to an existing unknown cluster or create a new one."""
        best_idx = -1
        best_sim = -1.0

        for i, centroid in enumerate(clusters):
            sim = cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= UNKNOWN_CLUSTER_THRESHOLD and best_idx >= 0:
            return best_idx

        # New cluster
        clusters.append(embedding)
        return len(clusters) - 1

    @staticmethod
    def _find_nearest_labeled(idx: int, segments: list[SpeechSegment]) -> str | None:
        """Find speaker of the nearest segment that has a label."""
        # Search outward from idx
        left, right = idx - 1, idx + 1
        while left >= 0 or right < len(segments):
            if left >= 0 and segments[left].speaker:
                return segments[left].speaker
            if right < len(segments) and segments[right].speaker:
                return segments[right].speaker
            left -= 1
            right += 1
        return None
