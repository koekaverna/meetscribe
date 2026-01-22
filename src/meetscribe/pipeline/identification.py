"""Speaker identification with voice enrollment."""

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingExtractor


@dataclass
class SpeakerMatch:
    """Speaker identification result."""

    name: str
    confidence: float
    is_known: bool


class SpeakerIdentifier:
    """Identify speakers using enrolled voiceprints."""

    DEFAULT_THRESHOLD = 0.7

    def __init__(
        self,
        voiceprints_path: Path,
        extractor: EmbeddingExtractor,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.voiceprints_path = voiceprints_path
        self.extractor = extractor
        self.threshold = threshold
        self.voiceprints: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if self.voiceprints_path.exists():
            with open(self.voiceprints_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for name, emb_list in data.items():
                    self.voiceprints[name] = np.array(emb_list)

    def _save(self) -> None:
        self.voiceprints_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: emb.tolist() for name, emb in self.voiceprints.items()}
        with open(self.voiceprints_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def enroll(self, name: str, audio_files: list[Path]) -> np.ndarray:
        """Enroll speaker from audio samples."""
        embeddings = [self.extractor.extract_from_file(f) for f in audio_files]
        voiceprint = np.mean(embeddings, axis=0)
        self.voiceprints[name] = voiceprint
        self._save()
        return voiceprint

    def remove(self, name: str) -> bool:
        """Remove speaker from database."""
        if name in self.voiceprints:
            del self.voiceprints[name]
            self._save()
            return True
        return False

    def list_speakers(self) -> list[str]:
        """List enrolled speakers."""
        return list(self.voiceprints.keys())

    def identify(self, embedding: np.ndarray) -> SpeakerMatch:
        """Identify speaker from embedding."""
        if not self.voiceprints:
            return SpeakerMatch(name="Unknown", confidence=0.0, is_known=False)

        best_match, best_score = None, -1.0
        for name, voiceprint in self.voiceprints.items():
            score = self.extractor.cosine_similarity(embedding, voiceprint)
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= self.threshold:
            return SpeakerMatch(name=best_match, confidence=best_score, is_known=True)
        return SpeakerMatch(name="Unknown", confidence=best_score, is_known=False)

    def identify_clusters(self, centroids: dict[int, np.ndarray]) -> dict[int, SpeakerMatch]:
        """Identify speakers for each cluster."""
        results = {}
        unknown_counter = 0

        for cluster_id, centroid in centroids.items():
            match = self.identify(centroid)
            if not match.is_known:
                unknown_counter += 1
                match = SpeakerMatch(f"Unknown {unknown_counter}", match.confidence, False)
            results[cluster_id] = match

        return results
