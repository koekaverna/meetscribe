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
        voiceprints_dir: Path,
        extractor: EmbeddingExtractor,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.voiceprints_dir = voiceprints_dir
        self.extractor = extractor
        self.threshold = threshold
        self.voiceprints: dict[str, np.ndarray] = {}
        self._load()

    def _speaker_file(self, name: str) -> Path:
        """Get path to speaker's voiceprint file."""
        return self.voiceprints_dir / f"{name}.json"

    def _load(self) -> None:
        """Load all voiceprints from individual speaker files."""
        if not self.voiceprints_dir.exists():
            return
        for filepath in self.voiceprints_dir.glob("*.json"):
            name = filepath.stem
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.voiceprints[name] = np.array(data["embedding"])

    def _save_speaker(self, name: str) -> None:
        """Save a single speaker's voiceprint to their file."""
        self.voiceprints_dir.mkdir(parents=True, exist_ok=True)
        data = {"embedding": self.voiceprints[name].tolist()}
        with open(self._speaker_file(name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def enroll(self, name: str, audio_files: list[Path]) -> np.ndarray:
        """Enroll speaker from audio samples.

        Combines provided audio_files with any existing enrolled samples
        from ENROLLED_SAMPLES_DIR/<name>/.
        """
        from meetscribe.config import ENROLLED_SAMPLES_DIR

        # Collect existing enrolled samples
        enrolled_dir = ENROLLED_SAMPLES_DIR / name
        existing = list(enrolled_dir.glob("*.wav")) if enrolled_dir.exists() else []

        # Combine existing + new
        all_samples = existing + list(audio_files)
        if not all_samples:
            raise ValueError(f"No samples for {name}")

        embeddings = [self.extractor.extract_from_file(f) for f in all_samples]
        voiceprint = np.mean(embeddings, axis=0)
        self.voiceprints[name] = voiceprint
        self._save_speaker(name)
        return voiceprint

    def remove(self, name: str) -> bool:
        """Remove speaker from database."""
        if name in self.voiceprints:
            del self.voiceprints[name]
            speaker_file = self._speaker_file(name)
            if speaker_file.exists():
                speaker_file.unlink()
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
