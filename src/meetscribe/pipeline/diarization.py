"""Speaker diarization using Spectral Clustering."""

from dataclasses import dataclass

import numpy as np
from speechbrain.processing.diarization import Spec_Clust_unorm


@dataclass
class DiarizedSegment:
    """Speech segment with cluster assignment."""

    start_ms: int
    end_ms: int
    cluster_id: int
    embedding: np.ndarray


class SpectralClusterer:
    """Speaker clustering using spectral clustering."""

    def __init__(self, min_speakers: int = 2, max_speakers: int = 10):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def cluster(
        self,
        embeddings: list[np.ndarray],
        segments: list[tuple[int, int]],
        num_speakers: int | None = None,
    ) -> list[DiarizedSegment]:
        """Cluster embeddings into speaker groups."""
        if len(embeddings) == 0:
            return []

        if len(embeddings) == 1:
            return [
                DiarizedSegment(
                    start_ms=segments[0][0],
                    end_ms=segments[0][1],
                    cluster_id=0,
                    embedding=embeddings[0],
                )
            ]

        emb_matrix = np.stack(embeddings)
        clusterer = Spec_Clust_unorm(
            min_num_spkrs=self.min_speakers, max_num_spkrs=self.max_speakers
        )

        k = num_speakers if num_speakers is not None else self.max_speakers
        clusterer.cluster_embs(emb_matrix, k)
        labels = clusterer.labels_

        results = []
        for i, (emb, (start, end)) in enumerate(zip(embeddings, segments)):
            results.append(
                DiarizedSegment(
                    start_ms=start, end_ms=end, cluster_id=int(labels[i]), embedding=emb
                )
            )

        return results

    def _cosine_affinity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity affinity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        affinity = np.dot(normalized, normalized.T)
        return (affinity + 1) / 2

    def get_cluster_centroids(self, segments: list[DiarizedSegment]) -> dict[int, np.ndarray]:
        """Compute centroid embedding for each cluster."""
        clusters: dict[int, list[np.ndarray]] = {}
        for seg in segments:
            if seg.cluster_id not in clusters:
                clusters[seg.cluster_id] = []
            clusters[seg.cluster_id].append(seg.embedding)

        return {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
