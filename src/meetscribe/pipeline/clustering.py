"""Speaker clustering using Agglomerative Hierarchical Clustering."""

import logging
from typing import cast

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def cluster_embeddings(embeddings: list[list[float]], threshold: float) -> list[int]:
    """Cluster speaker embeddings using Agglomerative Hierarchical Clustering.

    Uses cosine distance with average linkage — the standard approach
    for speaker diarization (pyannote-audio, NeMo).

    Args:
        embeddings: List of embedding vectors.
        threshold: Cosine similarity threshold for merging clusters.
            Pairs with similarity >= threshold are grouped as same speaker.

    Returns:
        List of cluster IDs (0-indexed), same length as embeddings.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    X = np.array(embeddings, dtype=np.float64)

    # L2-normalize for cosine distance = 1 - dot(a, b)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X / norms

    # Cosine distance matrix
    similarity = X_norm @ X_norm.T
    distance = 1.0 - similarity
    np.clip(distance, 0.0, 2.0, out=distance)
    # Zero out diagonal (floating point artifacts)
    np.fill_diagonal(distance, 0.0)

    # Diagnostic: distance distribution (helps tune threshold)
    upper = distance[np.triu_indices(n, k=1)]
    logger.info(
        "Distance stats: min=%.3f, p5=%.3f, p25=%.3f, median=%.3f, p75=%.3f, p95=%.3f, max=%.3f",
        np.min(upper),
        np.percentile(upper, 5),
        np.percentile(upper, 25),
        np.median(upper),
        np.percentile(upper, 75),
        np.percentile(upper, 95),
        np.max(upper),
    )

    distance_threshold = 1.0 - threshold
    logger.info("Using distance_threshold=%.3f (similarity=%.3f)", distance_threshold, threshold)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distance)

    n_clusters = len(set(labels))
    logger.info(
        "AHC clustering: %d embeddings -> %d clusters (threshold=%.2f)",
        n,
        n_clusters,
        threshold,
    )

    return cast(list[int], labels.tolist())
