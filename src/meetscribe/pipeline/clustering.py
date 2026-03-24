"""Speaker clustering using Agglomerative Hierarchical Clustering."""

import logging
import time
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

    t0 = time.perf_counter()

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
    logger.debug(
        "Distance distribution",
        extra={
            "min": round(float(np.min(upper)), 3),
            "p5": round(float(np.percentile(upper, 5)), 3),
            "p25": round(float(np.percentile(upper, 25)), 3),
            "median": round(float(np.median(upper)), 3),
            "p75": round(float(np.percentile(upper, 75)), 3),
            "p95": round(float(np.percentile(upper, 95)), 3),
            "max": round(float(np.max(upper)), 3),
        },
    )

    distance_threshold = 1.0 - threshold

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distance)

    n_clusters = len(set(labels))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Clustering completed",
        extra={
            "embeddings": n,
            "clusters": n_clusters,
            "threshold": threshold,
            "elapsed_ms": round(elapsed_ms),
        },
    )

    return cast(list[int], labels.tolist())
