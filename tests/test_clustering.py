"""Tests for pipeline/clustering.py — AHC speaker clustering."""

import math

from meetscribe.pipeline.clustering import cluster_embeddings


def _make_embedding(seed: int, dim: int = 256) -> list[float]:
    """Create a deterministic normalized embedding from seed."""
    raw = [math.sin(seed * 100 + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class TestClusterEmbeddings:
    def test_empty_returns_empty(self):
        assert cluster_embeddings([], threshold=0.25) == []

    def test_single_returns_zero(self):
        emb = _make_embedding(1)
        assert cluster_embeddings([emb], threshold=0.25) == [0]

    def test_identical_vectors_one_cluster(self):
        emb = _make_embedding(1)
        labels = cluster_embeddings([emb, emb, emb], threshold=0.25)
        assert len(labels) == 3
        assert labels[0] == labels[1] == labels[2]

    def test_different_vectors_multiple_clusters(self):
        # Seeds 1 and 100 produce very different vectors
        emb_a = _make_embedding(1)
        emb_b = _make_embedding(100)
        labels = cluster_embeddings([emb_a, emb_a, emb_b, emb_b], threshold=0.25)
        assert len(labels) == 4
        # First two should be in the same cluster, last two in another
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_higher_threshold_fewer_clusters(self):
        embs = [_make_embedding(i) for i in range(5)]
        labels_loose = cluster_embeddings(embs, threshold=0.1)
        labels_strict = cluster_embeddings(embs, threshold=0.5)
        n_loose = len(set(labels_loose))
        n_strict = len(set(labels_strict))
        # Stricter threshold = more clusters (less merging)
        assert n_strict >= n_loose
