"""Seeded k-means (k-means++ init, Lloyd iterations), shared by geo anchors, Semantic IDs and the ANN index.

k-means++ seeding: Arthur & Vassilvitskii (SODA 2007). Lloyd iterations: Lloyd (IEEE Trans. Inf. Theory 1982).
"""
from __future__ import annotations

import numpy as np


def _sq_distances(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.maximum((x * x).sum(1)[:, None] - 2 * x @ centroids.T + (centroids * centroids).sum(1)[None], 0.)


def kmeans(x: np.ndarray, k: int, iterations: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids, assignment). Empty clusters are re-seeded at the worst-fit points."""
    x = np.asarray(x, dtype=np.float64)
    k = min(k, len(x))
    rng = np.random.default_rng(seed)
    centroids = [x[rng.integers(len(x))]]
    closest = _sq_distances(x, np.array(centroids))[:, 0]
    for _ in range(1, k):
        probabilities = closest / closest.sum() if closest.sum() > 0 else np.full(len(x), 1 / len(x))
        centroids.append(x[rng.choice(len(x), p=probabilities)])
        closest = np.minimum(closest, _sq_distances(x, centroids[-1][None])[:, 0])
    centroids = np.array(centroids)
    assignment = np.zeros(len(x), dtype=np.int64)
    for _ in range(iterations):
        distances = _sq_distances(x, centroids)
        new_assignment = distances.argmin(1)
        counts = np.bincount(new_assignment, minlength=k)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            worst = np.argsort(-distances[np.arange(len(x)), new_assignment])[:len(empty)]
            new_assignment[worst] = empty
            counts = np.bincount(new_assignment, minlength=k)
        sums = np.zeros_like(centroids)
        np.add.at(sums, new_assignment, x)
        centroids = sums / counts[:, None]
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
    return centroids, assignment
