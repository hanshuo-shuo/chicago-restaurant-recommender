"""Semantic IDs by residual k-means (RQ-Kmeans), the tokenizer of the OneRec Technical Report
(Zhou et al., arXiv:2506.13695).

Each item gets ``levels`` codes. Level l clusters the residual left after levels
< l, so items sharing a code prefix are similar at that granularity. Prefix
embeddings then let rare items borrow strength from similar, popular ones
(Singh et al., RecSys 2024), unlike random per-item IDs.
"""
from __future__ import annotations

import numpy as np

from .cluster import kmeans


def rq_kmeans(x: np.ndarray, levels: int = 3, codebook_size: int = 32, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Return (codes [n_items x levels], stats)."""
    residual = np.asarray(x, dtype=np.float64).copy()
    total = float((residual ** 2).sum())
    codes, utilization, error = [], [], []
    for level in range(levels):
        centroids, assignment = kmeans(residual, codebook_size, seed=seed + level)
        codes.append(assignment)
        residual -= centroids[assignment]
        utilization.append(round(len(np.unique(assignment)) / codebook_size, 4))
        error.append(round(float((residual ** 2).sum()) / total, 4))
    codes = np.stack(codes, axis=1)
    _, counts = np.unique(codes, axis=0, return_counts=True)
    stats = {"levels": levels, "codebook_size": codebook_size, "codebook_utilization": utilization,
             "relative_reconstruction_error": error, "unique_ids": int(len(counts)),
             "items": int(len(codes)), "max_items_per_id": int(counts.max())}
    return codes, stats


def prefix_ids(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """Dense id of each item's length-1, 2, ... code prefix; column l indexes a table with one row per
    prefix that actually occurs (``prefixes[:, l].max() + 1`` rows), not K**(l+1)."""
    prefixes = np.zeros_like(codes, dtype=np.int64)
    running = np.zeros(len(codes), dtype=np.int64)
    for level in range(codes.shape[1]):
        running = running * codebook_size + codes[:, level]
        prefixes[:, level] = np.unique(running, return_inverse=True)[1]
    return prefixes
