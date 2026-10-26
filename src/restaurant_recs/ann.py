"""Inverted-file (IVF) approximate nearest-neighbour index for maximum-inner-product retrieval.

The two-tower score ``<q, v>/T + b`` is an exact inner product between [q, 1] and
[v/T, b], so the index searches augmented vectors and returns the same scores as
exact scoring for every item it visits. The inverted file is the coarse quantizer of
IVF-ADC (Jégou, Douze & Schmid, IEEE TPAMI 2011).
"""
from __future__ import annotations

import time

import numpy as np

from .cluster import kmeans


class IVFIndex:
    def __init__(self, item_embeddings: np.ndarray, item_bias: np.ndarray, temperature: float,
                 n_lists: int | None = None, seed: int = 0):
        self.vectors = np.concatenate([item_embeddings / temperature, item_bias[:, None]], axis=1)
        n_lists = n_lists or max(1, int(round(np.sqrt(len(self.vectors)))))
        self.centroids, assignment = kmeans(item_embeddings, n_lists, seed=seed)
        self.lists = [np.flatnonzero(assignment == c) for c in range(len(self.centroids))]

    def search(self, queries: np.ndarray, k: int, n_probe: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Top-k item indices per query (normalized query embeddings) and items scanned per query."""
        probes = np.argsort(-(queries @ self.centroids.T), axis=1)[:, :n_probe]
        results, scanned = [], np.zeros(len(queries), dtype=int)
        for row, lists in enumerate(probes):
            candidates = np.concatenate([self.lists[c] for c in lists])
            scores = self.vectors[candidates] @ np.append(queries[row], 1.)
            results.append(candidates[np.argsort(-scores, kind="stable")[:k]])
            scanned[row] = len(candidates)
        return results, scanned


def benchmark(index: IVFIndex, queries: np.ndarray, k: int, n_probes: tuple[int, ...]) -> list[dict]:
    """Recall of IVF top-k against exact top-k, items scanned, and per-query latency."""
    k = min(k, len(index.vectors))
    started = time.perf_counter()
    exact = [np.argsort(-(index.vectors @ np.append(q, 1.)), kind="stable")[:k] for q in queries]
    exact_ms = (time.perf_counter() - started) * 1000 / len(queries)
    rows = [{"method": "exact", "recall_vs_exact": 1.0, "items_scanned": len(index.vectors),
             "ms_per_query": round(exact_ms, 4)}]
    for n_probe in n_probes:
        started = time.perf_counter()
        found, scanned = index.search(queries, k, n_probe)
        ms = (time.perf_counter() - started) * 1000 / len(queries)
        recall = np.mean([len(np.intersect1d(f, e)) / k for f, e in zip(found, exact)])
        rows.append({"method": f"ivf nprobe={n_probe}/{len(index.lists)}", "recall_vs_exact": round(float(recall), 4),
                     "items_scanned": int(scanned.mean()), "ms_per_query": round(ms, 4)})
    return rows
