"""Leakage-safe offline evaluation: full-catalog ranking metrics in batches, user-cluster bootstrap.

NDCG follows Järvelin & Kekäläinen (ACM TOIS 2002). Scoring the full catalog instead of sampled
negatives follows Krichene & Rendle (KDD 2020). Intervals are a percentile bootstrap (Efron,
Annals of Statistics 1979) that resamples users rather than queries.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from .config import BOOTSTRAP_RESAMPLES, EVAL_K, SEED
from .dataset import Queries


def ranking_metrics(ranking: list[str], targets: list[str], k: int = EVAL_K) -> tuple[float, float]:
    relevant = set(targets)
    if not relevant:
        return 0., 0.
    hits = [int(item in relevant) for item in ranking[:k]]
    dcg = sum(hit / np.log2(position + 2) for position, hit in enumerate(hits))
    ideal = sum(1 / np.log2(position + 2) for position in range(min(k, len(relevant))))
    return float(sum(hits) / len(relevant)), float(dcg / ideal)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k indices, identical to ``argsort(-scores, kind="stable")[:, :k]`` (ties -> lower index)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    kth = np.take_along_axis(scores, part, 1).min(1)
    out = np.empty((len(scores), k), dtype=np.int64)
    for row in range(len(scores)):
        candidates = np.flatnonzero(scores[row] >= kth[row])
        out[row] = candidates[np.argsort(-scores[row, candidates], kind="stable")][:k]
    return out


def _cluster_interval(users: np.ndarray, values: np.ndarray, seed: int = SEED) -> list[float] | None:
    """Resample users rather than pretending multiple queries from one user are independent."""
    unique, inverse = np.unique(users, return_inverse=True)
    if len(unique) < 2:
        return None
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    indices = np.random.default_rng(seed).integers(0, len(unique), size=(BOOTSTRAP_RESAMPLES, len(unique)))
    samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return [round(float(v), 6) for v in np.quantile(samples, [.025, .975])]


def evaluate(queries: Queries, n_items: int, scorer: Callable[[Queries], np.ndarray], k: int = EVAL_K,
             uncertainty: bool = False, item_groups: dict[str, np.ndarray] | None = None,
             batch_size: int = 1024) -> dict:
    """Every policy is scored on the same queries, targets and full catalog."""
    recall, ndcg, shown = np.zeros(len(queries)), np.zeros(len(queries)), []
    group_hits = defaultdict(lambda: [0, 0])
    discounts = 1 / np.log2(np.arange(k) + 2)
    for start in range(0, len(queries), batch_size):
        rows = np.arange(start, min(start + batch_size, len(queries)))
        scores = np.asarray(scorer(queries.subset(rows)), dtype=float)
        if scores.shape != (len(rows), n_items) or not np.all(np.isfinite(scores)):
            raise ValueError("Scorer must return one finite score for every catalog item.")
        top = top_k(scores, k)
        shown.append(top)
        for j, q in enumerate(rows):
            targets = queries.targets[q]
            hit = np.isin(top[j], targets)
            recall[q] = hit.sum() / len(targets)
            ndcg[q] = (hit * discounts[:len(hit)]).sum() / discounts[:min(k, len(targets))].sum()
            for name, mask in (item_groups or {}).items():
                in_group = targets[mask[targets]]
                group_hits[name][0] += int(np.isin(in_group, top[j]).sum())
                group_hits[name][1] += len(in_group)
    shown = np.concatenate(shown) if shown else np.zeros((0, k), dtype=np.int64)
    warm, positive_history = queries.warm, queries.positive_history

    def aggregate(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        output = {"queries": n, "users": int(len(np.unique(queries.user[mask]))),
                  "positive_targets": int(sum(len(queries.targets[i]) for i in np.flatnonzero(mask))),
                  f"recall_at_{k}": round(float(recall[mask].mean()), 6) if n else None,
                  f"ndcg_at_{k}": round(float(ndcg[mask].mean()), 6) if n else None,
                  f"catalog_coverage_at_{k}": round(len(np.unique(shown[mask])) / n_items, 6) if n else None,
                  "distinct_recommended_items": int(len(np.unique(shown[mask]))), "catalog_size": n_items}
        if uncertainty and n:
            output["recall_95pct_user_bootstrap"] = _cluster_interval(queries.user[mask], recall[mask])
            output["ndcg_95pct_user_bootstrap"] = _cluster_interval(queries.user[mask], ndcg[mask])
        return output
    result = {"overall": aggregate(np.ones(len(queries), dtype=bool)), "warm": aggregate(warm),
              "cold": aggregate(~warm), "with_positive_history": aggregate(positive_history)}
    for name, (hits, total) in group_hits.items():
        result[f"recall_at_{k}_{name}_targets"] = {"targets": total, "recall": round(hits / total, 6) if total else None}
    return result
