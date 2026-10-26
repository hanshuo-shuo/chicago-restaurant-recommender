"""Candidate scorers. Inference is NumPy-only; PyTorch is needed only for training."""
from __future__ import annotations

import numpy as np

from .config import POPULARITY_PRIOR_STRENGTH
from .dataset import Queries
from .features import ItemFeatures, normalize
from .geo import GeoEncoder, user_location


def popularity_scores(n_items: int, item: np.ndarray, rating: np.ndarray,
                      strength: float = POPULARITY_PRIOR_STRENGTH) -> np.ndarray:
    """Bayesian average rating, shrunk toward the global train mean."""
    mean = float(rating.mean()) if len(rating) else 3.
    sums = np.bincount(item, weights=rating, minlength=n_items)
    counts = np.bincount(item, minlength=n_items)
    return (sums + strength * mean) / (counts + strength)


def interaction_counts(n_items: int, item: np.ndarray) -> np.ndarray:
    return np.bincount(item, minlength=n_items).astype(float)


def content_scores(features: ItemFeatures, query: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Cosine similarity to the query; falls back to popularity when there is no query."""
    # The popularity term is a deterministic tie-breaker, not a tuned weight.
    return features.matrix @ query + 1e-4 * (popularity - 3) if np.linalg.norm(query) > 0 else popularity.copy()


def content_queries(features: ItemFeatures, queries: Queries) -> np.ndarray:
    return np.array([features.query_from_indices(i, r) for i, r in zip(queries.history_items, queries.history_ratings)])


def content_batch(features: ItemFeatures, queries: Queries, popularity: np.ndarray) -> np.ndarray:
    vectors = content_queries(features, queries)
    scores = vectors @ features.matrix.T + 1e-4 * (popularity - 3)
    empty = np.linalg.norm(vectors, axis=1) == 0
    scores[empty] = popularity
    return scores


def query_locations(queries: Queries, coords: np.ndarray) -> np.ndarray:
    return np.array([user_location(i, r, coords) for i, r in zip(queries.history_items, queries.history_ratings)]).reshape(-1, 2)


def two_tower_inputs(content: np.ndarray, locations: np.ndarray, geo: GeoEncoder | None) -> np.ndarray:
    """Query-tower input: content-space preference vector, geo RBF of the user's location, and a bias."""
    parts = [np.atleast_2d(content)]
    if geo is not None:
        parts.append(geo.encode(locations))
    parts.append(np.ones((len(parts[0]), 1)))
    return np.concatenate(parts, axis=1)


def mlp(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for depth, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if depth < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def two_tower_scores(weights: dict, inputs: np.ndarray) -> np.ndarray:
    """NumPy forward pass matching ``two_tower.TwoTower``: cosine / temperature + per-item bias."""
    query = normalize(mlp(weights["query_layers"], np.atleast_2d(inputs)))
    return query @ weights["item_embeddings"].T / weights["temperature"] + weights["item_bias"]


def save_two_tower(path, weights: dict) -> None:
    arrays = {"item_embeddings": weights["item_embeddings"], "item_bias": weights["item_bias"],
              "temperature": np.array(weights["temperature"])}
    for i, (w, b) in enumerate(weights["query_layers"]):
        arrays[f"query_{i}_weight"], arrays[f"query_{i}_bias"] = w, b
    np.savez_compressed(path, **arrays)


def load_two_tower(path) -> dict:
    with np.load(path) as archive:
        layers, i = [], 0
        while f"query_{i}_weight" in archive:
            layers.append((archive[f"query_{i}_weight"].copy(), archive[f"query_{i}_bias"].copy()))
            i += 1
        return {"query_layers": layers, "item_embeddings": archive["item_embeddings"].copy(),
                "item_bias": archive["item_bias"].copy(), "temperature": float(archive["temperature"])}
