"""Geography: great-circle distance, smooth location features and a user's location from history."""
from __future__ import annotations

import numpy as np

from .cluster import kmeans
from .config import POSITIVE_RATING

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def item_coordinates(items: list[dict]) -> np.ndarray:
    return np.array([[it.get("latitude") if it.get("latitude") is not None else np.nan,
                      it.get("longitude") if it.get("longitude") is not None else np.nan] for it in items], dtype=float)


class GeoEncoder:
    """Gaussian RBF features over k-means anchors of item locations.

    A nearby user and restaurant get similar vectors, which lets a two-tower model
    learn locality (the idea behind geo-aware tokens in OneLoc) without IDs.
    """
    def __init__(self, coords: np.ndarray | None = None, n_anchors: int = 32, config: dict | None = None, seed: int = 0):
        if config is None:
            known = coords[~np.isnan(coords).any(1)]
            scale = np.cos(np.radians(known[:, 0].mean()))
            anchors, _ = kmeans(known * [1, scale], min(n_anchors, max(1, len(known) // 5)), seed=seed)
            anchors = anchors / [1, scale]
            if len(anchors) > 1:
                pairwise = haversine_km(anchors[:, None, 0], anchors[:, None, 1], anchors[None, :, 0], anchors[None, :, 1])
                np.fill_diagonal(pairwise, np.inf)
                bandwidth = float(np.median(pairwise.min(1)))
            else:
                bandwidth = 1.
            config = {"anchors": anchors.tolist(), "bandwidth_km": max(bandwidth, .25)}
        self.config = config
        self.anchors = np.array(config["anchors"], dtype=float)
        self.bandwidth = config["bandwidth_km"]

    @property
    def dim(self) -> int:
        return len(self.anchors)

    def encode(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        out = np.zeros((len(coords), self.dim))
        known = ~np.isnan(coords).any(1)
        if known.any():
            d = haversine_km(coords[known, 0, None], coords[known, 1, None], self.anchors[None, :, 0], self.anchors[None, :, 1])
            rbf = np.exp(-.5 * (d / self.bandwidth) ** 2)
            out[known] = rbf / np.maximum(np.linalg.norm(rbf, axis=1, keepdims=True), 1e-12)
        return out


def user_location(history_items: np.ndarray, history_ratings: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Median location of liked restaurants (all visited ones if none were liked); NaN without history."""
    liked = history_items[history_ratings >= POSITIVE_RATING]
    chosen = coords[liked if len(liked) else history_items]
    chosen = chosen[~np.isnan(chosen).any(1)] if len(chosen) else chosen
    return np.median(chosen, axis=0) if len(chosen) else np.array([np.nan, np.nan])
