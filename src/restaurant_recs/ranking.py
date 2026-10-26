"""Second stage: blend retrieval with the popularity prior, then apply hard constraints."""
from __future__ import annotations

import numpy as np

from .config import POPULARITY_WEIGHT, RETRIEVAL_WEIGHT
from .geo import haversine_km

RANKING_POLICY = {
    "retrieval_weight": RETRIEVAL_WEIGHT, "train_only_bayesian_rating_weight": POPULARITY_WEIGHT,
    "normalization": "min-max over the full catalog per request",
    "coefficient_source": "fixed before test evaluation; not tuned on the test split",
    "retrieval_pool": "exact scores for the full catalog",
    "score_semantics": "relative priorities within one model, not probabilities or predicted ratings",
}


def _minmax(values: np.ndarray) -> np.ndarray:
    low = np.min(values, axis=-1, keepdims=True)
    span = np.max(values, axis=-1, keepdims=True) - low
    return np.where(span > 1e-12, (values - low) / np.where(span > 1e-12, span, 1), 0.)


def ranking_scores(retrieval: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Row-wise blend; ``retrieval`` may be one request (1-D) or a batch (2-D)."""
    return RETRIEVAL_WEIGHT * _minmax(retrieval) + POPULARITY_WEIGHT * _minmax(popularity)


class CatalogIndex:
    """Catalog metadata as arrays, built once, so hard filters are vectorized per request."""
    def __init__(self, items: list[dict], coords: np.ndarray | None = None):
        self.ids = np.array([item["item_id"] for item in items])
        self.cuisine = np.array([str(item.get("cuisine") or "").casefold() for item in items])
        self.cost = np.array([np.nan if item.get("average_cost") is None else float(item["average_cost"]) for item in items])
        self.price = np.array([np.nan if item.get("price_level") is None else float(item["price_level"]) for item in items])
        self.late = np.array([item.get("open_after_8pm") is True for item in items])
        self.closed = np.array([bool(item.get("closed")) for item in items])
        self.coords = coords

    def mask(self, preferences: dict, seen: set[str], location: np.ndarray | None = None) -> np.ndarray:
        """Hard filters. Unknown metadata never satisfies a constraint, and nothing is silently relaxed."""
        mask = np.ones(len(self.ids), dtype=bool)
        if preferences.get("cuisines"):
            mask &= np.isin(self.cuisine, [str(c).casefold() for c in preferences["cuisines"]])
        if preferences.get("budget") is not None:
            mask &= ~np.isnan(self.cost) & (np.nan_to_num(self.cost, nan=np.inf) <= preferences["budget"])
        if preferences.get("max_price_level") is not None:
            mask &= ~np.isnan(self.price) & (np.nan_to_num(self.price, nan=np.inf) <= preferences["max_price_level"])
        if preferences.get("open_after_8pm"):
            mask &= self.late
        if not preferences.get("include_closed"):
            mask &= ~self.closed
        if preferences.get("exclude_seen") and seen:
            mask &= ~np.isin(self.ids, list(seen))
        if preferences.get("max_distance_km") is not None:
            if location is None or np.isnan(location).any():
                raise ValueError("max_distance_km needs a location.")
            distance = haversine_km(location[0], location[1], self.coords[:, 0], self.coords[:, 1])
            mask &= ~np.isnan(distance) & (np.nan_to_num(distance, nan=np.inf) <= preferences["max_distance_km"])
        return mask


def eligible_mask(items: list[dict], preferences: dict, seen: set[str], location: np.ndarray | None = None,
                  coords: np.ndarray | None = None) -> np.ndarray:
    return CatalogIndex(items, coords).mask(preferences, seen, location)


def eligible_indices(items: list[dict], preferences: dict, seen: set[str]) -> list[int]:
    return np.flatnonzero(eligible_mask(items, preferences, seen)).tolist()


def explain(item: dict, preferences: dict, has_context: bool, distance_km: float | None = None) -> list[str]:
    reasons = []
    if preferences.get("cuisines"):
        reasons.append("Matches your selected cuisine.")
    if preferences.get("budget") is not None:
        reasons.append("Average cost fits your budget.")
    if preferences.get("max_price_level") is not None:
        reasons.append("Price level fits your limit.")
    if preferences.get("open_after_8pm"):
        reasons.append("Open after 8 pm.")
    if distance_km is not None:
        reasons.append(f"{distance_km:.1f} km from your location.")
    if has_context:
        reasons.append("Similar to the restaurants and cuisines you picked.")
    else:
        reasons.append("Popular overall; no preferences were given.")
    if item.get("metadata_status") == "review_only":
        reasons.append("Catalog details are missing for this restaurant.")
    return reasons
