"""Columnar dataset shared by training, evaluation and serving, for every data source.

Interactions are parallel NumPy arrays sorted by (day, user, item); users are
anonymous integer indices and items index into ``items`` (sorted by item_id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import POSITIVE_RATING
from .data import fingerprint as prepared_fingerprint, load_prepared

EPOCH = np.datetime64("1970-01-01", "D")
DATASETS = ("evanston", "chicago")


@dataclass
class Dataset:
    name: str
    items: list[dict]
    user: np.ndarray
    item: np.ndarray
    rating: np.ndarray
    day: np.ndarray
    quality: dict
    fingerprint: str
    n_users: int = 0
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {it["item_id"]: i for i, it in enumerate(self.items)}
        self.n_users = int(self.user.max()) + 1 if len(self.user) else 0

    @property
    def n_items(self) -> int:
        return len(self.items)

    def date(self, day: int) -> str:
        return str(EPOCH + int(day))


def normalize_item(item: dict) -> dict:
    """Fill the fields that only one source provides, so downstream code sees one schema."""
    item = dict(item)
    item.setdefault("categories", [item["cuisine"]] if item.get("cuisine") else [])
    for key, default in (("price_level", None), ("attributes", []), ("closed", False), ("city", None),
                         ("average_cost", None), ("description", "")):
        item.setdefault(key, default)
    item["description"] = item["description"] or ""
    return item


def from_prepared(prepared: dict, name: str = "evanston") -> Dataset:
    """Convert the Evanston prepared.json; undated (quarantined) interactions are excluded."""
    items = sorted((normalize_item(it) for it in prepared["items"]), key=lambda it: it["item_id"])
    index = {it["item_id"]: i for i, it in enumerate(items)}
    events = [e for e in prepared["interactions"] if e.get("date")]
    users = {u: i for i, u in enumerate(sorted({e["user_id"] for e in events}))}
    day = np.array([(np.datetime64(e["date"], "D") - EPOCH).astype(int) for e in events], dtype=np.int32)
    user = np.array([users[e["user_id"]] for e in events], dtype=np.int32)
    item = np.array([index[e["item_id"]] for e in events], dtype=np.int32)
    rating = np.array([e["rating"] for e in events], dtype=np.int8)
    order = np.lexsort((item, user, day))
    return Dataset(name, items, user[order], item[order], rating[order], day[order], prepared["quality"],
                   prepared_fingerprint(prepared))


def load_dataset(name: str, data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    if name == "evanston":
        return from_prepared(load_prepared(data_dir / "prepared.json"))
    if name == "chicago":
        items_bytes = (data_dir / "items.json").read_bytes()
        arrays_bytes = (data_dir / "interactions.npz").read_bytes()
        items = [normalize_item(it) for it in json.loads(items_bytes)["items"]]
        with np.load(data_dir / "interactions.npz") as arrays:
            columns = {key: arrays[key] for key in ("user", "item", "rating", "day")}
        quality = json.loads((data_dir / "data_quality.json").read_text())
        digest = hashlib.sha256(items_bytes + arrays_bytes).hexdigest()
        return Dataset(name, items, quality=quality, fingerprint=digest, **columns)
    raise ValueError(f"Unknown dataset {name!r}; choose one of {DATASETS}.")


def temporal_split(day: np.ndarray, train: float = .8, validation: float = .9) -> tuple[int, int]:
    """Global split at event quantiles; whole days stay on one side of each boundary (Ji et al., ACM TOIS 2023)."""
    days = np.unique(day)
    if len(days) < 3:
        raise ValueError("At least three distinct valid dates are needed for chronological evaluation.")
    ordered = np.sort(day)
    train_end = int(ordered[max(0, int(len(ordered) * train) - 1)])
    val_end = int(ordered[max(0, int(len(ordered) * validation) - 1)])
    train_end = min(train_end, int(days[-3]))
    val_end = max(train_end, min(val_end, int(days[-2])))
    if val_end == train_end:
        val_end = int(days[np.searchsorted(days, train_end) + 1])
    return train_end, val_end


class Histories:
    """Per-user interaction histories (CSR), restricted to interactions on or before ``cutoff``."""
    def __init__(self, ds: Dataset, cutoff: int | None = None):
        keep = np.ones(len(ds.day), dtype=bool) if cutoff is None else ds.day <= cutoff
        user, item, day, rating = ds.user[keep], ds.item[keep], ds.day[keep], ds.rating[keep]
        order = np.lexsort((item, day, user))
        self.item, self.day, self.rating = item[order], day[order], rating[order]
        self.ptr = np.concatenate([[0], np.cumsum(np.bincount(user[order], minlength=ds.n_users))])

    def before(self, user: int, day: int) -> tuple[np.ndarray, np.ndarray]:
        """Items and ratings strictly before ``day``, oldest first."""
        lo, hi = self.ptr[user], self.ptr[user + 1]
        end = lo + np.searchsorted(self.day[lo:hi], day, side="left")
        return self.item[lo:end], self.rating[lo:end]


@dataclass
class Queries:
    """One query per (user, day) with at least one positive interaction in the evaluated period."""
    user: np.ndarray
    day: np.ndarray
    targets: list[np.ndarray]
    history_items: list[np.ndarray]
    history_ratings: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.user)

    @property
    def warm(self) -> np.ndarray:
        return np.array([len(h) > 0 for h in self.history_items])

    @property
    def positive_history(self) -> np.ndarray:
        return np.array([bool(np.any(r >= POSITIVE_RATING)) for r in self.history_ratings])

    def subset(self, rows: np.ndarray) -> "Queries":
        return Queries(self.user[rows], self.day[rows], [self.targets[i] for i in rows],
                       [self.history_items[i] for i in rows], [self.history_ratings[i] for i in rows])


def make_queries(ds: Dataset, start: int, end: int, histories: Histories, max_queries: int | None = None,
                 seed: int = 0) -> Queries:
    """Queries for positives with start < day <= end; history comes from ``histories`` before the query day.

    ``max_queries`` draws a seeded sample of whole users, so per-user clustering is preserved.
    """
    rows = np.flatnonzero((ds.day > start) & (ds.day <= end) & (ds.rating >= POSITIVE_RATING))
    keys = ds.user[rows].astype(np.int64) * 100_000 + ds.day[rows]
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    users, days = ds.user[rows][first], ds.day[rows][first]
    if max_queries is not None and len(unique) > max_queries:
        candidates, counts = np.unique(users, return_counts=True)
        order = np.random.default_rng(seed).permutation(len(candidates))
        taken = order[:np.searchsorted(np.cumsum(counts[order]), max_queries) + 1]
        selected = np.flatnonzero(np.isin(users, candidates[taken]))
    else:
        selected = np.arange(len(unique))
    targets_by_query = np.split(ds.item[rows][np.argsort(inverse, kind="stable")],
                                np.cumsum(np.bincount(inverse, minlength=len(unique)))[:-1])
    history = [histories.before(int(users[q]), int(days[q])) for q in selected]
    return Queries(users[selected], days[selected], [np.unique(targets_by_query[q]) for q in selected],
                   [h[0] for h in history], [h[1] for h in history])
