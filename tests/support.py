"""Small synthetic datasets and fakes shared by the tests (no network, no real data)."""
from __future__ import annotations

import hashlib

import numpy as np

from restaurant_recs.dataset import Dataset, normalize_item

CUISINES = ("Italian", "Thai", "Mexican", "Coffee shop")


class FakeEncoder:
    """Deterministic stand-in for the sentence encoder: hashed bag-of-words, L2-normalized."""
    model_name = "fake-encoder"

    def encode(self, texts):
        out = np.zeros((len(texts), 32))
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 32] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def synthetic_items(n_items: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        cuisine = CUISINES[i % len(CUISINES)]
        items.append(normalize_item({
            "item_id": f"item-{i:03d}", "name": f"{cuisine} place {i}", "cuisine": cuisine,
            "categories": [f"{cuisine} restaurant" if cuisine != "Coffee shop" else cuisine, "Restaurant"],
            "latitude": 41.85 + .15 * rng.random(), "longitude": -87.70 + .10 * rng.random(),
            "price_level": int(rng.integers(1, 4)) if i % 5 else None, "open_after_8pm": bool(i % 3) if i % 7 else None,
            "description": f"A {cuisine.lower()} spot with {'cozy' if i % 2 else 'lively'} seating.",
            "attributes": ["Casual"] if i % 2 else ["Cozy"], "closed": i % 11 == 0, "city": "Chicago",
            "metadata_status": "catalog"}))
    return items


def synthetic_dataset(name: str = "chicago", n_users: int = 400, n_items: int = 40, days: int = 120,
                      per_user: int = 8, seed: int = 0) -> Dataset:
    """Users prefer one cuisine and restaurants near a home location, so models have signal to learn."""
    rng = np.random.default_rng(seed)
    items = synthetic_items(n_items, seed)
    coords = np.array([[it["latitude"], it["longitude"]] for it in items])
    cuisine = np.array([CUISINES.index(it["cuisine"]) for it in items])
    rows = []
    for user in range(n_users):
        favourite, home = rng.integers(len(CUISINES)), coords[rng.integers(n_items)]
        affinity = 2. * (cuisine == favourite) - 25 * np.linalg.norm(coords - home, axis=1)
        p = np.exp(affinity - affinity.max())
        chosen = rng.choice(n_items, size=min(per_user, n_items), replace=False, p=p / p.sum())
        start = rng.integers(0, days - per_user)
        for j, item in enumerate(chosen):
            rating = 5 if cuisine[item] == favourite else int(rng.integers(2, 5))
            rows.append((start + j * rng.integers(1, 3), user, item, rating))
    day, user, item, rating = (np.array(column) for column in zip(*sorted(rows)))
    return Dataset(name, items, user.astype(np.int32), item.astype(np.int32), rating.astype(np.int8),
                   (day + 18_000).astype(np.int32), {"synthetic": True}, f"synthetic-{seed}")
