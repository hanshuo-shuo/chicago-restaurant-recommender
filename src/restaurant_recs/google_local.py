"""Ingest Google Local (2021) ratings for Chicago and Evanston restaurants.

Source: UCSD McAuley Lab, Google Local Data (2021), a public academic release
that asks users to cite Li et al. (ACL 2022) and Yan et al. (SIGIR 2023). The raw
files are downloaded by ``scripts/download_google_local.sh`` and never committed.

Output (``data/processed/chicago/``):
  items.json          catalog metadata, sorted by item_id
  interactions.npz    user, item, rating, day (days since 1970-01-01); users are
                      anonymous integer indices, raw Google account IDs are dropped
  data_quality.json   filtering funnel and audit counts
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .data import _write_json

CITY_PATTERN = re.compile(r",\s*(Chicago|Evanston),\s*IL\s+\d{5}")
FOOD_PATTERN = re.compile(
    r"restaurant|\bcafe\b|café|coffee|bakery|pizza|\bdeli\b|bistro|diner|steak|sushi|taqueria|brewpub|gastropub|"
    r"barbecue|\bbbq\b|bagel|donut|dessert|ice cream|tea house|bubble tea|sandwich|burger|noodle|ramen|brunch|"
    r"breakfast|food court|bar & grill|creperie|juice", re.I)
# Categories too generic to describe a cuisine; the first remaining category is the primary cuisine.
GENERIC_CATEGORIES = {"Restaurant", "Takeout Restaurant", "Delivery Restaurant", "Bar", "Caterer", "Lunch restaurant",
                      "Family restaurant", "Fast food restaurant", "Dinner theater", "Event venue", "Banquet hall"}
TEXT_ATTRIBUTES = ("Offerings", "Atmosphere", "Popular for", "Dining options", "Highlights", "Crowd", "Planning",
                   "Service options")
EPOCH = np.datetime64("1970-01-01", "D")


def _closing_minutes(span: str) -> int | None:
    """Minutes after midnight at which the last shift of a day ends; past-midnight closes add 24h."""
    if span == "Open 24 hours":
        return 48 * 60
    if "–" not in span:
        return None
    end = span.split(",")[-1].split("–")[-1].strip().replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(AM|PM)", end)
    if not match:
        return None
    hour, minute = int(match.group(1)) % 12, int(match.group(2) or 0)
    minutes = hour * 60 + minute + (720 if match.group(3) == "PM" else 0)
    return minutes + 24 * 60 if match.group(3) == "AM" and hour < 6 else minutes


def _open_after_8pm(hours: list | None) -> bool | None:
    closes = [_closing_minutes(span) for _, span in hours or []]
    closes = [c for c in closes if c is not None]
    return any(c > 20 * 60 for c in closes) if closes else None


def _price_level(price: str | None) -> int | None:
    return len(price) if price and set(price) <= {"$", "₩"} and len(price) <= 4 else None


def _item(meta: dict, city: str) -> dict[str, Any]:
    categories = list(dict.fromkeys(meta.get("category") or []))
    specific = [c for c in categories if c not in GENERIC_CATEGORIES]
    misc = meta.get("MISC") or {}
    attributes = sorted({value for key in TEXT_ATTRIBUTES for value in misc.get(key, [])})
    return {"item_id": meta["gmap_id"], "name": meta.get("name") or "", "city": city,
            "cuisine": (specific or categories or ["Unknown"])[0], "categories": categories,
            "latitude": meta.get("latitude"), "longitude": meta.get("longitude"),
            "average_cost": None, "price_level": _price_level(meta.get("price")),
            "open_after_8pm": _open_after_8pm(meta.get("hours")),
            "description": meta.get("description") or "", "attributes": attributes,
            "closed": (meta.get("state") or "").startswith("Permanently closed"), "metadata_status": "catalog"}


def _k_core(user: np.ndarray, item: np.ndarray, min_user: int, min_item: int) -> np.ndarray:
    keep = np.ones(len(user), dtype=bool)
    while True:
        item_counts = np.bincount(item[keep], minlength=item.max() + 1)
        user_counts = np.bincount(user[keep], minlength=user.max() + 1)
        new_keep = keep & (item_counts[item] >= min_item) & (user_counts[user] >= min_user)
        if new_keep.sum() == keep.sum():
            return keep
        keep = new_keep


def prepare_google_local(raw_dir: str | Path, output_dir: str | Path, start: str = "2016-01-01",
                         end: str = "2019-12-31", min_user: int = 5, min_item: int = 10) -> dict[str, Any]:
    """Filter to Chicago/Evanston restaurants in [start, end], dedupe, k-core, and write the dataset."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}
    meta_rows = duplicate_meta = 0
    with gzip.open(raw_dir / "meta-Illinois.json.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            meta_rows += 1
            meta = json.loads(line)
            city = CITY_PATTERN.search(meta.get("address") or "")
            if not city or not any(FOOD_PATTERN.search(c) for c in meta.get("category") or []):
                continue
            if meta.get("latitude") is None or meta.get("longitude") is None:
                continue
            if meta["gmap_id"] in items:
                duplicate_meta += 1
                continue
            items[meta["gmap_id"]] = _item(meta, city.group(1))

    start_ms = int(np.datetime64(start, "ms").astype(np.int64))
    end_ms = int((np.datetime64(end, "D") + 1).astype("datetime64[ms]").astype(np.int64))
    rows: dict[tuple[str, str], tuple[int, int]] = {}
    raw_ratings = in_scope = repeated = 0
    with gzip.open(raw_dir / "rating-Illinois.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["business", "user", "rating", "timestamp"]:
            raise ValueError("Unexpected header in rating-Illinois.csv.gz")
        for business, user, rating, timestamp in reader:
            raw_ratings += 1
            if business not in items:
                continue
            ts = int(timestamp)
            if not start_ms <= ts < end_ms:
                continue
            in_scope += 1
            key = (user, business)
            if key in rows:
                repeated += 1
                if ts < rows[key][1]:
                    continue
            rows[key] = (int(rating), ts)  # one opinion per user and restaurant: the latest

    keys = list(rows)
    user_codes = {u: i for i, u in enumerate(sorted({u for u, _ in keys}))}
    item_codes = {b: i for i, b in enumerate(sorted({b for _, b in keys}))}
    user = np.fromiter((user_codes[u] for u, _ in keys), dtype=np.int32, count=len(keys))
    item = np.fromiter((item_codes[b] for _, b in keys), dtype=np.int32, count=len(keys))
    rating = np.fromiter((rows[k][0] for k in keys), dtype=np.int8, count=len(keys))
    day = np.fromiter((rows[k][1] // 86_400_000 for k in keys), dtype=np.int32, count=len(keys))
    del rows, keys
    if np.any((rating < 1) | (rating > 5)):
        raise ValueError("Ratings outside 1-5")
    keep = _k_core(user, item, min_user, min_item)

    item_kept_mask = np.bincount(item[keep], minlength=len(item_codes)) > 0
    old_to_business = sorted(item_codes, key=item_codes.get)
    catalog = sorted((items[b] for b, k in zip(old_to_business, item_kept_mask) if k), key=lambda x: x["item_id"])
    new_item_index = {it["item_id"]: i for i, it in enumerate(catalog)}
    remap_item = np.array([new_item_index.get(b, -1) for b in old_to_business], dtype=np.int32)
    user_kept = np.unique(user[keep])
    remap_user = np.full(len(user_codes), -1, dtype=np.int32)
    remap_user[user_kept] = np.arange(len(user_kept), dtype=np.int32)
    user, item, rating, day = remap_user[user[keep]], remap_item[item[keep]], rating[keep], day[keep]
    order = np.lexsort((item, user, day))
    user, item, rating, day = user[order], item[order], rating[order], day[order]

    user_counts = np.bincount(user)
    quality = {
        "source": "Google Local Data (2021), UCSD McAuley Lab; Chicago and Evanston restaurants",
        "window": {"start": start, "end": end}, "k_core": {"min_ratings_per_user": min_user, "min_ratings_per_item": min_item},
        "funnel": {"illinois_businesses": meta_rows, "duplicate_metadata_rows": duplicate_meta,
                   "chicago_evanston_restaurants": len(items), "illinois_ratings": raw_ratings,
                   "ratings_in_scope": in_scope, "repeat_user_restaurant_ratings_collapsed": repeated,
                   "ratings_after_k_core": int(len(user))},
        "catalog_items": len(catalog), "unique_users": int(len(user_kept)), "retained_interactions": int(len(user)),
        "items_by_city": dict(Counter(it["city"] for it in catalog)),
        "permanently_closed_items": sum(it["closed"] for it in catalog),
        "items_missing_price": sum(it["price_level"] is None for it in catalog),
        "items_missing_hours": sum(it["open_after_8pm"] is None for it in catalog),
        "items_with_description": sum(bool(it["description"]) for it in catalog),
        "rating_histogram": {str(r): int(n) for r, n in zip(*np.unique(rating, return_counts=True))},
        "interactions_per_user": {"median": float(np.median(user_counts)), "p90": float(np.percentile(user_counts, 90))},
        "matrix_observation_density": float(len(user) / (len(user_kept) * len(catalog))),
        "earliest_date": str(EPOCH + int(day.min())), "latest_date": str(EPOCH + int(day.max())),
        "excluded_metadata": "Google avg_rating and num_of_reviews (computed over all time, so they leak the future); "
                             "COVID-era health & safety, payment and accessibility attributes.",
    }
    _write_json(output_dir / "items.json", {"items": catalog})
    np.savez_compressed(output_dir / "interactions.npz", user=user, item=item, rating=rating, day=day)
    _write_json(output_dir / "data_quality.json", quality)
    return quality
