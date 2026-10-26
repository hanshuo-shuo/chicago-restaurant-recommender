"""Online inference: load versioned artifacts and serve constrained top-k recommendations.

Serving depends on NumPy (plus LightGBM when a learned ranker is deployed). Trained
two-tower weights are applied with a NumPy forward pass, and sentence-encoder item
vectors are loaded precomputed, so the API never imports PyTorch or the encoder.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import uuid

import numpy as np

from .config import LOCATION_PRESETS, MAX_K, MAX_LIST_PREFERENCES
from .data import public_quality
from .dataset import Dataset, temporal_split
from .features import ItemFeatures
from .geo import GeoEncoder, haversine_km, item_coordinates, user_location
from .ranker import ItemStats, LearnedRanker, candidates, features as ranker_features, source_scores
from .ranking import CatalogIndex, explain, ranking_scores
from .retrieval import content_scores, interaction_counts, load_two_tower, popularity_scores, two_tower_inputs, two_tower_scores

BASELINE_VERSION = "baseline_v1"
LLM_CANDIDATES = 20
PUBLIC_FIELDS = ("item_id", "name", "cuisine", "categories", "city", "latitude", "longitude", "average_cost",
                 "price_level", "open_after_8pm", "description", "closed", "metadata_status")


class Engine:
    def __init__(self, ds: Dataset, artifact_dir: str | Path, llm_reranker=None):
        self.ds = ds
        self.llm = llm_reranker
        self.artifact_dir = Path(artifact_dir)
        self.weights, self.ranker, self.report, self.geo = None, None, None, None
        self.model_version = BASELINE_VERSION
        config_file = self.artifact_dir / "model_config.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else None
        if config:
            text_matrix = None
            if config["features"]["text_backend"] == "sentence":
                text_matrix = np.load(self.artifact_dir / "text_matrix.npy")
            self.features = ItemFeatures(ds.items, config["features"], text_matrix)
            if config["item_ids"] != self.features.ids or config.get("prepared_sha256") != ds.fingerprint:
                raise ValueError("Prepared data does not match the trained artifacts. Run train again.")
            self.train_end = config["train_end_inclusive"]
            self.geo = GeoEncoder(config=config["geo"]) if config.get("geo") else None
            self.weights = load_two_tower(self.artifact_dir / "two_tower.npz")
            if (self.artifact_dir / "ranker.txt").exists():
                self.ranker = LearnedRanker.load(self.artifact_dir / "ranker.txt")
                self.sources = config["ranker_sources"]
            self.model_version = config["model_version"]
            self.default_k = config.get("k", 5)
        else:
            self.features = ItemFeatures(ds.items)
            try:
                self.train_end = temporal_split(ds.day)[0]
            except ValueError:
                self.train_end = int(ds.day.max()) if len(ds.day) else 0
            self.default_k = 5
        self.items = self.features.items
        self.coords = item_coordinates(self.items)
        self.catalog_index = CatalogIndex(self.items, self.coords)
        train = ds.day <= self.train_end
        self.popularity = popularity_scores(ds.n_items, ds.item[train], ds.rating[train])
        self.counts = interaction_counts(ds.n_items, ds.item[train])
        self.popular = np.log1p(self.counts) + 1e-3 * self.popularity
        self.stats = ItemStats.build(self.items, ds.item, ds.day, self.train_end, self.popularity, self.coords)
        report_path = self.artifact_dir / "offline_report.json"
        if report_path.exists():
            self.report = json.loads(report_path.read_text())

    @property
    def models(self) -> list[str]:
        return (["baseline"] + (["two_tower"] if self.weights else []) + (["learned"] if self.ranker else [])
                + (["llm"] if self.llm else []))

    def catalog(self) -> list[dict]:
        return [{key: item.get(key) for key in PUBLIC_FIELDS} for item in self.items]

    def info(self) -> dict:
        cuisines = Counter(str(item.get("cuisine")) for item in self.items if not item.get("closed"))
        return {"dataset": self.ds.name, "catalog_size": len(self.items), "models": self.models,
                "default_model": "learned" if self.ranker else "baseline", "model_version": self.model_version,
                "default_k": self.default_k, "cuisines": [c for c, _ in cuisines.most_common(36)],
                "has_price_level": any(item.get("price_level") for item in self.items),
                "has_budget": any(item.get("average_cost") for item in self.items),
                "locations": LOCATION_PRESETS.get(self.ds.name, [])}

    def search(self, text: str, limit: int = 12) -> list[dict]:
        needle = text.casefold().strip()
        if not needle:
            return []
        hits = [i for i, item in enumerate(self.items) if needle in item["name"].casefold()]
        hits.sort(key=lambda i: (not self.items[i]["name"].casefold().startswith(needle), -self.counts[i], self.items[i]["name"]))
        return [{key: self.items[i].get(key) for key in ("item_id", "name", "cuisine", "city")} for i in hits[:limit]]

    def metrics(self) -> dict:
        if self.report:
            return dict(self.report, quality=public_quality(self.report.get("quality", {})))
        return {"status": "not_trained", "quality": public_quality(self.ds.quality)}

    def recommend(self, preferences: dict | None = None, model: str = "baseline", k: int = 5) -> dict:
        if model not in ("baseline", "two_tower", "learned", "llm"):
            raise ValueError("Model must be baseline, two_tower, learned or llm.")
        if model == "llm":
            return self._llm_recommend(preferences or {}, k)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer between 1 and {MAX_K}.")
        preferences = {} if preferences is None else preferences
        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be an object.")
        for key in ("cuisines", "seed_item_ids"):
            value = preferences.get(key, [])
            if not isinstance(value, list) or len(value) > MAX_LIST_PREFERENCES or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{key} must be a list of at most {MAX_LIST_PREFERENCES} strings.")
        for key in ("open_after_8pm", "exclude_seen", "include_closed"):
            if key in preferences and not isinstance(preferences[key], bool):
                raise ValueError(key + " must be boolean.")
        if any(item_id not in self.features.index for item_id in preferences.get("seed_item_ids", [])):
            raise ValueError("A selected seed restaurant is not in the catalog.")
        for key in ("budget", "max_distance_km"):
            value = preferences.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be a positive finite number.")

        seeds = np.array([self.features.index[i] for i in preferences.get("seed_item_ids", [])], dtype=int)
        seed_ratings = np.full(len(seeds), 5)
        location = np.array([np.nan, np.nan])
        if preferences.get("location"):
            location = np.array([preferences["location"]["latitude"], preferences["location"]["longitude"]], dtype=float)
        elif len(seeds):
            location = user_location(seeds, seed_ratings, self.coords)
        query = self.features.query([], preferences)
        has_context = bool(np.linalg.norm(query) > 0)
        requested_model, fallback = model, None
        if model == "learned" and self.ranker is None:
            model, fallback = "two_tower", "Learned ranker not deployed; served the two-tower ranking instead."
        if model == "two_tower" and self.weights is None:
            model, fallback = "baseline", "Two-tower artifacts not found; served the content baseline instead."
        content = content_scores(self.features, query, self.popularity)
        two_tower = (two_tower_scores(self.weights, two_tower_inputs(query, location[None], self.geo))[0]
                     if self.weights else None)
        seen = set(preferences.get("seed_item_ids", []))
        eligible = self.catalog_index.mask(preferences, seen, location)
        if model == "learned":
            cand = candidates(source_scores(content, two_tower, self.popular, location, self.coords), self.sources, eligible)
            scores = np.full(len(self.items), -np.inf)
            if len(cand):
                scores[cand] = self.ranker.predict(ranker_features(cand, content, two_tower, seeds, seed_ratings, location, self.stats))
            retrieval = two_tower
        else:
            retrieval = two_tower if model == "two_tower" else content
            scores = ranking_scores(retrieval, self.popularity)
        order = [i for i in np.flatnonzero(eligible & np.isfinite(scores))]
        order.sort(key=lambda i: (-float(scores[i]), self.features.ids[i]))
        distances = (haversine_km(location[0], location[1], self.coords[:, 0], self.coords[:, 1])
                     if not np.isnan(location).any() else None)
        recommendations = []
        for i in order[:k]:
            item = {key: self.items[i].get(key) for key in PUBLIC_FIELDS}
            distance = None if distances is None or np.isnan(distances[i]) else round(float(distances[i]), 2)
            item.update(score=round(float(scores[i]), 6), ranking_score=round(float(scores[i]), 6),
                        retrieval_score=round(float(retrieval[i]), 6), distance_km=distance,
                        reasons=explain(item, preferences, has_context, distance))
            recommendations.append(item)
        return {"recommendations": recommendations, "request_id": str(uuid.uuid4()),
                "model_version": self.model_version,
                "model": model, "requested_model": requested_model, "candidate_count": int(eligible.sum()),
                "context": "preference_or_history" if has_context else "cold_start_global_prior",
                "location_source": "request" if preferences.get("location") else ("seeds" if len(seeds) and distances is not None else None),
                "fallback": fallback,
                "message": None if order else "No restaurant satisfies every constraint. Relax one and try again "
                                              "(items with unknown cost, price or hours never match those filters)."}

    def _llm_recommend(self, preferences: dict, k: int) -> dict:
        """Re-rank the best available arm's top candidates with the LLM; any failure keeps the base order."""
        base = "learned" if self.ranker else ("two_tower" if self.weights else "baseline")
        result = self.recommend(preferences, base, max(k, LLM_CANDIDATES))
        if self.llm is None:
            result.update(requested_model="llm", fallback=f"LLM reranker not enabled; served the {base} ranking.")
            result["recommendations"] = result["recommendations"][:k]
            return result
        stated = {key: preferences[key] for key in ("cuisines", "budget", "max_price_level", "open_after_8pm",
                                                   "max_distance_km") if preferences.get(key) not in (None, [], False)}
        liked = [self.items[self.features.index[i]]["name"] for i in preferences.get("seed_item_ids", [])]
        if liked:
            stated["liked_restaurants"] = liked
        reordered, error = self.llm.rerank(stated, result["recommendations"])
        for item in reordered:
            item["reasons"] = item["reasons"] + ["Re-ranked by an LLM against your stated preferences."]
        result.update(recommendations=reordered[:k], model="llm", requested_model="llm", fallback=error,
                      model_version=f"{self.model_version}+{self.llm.version}")
        return result
