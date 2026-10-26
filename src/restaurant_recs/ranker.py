"""Learned second-stage ranker: LightGBM over candidates from several retrievers.

Uses LightGBM (Ke et al., NeurIPS 2017). The objective is chosen on validation among
LambdaRank (Burges, MSR-TR-2010-82), pointwise binary logistic and XE-NDCG (Bruch, WWW 2021).

The retrievers are frozen at the training cutoff; the ranker is fitted on
validation-period queries and evaluated once on test, so no stage sees test labels.
Item statistics come from training interactions only. The catalog's 2021
"permanently closed" flag is deliberately not a feature (it would leak the future
into 2016-2019 predictions); it is only a serving-time filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import POSITIVE_RATING
from .geo import haversine_km

FEATURES = ("content_score", "content_rank", "two_tower_score", "two_tower_rank", "log_train_count",
            "log_recent_count", "bayesian_rating", "distance_km", "cuisine_affinity", "price_level", "price_gap",
            "open_late", "log_history_length", "user_mean_rating")
GEO_FEATURES = ("distance_km",)


@dataclass
class ItemStats:
    """Training-period item statistics and static metadata used as ranker features."""
    log_count: np.ndarray
    log_recent: np.ndarray
    bayesian: np.ndarray
    price: np.ndarray
    open_late: np.ndarray
    cuisine: np.ndarray
    coords: np.ndarray

    @classmethod
    def build(cls, items: list[dict], item: np.ndarray, day: np.ndarray, train_end: int, bayesian: np.ndarray,
              coords: np.ndarray, recent_days: int = 90) -> "ItemStats":
        n = len(items)
        train = day <= train_end
        counts = np.bincount(item[train], minlength=n)
        recent = np.bincount(item[train & (day > train_end - recent_days)], minlength=n)
        cuisines = {c: i for i, c in enumerate(sorted({str(it.get("cuisine")) for it in items}))}
        return cls(np.log1p(counts), np.log1p(recent), bayesian,
                   np.array([it.get("price_level") or np.nan for it in items], dtype=float),
                   np.array([np.nan if it.get("open_after_8pm") is None else float(it["open_after_8pm"]) for it in items]),
                   np.array([cuisines[str(it.get("cuisine"))] for it in items]), coords)


def source_scores(content: np.ndarray, two_tower: np.ndarray, popular: np.ndarray, location: np.ndarray,
                  coords: np.ndarray) -> dict[str, np.ndarray]:
    """Per-source retrieval scores; ``nearby`` (closest restaurants first) exists only when a location is known."""
    sources = {"two_tower": two_tower, "content": content, "popular": popular}
    if not np.isnan(location).any():
        distance = haversine_km(location[0], location[1], coords[:, 0], coords[:, 1])
        sources["nearby"] = np.where(np.isnan(distance), -np.inf, -distance)
    return sources


def candidates(sources: dict[str, np.ndarray], sizes: dict[str, int], eligible: np.ndarray | None = None) -> np.ndarray:
    """Union of each source's top-n eligible items, as sorted unique indices."""
    pools = []
    for name, scores in sources.items():
        n = min(sizes.get(name, 0), len(scores))
        if n == 0:
            continue
        scores = scores if eligible is None else np.where(eligible, scores, -np.inf)
        top = np.argpartition(-scores, n - 1)[:n]
        pools.append(top[np.isfinite(scores[top])])
    return np.unique(np.concatenate(pools)) if pools else np.zeros(0, dtype=np.int64)


def _ranks(scores: np.ndarray, index: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores))
    rank[order] = np.arange(1, len(scores) + 1)
    return np.log1p(rank[index])


def features(cand: np.ndarray, content: np.ndarray, two_tower: np.ndarray, history_items: np.ndarray,
             history_ratings: np.ndarray, location: np.ndarray, stats: ItemStats) -> np.ndarray:
    liked = history_items[history_ratings >= POSITIVE_RATING]
    n = len(cand)
    if np.isnan(location).any():
        distance = np.full(n, np.nan)
    else:
        distance = haversine_km(location[0], location[1], stats.coords[cand, 0], stats.coords[cand, 1])
    affinity = (np.mean(stats.cuisine[liked][None, :] == stats.cuisine[cand][:, None], axis=1)
                if len(liked) else np.zeros(n))
    liked_price = stats.price[liked]
    liked_price = liked_price[~np.isnan(liked_price)]
    price_gap = np.abs(stats.price[cand] - liked_price.mean()) if len(liked_price) else np.full(n, np.nan)
    columns = [content[cand], _ranks(content, cand), two_tower[cand], _ranks(two_tower, cand),
               stats.log_count[cand], stats.log_recent[cand], stats.bayesian[cand], distance, affinity,
               stats.price[cand], price_gap, stats.open_late[cand], np.full(n, np.log1p(len(history_items))),
               np.full(n, history_ratings.mean() if len(history_ratings) else np.nan)]
    return np.column_stack(columns)


OBJECTIVES = {
    # Same trees for every objective; only the loss (and its early-stopping metric) changes.
    "lambdarank": {"objective": "lambdarank", "metric": "ndcg", "min_data_in_leaf": 50, "lambdarank_truncation_level": 30},
    "binary": {"objective": "binary", "metric": "binary_logloss", "min_data_in_leaf": 200},
    "rank_xendcg": {"objective": "rank_xendcg", "metric": "ndcg", "min_data_in_leaf": 200},
}


class LearnedRanker:
    """LightGBM over candidate features. The objective is a hyperparameter chosen on validation: with one or
    two positives among ~200 candidates, pairwise LambdaRank gradients are sparse and noisy, and a pointwise
    logistic loss can rank better."""
    def __init__(self, booster=None, feature_names: tuple[str, ...] = FEATURES, objective: str = "lambdarank"):
        self.booster = booster
        self.objective = objective
        self.feature_names = tuple(feature_names)
        self.columns = [FEATURES.index(f) for f in self.feature_names]

    def fit(self, train: tuple, valid: tuple, k: int, rounds: int = 2000) -> dict:
        import lightgbm as lgb
        params = {"eval_at": [k], "learning_rate": .05, "num_leaves": 31, "feature_fraction": .9, "bagging_fraction": .8,
                  "bagging_freq": 1, "verbosity": -1, "seed": 0, "deterministic": True, "num_threads": 4,
                  **OBJECTIVES[self.objective]}
        (x, y, groups), (xv, yv, groups_v) = train, valid
        train_set = lgb.Dataset(x[:, self.columns], y, group=groups, feature_name=list(self.feature_names))
        valid_set = lgb.Dataset(xv[:, self.columns], yv, group=groups_v, reference=train_set)
        self.booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
                                 callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        gain = self.booster.feature_importance("gain")
        return {"objective": self.objective, "best_iteration": int(self.booster.best_iteration),
                "early_stopping_metric": {name: round(float(value), 6) for name, value in self.booster.best_score["valid_0"].items()},
                "feature_gain_share": {f: round(float(g / gain.sum()), 4) for f, g in
                                       sorted(zip(self.feature_names, gain), key=lambda p: -p[1])}}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x[:, self.columns], num_iteration=self.booster.best_iteration or None)

    def save(self, path) -> None:
        self.booster.save_model(str(path), num_iteration=self.booster.best_iteration or None)
        (path.parent / (path.stem + "_features.txt")).write_text("\n".join(self.feature_names) + "\n")

    @classmethod
    def load(cls, path) -> "LearnedRanker":
        import lightgbm as lgb
        names = tuple((path.parent / (path.stem + "_features.txt")).read_text().split())
        return cls(lgb.Booster(model_file=str(path)), names)
