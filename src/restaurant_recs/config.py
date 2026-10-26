"""Constants shared by training, offline evaluation and serving."""
from __future__ import annotations

SEED = 42

# Labels and metrics
POSITIVE_RATING = 4          # rating >= 4 is a positive interaction
EVAL_K = 5                   # Recall@K / NDCG@K cutoff
BOOTSTRAP_RESAMPLES = 1000

# Popularity prior: Bayesian average with this many pseudo-ratings at the global mean
POPULARITY_PRIOR_STRENGTH = 5.

# Ranking: fixed blend of min-max normalized retrieval score and popularity prior.
# Chosen before looking at the test split; not tuned on it.
RETRIEVAL_WEIGHT = .85
POPULARITY_WEIGHT = .15

# Two-tower retrieval
TWO_TOWER_TEMPERATURE = .2
TWO_TOWER_SEARCH_SPACE = (
    {"embedding_dim": 8, "learning_rate": .01, "weight_decay": .001, "max_epochs": 80},
    {"embedding_dim": 8, "learning_rate": .01, "weight_decay": .01, "max_epochs": 80},
    {"embedding_dim": 4, "learning_rate": .01, "weight_decay": .01, "max_epochs": 80},
)
VALIDATE_EVERY = 5           # epochs between validation checks
EARLY_STOPPING_PATIENCE = 5  # validation checks without improvement

# Serving
MAX_K = 20
MAX_LIST_PREFERENCES = 20

# Approximate neighborhood centers offered as "near" presets in the UI.
LOCATION_PRESETS = {
    "evanston": [
        {"name": "Downtown Evanston", "latitude": 42.0471, "longitude": -87.6835},
        {"name": "Northwestern campus", "latitude": 42.0565, "longitude": -87.6753},
        {"name": "Main Street", "latitude": 42.0334, "longitude": -87.6799},
    ],
    "chicago": [
        {"name": "The Loop", "latitude": 41.8837, "longitude": -87.6289},
        {"name": "River North", "latitude": 41.8924, "longitude": -87.6341},
        {"name": "West Loop", "latitude": 41.8826, "longitude": -87.6470},
        {"name": "Lincoln Park", "latitude": 41.9214, "longitude": -87.6513},
        {"name": "Lakeview", "latitude": 41.9434, "longitude": -87.6553},
        {"name": "Wicker Park", "latitude": 41.9088, "longitude": -87.6796},
        {"name": "Logan Square", "latitude": 41.9231, "longitude": -87.7093},
        {"name": "Andersonville", "latitude": 41.9800, "longitude": -87.6684},
        {"name": "Pilsen", "latitude": 41.8567, "longitude": -87.6563},
        {"name": "Chinatown", "latitude": 41.8520, "longitude": -87.6320},
        {"name": "Hyde Park", "latitude": 41.7943, "longitude": -87.5907},
        {"name": "Evanston", "latitude": 42.0471, "longitude": -87.6835},
    ],
}

# Per-dataset experiment profiles. Every choice marked "selected on validation" is made
# before the test split is scored.
PROFILES = {
    "evanston": {
        "k": 5, "max_queries": None, "validation_sample": None,
        "text_backends": ("tfidf", "sentence"),           # served backend selected on validation
        "tfidf_min_df": 1, "min_category_items": 1, "geo_anchors": 0,
        "two_tower": {"losses": ("full",), "extras": (None,), "search": (
            {"dim": 8, "hidden": 0, "temperature": .2, "learning_rate": .01, "weight_decay": .001},
            {"dim": 8, "hidden": 0, "temperature": .2, "learning_rate": .01, "weight_decay": .01},
            {"dim": 16, "hidden": 32, "temperature": .2, "learning_rate": .01, "weight_decay": .01}),
            "batch_size": None, "max_epochs": 80, "validate_every": 5, "patience": 5},
        "semantic_ids": None, "sequential": None, "ranker": None, "ann": None,
    },
    "chicago": {
        "k": 10, "max_queries": 20_000, "validation_sample": 4_000,
        "text_backends": ("sentence", "tfidf"),
        "tfidf_min_df": 5, "min_category_items": 10, "geo_anchors": 32,
        "two_tower": {"losses": ("in_batch_logq", "in_batch", "full"), "extras": (None, "id", "sid"), "search": (
            {"dim": 64, "hidden": 256, "temperature": .05, "learning_rate": 1e-3, "weight_decay": 0.},),
            "batch_size": 1024, "max_epochs": 8, "validate_every": 1, "patience": 2},
        "semantic_ids": {"levels": 3, "codebook_size": 64},
        "sequential": {"extras": (None, "id", "sid"), "dim": 64, "layers": 2, "heads": 2, "dropout": .2, "max_len": 50,
                       "learning_rate": 1e-3, "weight_decay": 0., "batch_size": 256, "max_epochs": 12, "patience": 2},
        "ranker": {"sources": {"two_tower": 100, "content": 50, "popular": 50, "nearby": 50},
                   "objectives": ("lambdarank", "binary", "rank_xendcg")},
        "ann": {"k": 100, "n_probes": (1, 2, 4, 8, 16), "queries": 2_000},
    },
}
