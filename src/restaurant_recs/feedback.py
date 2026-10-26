"""Learning from logged feedback: a multi-task (click, favorite) ranker with a position-bias shallow tower.

Follows Zhao et al. (RecSys 2019): a Multi-gate Mixture-of-Experts network predicts
each task's logit from request/item features, and a shallow tower over the display
position adds a bias logit during training only. Serving drops the position term, so
items are ranked by estimated utility instead of by where an earlier ranker put them.

Training data comes from the server's impression log, joined to the clicks and
favorites that reference each impression and to the request's logged preferences.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

FEATURES = ("ranking_score", "retrieval_score", "log_distance_km", "distance_missing", "log_train_count",
            "bayesian_rating", "price_level", "price_missing", "open_late", "cuisine_requested")
TASKS = ("click", "favorite")
MIN_IMPRESSIONS, MIN_CLICKS = 2_000, 100


def load_logs(db_path: str | Path, engine) -> dict[str, np.ndarray]:
    """One row per logged impression: features, display position, and click/favorite labels."""
    with sqlite3.connect(str(db_path)) as connection:
        rows = connection.execute("""
            SELECT i.item_id, i.position, i.model, i.retrieval_score, i.ranking_score, i.distance_km, r.preferences,
                   EXISTS(SELECT 1 FROM events e WHERE e.impression_id = i.id AND e.type = 'click'),
                   EXISTS(SELECT 1 FROM events e WHERE e.impression_id = i.id AND e.type = 'favorite')
            FROM impressions i LEFT JOIN requests r ON r.request_id = i.request_id ORDER BY i.id""").fetchall()
    x, position, labels, models = [], [], [], []
    for item_id, pos, model, retrieval, ranking, distance, preferences, clicked, favorited in rows:
        i = engine.features.index.get(item_id)
        if i is None:
            continue
        item = engine.items[i]
        requested = {c.casefold() for c in json.loads(preferences or "{}").get("cuisines", [])}
        price = item.get("price_level")
        x.append([ranking or 0., retrieval or 0., np.log1p(distance) if distance is not None else 0., float(distance is None),
                  engine.stats.log_count[i], engine.stats.bayesian[i], price or 0., float(price is None),
                  float(bool(item.get("open_after_8pm"))), float(str(item.get("cuisine", "")).casefold() in requested)])
        position.append(pos)
        labels.append([clicked, favorited])
        models.append(model)
    return {"x": np.array(x, dtype=np.float32).reshape(-1, len(FEATURES)), "position": np.array(position, dtype=np.int64),
            "labels": np.array(labels, dtype=np.float32).reshape(-1, len(TASKS)), "model": np.array(models)}


def _build_model(n_features: int, max_position: int, experts: int, hidden: int, position_tower: bool):
    import torch
    from torch import nn

    class MMoEWithPositionTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList(nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU()) for _ in range(experts))
            self.gates = nn.ModuleList(nn.Linear(n_features, experts) for _ in TASKS)
            self.towers = nn.ModuleList(nn.Linear(hidden, 1) for _ in TASKS)
            self.position = nn.Embedding(max_position + 1, len(TASKS)) if position_tower else None
            if self.position is not None:
                nn.init.zeros_(self.position.weight)

        def forward(self, x, position=None):
            expert_out = torch.stack([e(x) for e in self.experts], dim=1)
            logits = torch.cat([tower((torch.softmax(gate(x), -1)[..., None] * expert_out).sum(1))
                                for gate, tower in zip(self.gates, self.towers)], dim=1)
            if self.position is not None and position is not None:
                logits = logits + self.position(position)
            return logits
    return MMoEWithPositionTower()


def train_feedback(logs: dict[str, np.ndarray], position_tower: bool = True, experts: int = 4, hidden: int = 32,
                   epochs: int = 60, learning_rate: float = .01, seed: int = 0):
    """Fit on standardized features; returns (model, feature mean, feature std)."""
    import torch
    torch.manual_seed(seed)
    mean, std = logs["x"].mean(0), logs["x"].std(0) + 1e-6
    x = torch.tensor((logs["x"] - mean) / std)
    position = torch.tensor(np.minimum(logs["position"], 50))
    labels = torch.tensor(logs["labels"])
    model = _build_model(x.shape[1], 50, experts, hidden, position_tower)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x, position), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


def utility(model, mean: np.ndarray, std: np.ndarray, x: np.ndarray, task_weights=(1., 2.)) -> np.ndarray:
    """Position-free ranking score: weighted sum of task probabilities."""
    import torch
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.tensor((x - mean) / std, dtype=torch.float32))).numpy()
    return probabilities @ np.array(task_weights)


def learned_position_bias(model) -> np.ndarray:
    """Per-position, per-task bias logits (rows = positions 1..50)."""
    return model.position.weight.detach().numpy()[1:] if model.position is not None else np.zeros((50, len(TASKS)))


def train_from_log(db_path: str | Path, engine, artifact_dir: str | Path) -> dict:
    """CLI entry: refuses to train on too little traffic, otherwise saves the model and a summary."""
    import torch
    logs = load_logs(db_path, engine)
    impressions, clicks = len(logs["x"]), int(logs["labels"][:, 0].sum()) if len(logs["x"]) else 0
    if impressions < MIN_IMPRESSIONS or clicks < MIN_CLICKS:
        return {"trained": False, "impressions": impressions, "clicks": clicks,
                "reason": f"Need at least {MIN_IMPRESSIONS} impressions and {MIN_CLICKS} clicks of real traffic."}
    model, mean, std = train_feedback(logs)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "features": FEATURES},
               Path(artifact_dir) / "feedback_ranker.pt")
    bias = learned_position_bias(model)
    return {"trained": True, "impressions": impressions, "clicks": clicks,
            "position_bias_click_logit": [round(float(b), 3) for b in bias[:10, 0]]}
