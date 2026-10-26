"""Two-tower retrieval: training examples, model and three softmax variants.

Losses
  full           exact softmax over the whole catalog; the user's other known positives are masked
  in_batch       softmax over the other targets in the mini-batch (the scalable choice)
  in_batch_logq  in-batch plus the sampling-bias correction of Yi et al. (RecSys 2019): subtract
                 log q_j, where q_j is item j's share of training positives, so popular items are
                 not over-penalized just for appearing as negatives more often

The item tower can add a per-item ID embedding (memorization) or Semantic-ID prefix
embeddings (memorization shared across similar items; Singh et al., RecSys 2024).
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import POSITIVE_RATING, SEED
from .dataset import Dataset, Histories
from .geo import GeoEncoder, user_location

LOSSES = ("full", "in_batch", "in_batch_logq")
ITEM_EXTRAS = (None, "id", "sid")


@dataclass
class Examples:
    """One row per (training positive); history and known-positive lists are padded with n_items."""
    history: np.ndarray
    weights: np.ndarray
    geo: np.ndarray
    target: np.ndarray
    known: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def build_examples(ds: Dataset, train_end: int, coords: np.ndarray, geo: GeoEncoder | None,
                   max_history: int = 30, max_known: int = 40) -> Examples:
    """Training positives with the user's strictly-earlier history, as the serving path would see it."""
    hist = Histories(ds, cutoff=train_end)
    pad = ds.n_items
    history, weights, locations, targets, known = [], [], [], [], []
    for user in range(ds.n_users):
        lo, hi = hist.ptr[user], hist.ptr[user + 1]
        items, days, ratings = hist.item[lo:hi], hist.day[lo:hi], hist.rating[lo:hi]
        liked = ratings >= POSITIVE_RATING
        for day in np.unique(days[liked]):
            before = days < day
            prior_liked = items[before & liked][-max_history:]
            today = items[(days == day) & liked]
            location = user_location(items[before], ratings[before], coords) if before.any() else np.array([np.nan, np.nan])
            prior_weights = (ratings[before & liked][-max_history:] - (POSITIVE_RATING - 1)).astype(np.float32)
            mask_pool = np.unique(np.concatenate([items[before & liked], today]))
            for target in np.unique(today):
                others = mask_pool[mask_pool != target][-max_known:]
                history.append(np.pad(prior_liked, (0, max_history - len(prior_liked)), constant_values=pad))
                weights.append(np.pad(prior_weights, (0, max_history - len(prior_weights))))
                known.append(np.pad(others, (0, max_known - len(others)), constant_values=pad))
                locations.append(location)
                targets.append(target)
    locations = np.array(locations).reshape(-1, 2)
    geo_features = geo.encode(locations) if geo is not None else np.zeros((len(targets), 0))
    return Examples(np.array(history, dtype=np.int64), np.array(weights, dtype=np.float32),
                    geo_features.astype(np.float32), np.array(targets, dtype=np.int64), np.array(known, dtype=np.int64))


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, hidden), nn.Linear(hidden, outputs)] if hidden
                                    else [nn.Linear(inputs, outputs)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for depth, layer in enumerate(self.layers):
            x = layer(x)
            if depth < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def export(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(l.weight.detach().cpu().numpy().astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64))
                for l in self.layers]


class TwoTower(nn.Module):
    def __init__(self, item_inputs: np.ndarray, query_dim: int, dim: int, hidden: int, temperature: float,
                 extra: str | None = None, prefixes: np.ndarray | None = None):
        super().__init__()
        self.register_buffer("item_inputs", torch.tensor(item_inputs, dtype=torch.float32))
        self.query_tower = MLP(query_dim, hidden, dim)
        self.item_tower = MLP(item_inputs.shape[1], hidden, dim)
        self.item_bias = nn.Parameter(torch.zeros(len(item_inputs)))
        self.temperature, self.extra = temperature, extra
        if extra == "id":
            self.id_embedding = nn.Embedding(len(item_inputs), dim)
            nn.init.normal_(self.id_embedding.weight, std=.01)
        elif extra == "sid":
            self.register_buffer("prefixes", torch.tensor(prefixes, dtype=torch.long))
            self.sid_embeddings = nn.ModuleList(nn.Embedding(int(prefixes[:, level].max()) + 1, dim)
                                                for level in range(prefixes.shape[1]))
            for table in self.sid_embeddings:
                nn.init.normal_(table.weight, std=.01)

    def items(self, index: torch.Tensor | None = None) -> torch.Tensor:
        inputs = self.item_inputs if index is None else self.item_inputs[index]
        out = self.item_tower(inputs)
        if self.extra == "id":
            out = out + (self.id_embedding.weight if index is None else self.id_embedding(index))
        elif self.extra == "sid":
            prefixes = self.prefixes if index is None else self.prefixes[index]
            for level, table in enumerate(self.sid_embeddings):
                out = out + table(prefixes[:, level])
        return F.normalize(out, dim=-1)

    def queries(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_tower(inputs), dim=-1)

    def export(self) -> dict:
        with torch.no_grad():
            return {"query_layers": self.query_tower.export(),
                    "item_embeddings": self.items().cpu().numpy().astype(np.float64),
                    "item_bias": self.item_bias.detach().cpu().numpy().astype(np.float64),
                    "temperature": self.temperature}


def _query_inputs(content: torch.Tensor, history: torch.Tensor, weights: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
    pooled = F.normalize((content[history] * weights[..., None]).sum(1), dim=-1)
    return torch.cat([pooled, geo, torch.ones(len(pooled), 1)], dim=1)


def train_two_tower(examples: Examples, content: np.ndarray, item_inputs: np.ndarray, config: dict,
                    validate: Callable[[dict], float], item_counts: np.ndarray | None = None,
                    prefixes: np.ndarray | None = None, seed: int = SEED,
                    log: Callable[[str], None] = lambda message: None) -> tuple[dict, dict]:
    """Mini-batch training with validation-based early stopping; returns (best exported weights, trial summary)."""
    if config["loss"] not in LOSSES or config.get("extra") not in ITEM_EXTRAS:
        raise ValueError("Unknown loss or item extra.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    content_t = torch.tensor(np.vstack([content, np.zeros(content.shape[1])]), dtype=torch.float32)
    history, weights = torch.from_numpy(examples.history), torch.from_numpy(examples.weights)
    geo, target, known = torch.from_numpy(examples.geo), torch.from_numpy(examples.target), torch.from_numpy(examples.known)
    n_items = len(item_inputs)
    model = TwoTower(item_inputs, content.shape[1] + examples.geo.shape[1] + 1, config["dim"], config["hidden"],
                     config["temperature"], config.get("extra"), prefixes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    log_q = None
    if config["loss"] == "in_batch_logq":
        share = np.maximum(item_counts, 1) / np.maximum(item_counts, 1).sum()
        log_q = torch.tensor(np.log(share), dtype=torch.float32)
    batch_size = config.get("batch_size") or len(examples)
    best = {"score": -np.inf, "epoch": 0, "weights": None}
    curve, stale, started = [], 0, time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(examples)))
        losses = []
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            q = model.queries(_query_inputs(content_t, history[b], weights[b], geo[b]))
            if config["loss"] == "full":
                logits = q @ model.items().T / model.temperature + model.item_bias
                logits = torch.cat([logits, torch.zeros(len(b), 1)], dim=1)
                logits = logits.scatter(1, known[b], -1e9)[:, :n_items]
                labels = target[b]
            else:
                t = target[b]
                logits = q @ model.items(t).T / model.temperature + model.item_bias[t]
                if log_q is not None:
                    logits = logits - log_q[t][None, :]
                collide = (t[None, :] == t[:, None]) | (known[b][:, :, None] == t[None, None, :]).any(1)
                collide.fill_diagonal_(False)
                logits = logits.masked_fill(collide, -1e9)
                labels = torch.arange(len(b))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss))
        if epoch % config.get("validate_every", 1) and epoch != config["max_epochs"]:
            continue
        model.eval()
        exported = model.export()
        score = validate(exported)
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "validation_ndcg": round(score, 6)})
        log(f"    epoch {epoch}: loss {np.mean(losses):.4f}  val NDCG {score:.4f}")
        if score > best["score"] + 1e-6:
            best, stale = {"score": score, "epoch": epoch, "weights": exported}, 0
        else:
            stale += 1
            if stale >= config.get("patience", 2):
                break
    return best["weights"], {"configuration": config, "best_epoch": best["epoch"],
                             "validation_ndcg": round(best["score"], 6), "learning_curve": curve,
                             "training_examples": len(examples), "seconds": round(time.perf_counter() - started, 1)}
