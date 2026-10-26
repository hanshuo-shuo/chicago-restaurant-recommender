"""Sequential next-restaurant model: a causal Transformer over each user's visit history.

Architecture follows SASRec (Kang & McAuley, ICDM 2018), built on the Transformer encoder
(Vaswani et al., NeurIPS 2017): learned positions, causal self-attention, scores = hidden
state . item table. Training is autoregressive next-item prediction over the full catalog,
the generative objective HSTU scales up (Zhai et al., ICML 2024).
Items enter through their content features, optionally plus ID or Semantic-ID
prefix embeddings, so unseen restaurants still have a representation.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import POSITIVE_RATING, SEED
from .dataset import Dataset, Histories, Queries


def build_sequences(ds: Dataset, train_end: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping windows of each user's training history; targets are -1 where not a positive."""
    hist = Histories(ds, cutoff=train_end)
    pad = ds.n_items
    inputs, targets = [], []
    for user in range(ds.n_users):
        lo, hi = hist.ptr[user], hist.ptr[user + 1]
        items, ratings = hist.item[lo:hi], hist.rating[lo:hi]
        end = len(items)
        while end >= 2:
            start = max(0, end - (max_len + 1))
            window, window_ratings = items[start:end], ratings[start:end]
            x, y = window[:-1], np.where(window_ratings[1:] >= POSITIVE_RATING, window[1:], -1)
            inputs.append(np.pad(x, (0, max_len - len(x)), constant_values=pad))
            targets.append(np.pad(y, (0, max_len - len(y)), constant_values=-1))
            end = start + 1
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)


class SequentialRecommender(nn.Module):
    def __init__(self, item_inputs: np.ndarray, dim: int = 64, layers: int = 2, heads: int = 2, max_len: int = 50,
                 dropout: float = .2, extra: str | None = None, prefixes: np.ndarray | None = None):
        super().__init__()
        self.register_buffer("item_inputs", torch.tensor(item_inputs, dtype=torch.float32))
        self.n_items, self.max_len, self.extra = len(item_inputs), max_len, extra
        self.project = nn.Linear(item_inputs.shape[1], dim)
        if extra == "id":
            self.id_embedding = nn.Embedding(self.n_items, dim)
            nn.init.normal_(self.id_embedding.weight, std=.02)
        elif extra == "sid":
            self.register_buffer("prefixes", torch.tensor(prefixes, dtype=torch.long))
            self.sid_embeddings = nn.ModuleList(nn.Embedding(int(prefixes[:, level].max()) + 1, dim)
                                                for level in range(prefixes.shape[1]))
            for table in self.sid_embeddings:
                nn.init.normal_(table.weight, std=.02)
        self.position = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.triu(torch.full((max_len, max_len), float("-inf")), 1))

    def item_table(self) -> torch.Tensor:
        table = self.project(self.item_inputs)
        if self.extra == "id":
            table = table + self.id_embedding.weight
        elif self.extra == "sid":
            for level, embedding in enumerate(self.sid_embeddings):
                table = table + embedding(self.prefixes[:, level])
        return table

    def forward(self, sequences: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
        """Right-padded (pad = n_items) sequences -> hidden state per position; causal mask, no padding mask needed."""
        table = self.item_table() if table is None else table
        padded = torch.cat([table, torch.zeros(1, table.shape[1])])
        length = sequences.shape[1]
        x = padded[sequences] + self.position.weight[:length]
        return self.norm(self.encoder(self.dropout(x), mask=self.causal[:length, :length]))


def train_sequential(inputs: np.ndarray, targets: np.ndarray, item_inputs: np.ndarray, config: dict,
                     validate: Callable[[SequentialRecommender], float], prefixes: np.ndarray | None = None,
                     seed: int = SEED,
                     log: Callable[[str], None] = lambda message: None) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SequentialRecommender(item_inputs, config["dim"], config["layers"], config["heads"], inputs.shape[1],
                                  config["dropout"], config.get("extra"), prefixes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x_all, y_all = torch.from_numpy(inputs), torch.from_numpy(targets)
    lengths = (inputs != len(item_inputs)).sum(1)
    best = {"score": -np.inf, "epoch": 0, "state": None}
    curve, stale, started = [], 0, time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        jitter = lengths + rng.random(len(lengths))  # random order within equal lengths
        buckets = np.argsort(jitter, kind="stable")
        batches = [buckets[s:s + config["batch_size"]] for s in range(0, len(buckets), config["batch_size"])]
        for batch in rng.permutation(len(batches)):
            rows = torch.from_numpy(batches[batch])
            width = int(lengths[batches[batch]].max())
            x, y = x_all[rows, :width], y_all[rows, :width]
            table = model.item_table()
            hidden = model(x, table)
            valid = y >= 0
            logits = hidden[valid] @ table.T
            loss = F.cross_entropy(logits, y[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        score = validate(model)
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "validation_ndcg": round(score, 6)})
        log(f"    epoch {epoch}: loss {np.mean(losses):.4f}  val NDCG {score:.4f}")
        if score > best["score"] + 1e-6:
            best, stale = {"score": score, "epoch": epoch,
                           "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}, 0
        else:
            stale += 1
            if stale >= config.get("patience", 2):
                break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"configuration": config, "best_epoch": best["epoch"], "validation_ndcg": round(best["score"], 6),
                   "learning_curve": curve, "training_windows": int(len(inputs)),
                   "seconds": round(time.perf_counter() - started, 1)}


def sequential_scores(model: SequentialRecommender, queries: Queries, fallback: np.ndarray) -> np.ndarray:
    """Scores from the last history position; users without history get the fallback (popularity) scores."""
    pad, max_len = model.n_items, model.max_len
    scores = np.tile(fallback, (len(queries), 1))
    rows = [i for i, h in enumerate(queries.history_items) if len(h)]
    if not rows:
        return scores
    sequences = np.full((len(rows), max_len), pad, dtype=np.int64)
    last = np.zeros(len(rows), dtype=np.int64)
    for j, i in enumerate(rows):
        h = queries.history_items[i][-max_len:]
        sequences[j, :len(h)] = h
        last[j] = len(h) - 1
    with torch.no_grad():
        table = model.item_table()
        hidden = model(torch.from_numpy(sequences), table)
        out = hidden[torch.arange(len(rows)), torch.from_numpy(last)] @ table.T
    scores[rows] = out.numpy()
    return scores
