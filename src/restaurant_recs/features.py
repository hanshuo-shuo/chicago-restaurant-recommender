"""Item content features and preference/history query vectors.

Text is encoded by TF-IDF (Salton & Buckley, IP&M 1988) or by the frozen pretrained sentence
encoder all-MiniLM-L6-v2 (Apache-2.0; Sentence-BERT, Reimers & Gurevych, EMNLP 2019; MiniLM,
Wang et al., NeurIPS 2020).
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np

from .config import POSITIVE_RATING

TEXT_BACKENDS = ("tfidf", "sentence")
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z']{1,}", text.casefold())


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def item_text(item: dict) -> str:
    """Natural-language description used by the sentence encoder."""
    parts = [item.get("name") or "", ", ".join(item.get("categories") or []), item.get("description") or "",
             "; ".join(item.get("attributes") or [])]
    return ". ".join(p for p in parts if p)


class SentenceEncoder:
    """Frozen pretrained sentence encoder (Sentence-BERT family); loaded only at training time."""
    def __init__(self, model_name: str = DEFAULT_ENCODER):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)


class ItemFeatures:
    """Item content vectors: text + category multi-hot + cost, price-level and opening-hours scalars.

    The text block is TF-IDF over name/categories/description/attributes, or a frozen
    pretrained sentence encoder over the same fields. Fitted on catalog metadata only
    (never on interactions or review text), so new items are retrievable as soon as
    they enter the catalog. ``config`` (plus ``text_matrix`` for the sentence backend)
    reproduces the exact training-time features at serving time without the encoder.
    """
    def __init__(self, items: list[dict], config: dict | None = None, text_matrix: np.ndarray | None = None,
                 text_backend: str = "tfidf", encoder=None, min_category_items: int = 1, min_df: int = 1):
        self.items = sorted(items, key=lambda item: item["item_id"])
        self.ids = [item["item_id"] for item in self.items]
        self.index = {item_id: i for i, item_id in enumerate(self.ids)}
        if config is None:
            if text_backend not in TEXT_BACKENDS:
                raise ValueError(f"text_backend must be one of {TEXT_BACKENDS}")
            counts = Counter(c.casefold() for item in self.items for c in item.get("categories") or [])
            config = {"text_backend": text_backend, "feature_version": "item_content_v2",
                      "categories": sorted(c for c, n in counts.items() if n >= min_category_items),
                      "fit_scope": "catalog metadata only; no interactions"}
            if text_backend == "tfidf":
                docs = [self._tfidf_tokens(item) for item in self.items]
                frequency = Counter(token for doc in docs for token in set(doc))
                vocabulary = sorted(t for t, n in frequency.items() if n >= min_df)
                config.update(vocabulary=vocabulary,
                              idf=[float(np.log((1 + len(docs)) / (1 + frequency[t])) + 1) for t in vocabulary])
            else:
                encoder = encoder or SentenceEncoder()
                config["encoder"] = getattr(encoder, "model_name", type(encoder).__name__)
                text_matrix = np.asarray(encoder.encode([item_text(item) for item in self.items]), dtype=np.float64)
        self.config = config
        if config["text_backend"] == "tfidf":
            vocab = {token: i for i, token in enumerate(config["vocabulary"])}
            text = np.zeros((len(self.items), len(vocab)), dtype=np.float64)
            for i, item in enumerate(self.items):
                for token, count in Counter(self._tfidf_tokens(item)).items():
                    if token in vocab:
                        text[i, vocab[token]] = (1 + np.log(count)) * config["idf"][vocab[token]]
        else:
            if text_matrix is None or len(text_matrix) != len(self.items):
                raise ValueError("The sentence backend needs the saved text matrix for every item.")
            text = np.asarray(text_matrix, dtype=np.float64)
        self.text_matrix = normalize(text)
        category_index = {c: i for i, c in enumerate(config["categories"])}
        category = np.zeros((len(self.items), len(category_index)))
        scalar = np.zeros((len(self.items), 6))
        for i, item in enumerate(self.items):
            known = [category_index[c.casefold()] for c in item.get("categories") or [] if c.casefold() in category_index]
            category[i, known] = .8 / np.sqrt(len(known)) if known else 0
            cost = item.get("average_cost")
            if cost is not None and float(cost) > 0:
                scalar[i, 0] = min(np.log1p(float(cost)) / 5, 1) * .3
                scalar[i, 1] = .3
            opening = item.get("open_after_8pm")
            if opening is not None:
                scalar[i, 2 if opening else 3] = .25
            if item.get("price_level"):
                scalar[i, 4] = item["price_level"] / 4 * .3
                scalar[i, 5] = .3
        self.matrix = normalize(np.concatenate([self.text_matrix, category, scalar], axis=1))

    @staticmethod
    def _tfidf_tokens(item: dict) -> list[str]:
        fields = [item.get("name") or "", " ".join(item.get("categories") or []), item.get("description") or "",
                  " ".join(item.get("attributes") or [])]
        return _tokens(" ".join(fields))

    def query(self, history: list[dict], preferences: dict | None = None) -> np.ndarray:
        """Weighted mean of liked items, explicit seed items and requested cuisines."""
        items = np.array([self.index[e["item_id"]] for e in history if e["item_id"] in self.index], dtype=int)
        ratings = np.array([e["rating"] for e in history if e["item_id"] in self.index], dtype=int)
        return self.query_from_indices(items, ratings, preferences)

    def query_from_indices(self, items: np.ndarray, ratings: np.ndarray, preferences: dict | None = None) -> np.ndarray:
        vectors, weights = [], []
        liked = ratings >= POSITIVE_RATING
        if liked.any():
            vectors.extend(self.matrix[items[liked]])
            weights.extend(ratings[liked] - (POSITIVE_RATING - 1))
        preferences = preferences or {}
        for item_id in preferences.get("seed_item_ids", []):
            if item_id in self.index:
                vectors.append(self.matrix[self.index[item_id]])
                weights.append(2)
        requested = {str(value).casefold() for value in preferences.get("cuisines", [])}
        if requested:
            if not hasattr(self, "_by_cuisine"):
                self._by_cuisine = {}
                for i, item in enumerate(self.items):
                    self._by_cuisine.setdefault(str(item.get("cuisine") or "").casefold(), []).append(i)
            matches = sorted(i for c in requested for i in self._by_cuisine.get(c, []))
            if matches:
                vectors.append(self.matrix[matches].mean(axis=0))
                weights.append(2)
        vector = np.average(vectors, axis=0, weights=weights) if vectors else np.zeros(self.matrix.shape[1])
        return normalize(vector)
