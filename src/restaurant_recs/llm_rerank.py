"""Optional LLM reranker for the final shortlist, served as its own model arm ("llm") for A/B testing.

Following Hou et al. (ECIR 2024), the model ranks retrieved candidates zero-shot given the
user's stated preferences. LLM rankers are sensitive to candidate order, so each request
asks for rankings under several candidate orders and aggregates them with a Borda count.
Any failure (no credentials, network error, refusal, malformed output) returns the input
order unchanged, so the arm degrades to the ranker it wraps instead of failing the request.

Requires the ``anthropic`` package (``pip install -e '.[llm]'``) and API credentials.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

DEFAULT_MODEL = "claude-opus-5-5"
SYSTEM = ("You re-rank restaurant recommendations. You receive a user's stated preferences and a list of "
          "candidate restaurants already filtered to satisfy every hard constraint. Rank all candidates from best "
          "to worst fit for this user, judging cuisine, atmosphere, price, distance and similarity to restaurants "
          "they said they like. Use only the information given. Return every candidate id exactly once.")
SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}},
        "top_reason": {"type": "string"},
    },
    "required": ["ranking", "top_reason"],
    "additionalProperties": False,
}


def _describe(item: dict) -> dict:
    keep = ("item_id", "name", "cuisine", "categories", "price_level", "average_cost", "open_after_8pm", "distance_km")
    out = {key: item.get(key) for key in keep if item.get(key) not in (None, [], "")}
    if item.get("description"):
        out["description"] = item["description"][:240]
    return out


def borda(rankings: list[list[str]], candidates: list[str]) -> list[str]:
    """Aggregate several rankings; ties keep the original candidate order."""
    points = {c: 0. for c in candidates}
    for ranking in rankings:
        for position, item_id in enumerate(ranking):
            points[item_id] += len(ranking) - position
    original = {c: i for i, c in enumerate(candidates)}
    return sorted(candidates, key=lambda c: (-points[c], original[c]))


class LLMReranker:
    def __init__(self, client=None, model: str = DEFAULT_MODEL, permutations: int = 2, effort: str = "low"):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.permutations, self.effort = client, model, permutations, effort

    @property
    def version(self) -> str:
        return f"llm:{self.model}"

    def _rank_once(self, preferences: dict, items: list[dict]) -> list[str] | None:
        message = {"preferences": preferences, "candidates": [_describe(item) for item in items]}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a safety decline is retried server-side on Anthropic's recommended model
            system=SYSTEM,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": json.dumps(message, ensure_ascii=False)}],
        )
        if response.stop_reason != "end_turn":  # refusal (after fallback), max_tokens, ...
            return None
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            return None
        ranking = json.loads(text)["ranking"]
        valid = {item["item_id"] for item in items}
        seen, cleaned = set(), []
        for item_id in ranking:
            if item_id in valid and item_id not in seen:
                seen.add(item_id)
                cleaned.append(item_id)
        return cleaned + [item["item_id"] for item in items if item["item_id"] not in seen]

    def rerank(self, preferences: dict, items: list[dict]) -> tuple[list[dict], str | None]:
        """Return (reordered items, error message or None). Orders: as retrieved, reversed, then rotations."""
        if len(items) < 2:
            return items, None
        orders = [items, items[::-1]] + [items[i:] + items[:i] for i in range(1, self.permutations - 1)]
        orders = orders[:self.permutations]
        try:
            with ThreadPoolExecutor(max_workers=len(orders)) as pool:
                rankings = list(pool.map(lambda order: self._rank_once(preferences, order), orders))
        except Exception as exc:  # network, auth, rate limit, malformed JSON: keep the wrapped ranker's order
            return items, f"LLM rerank failed ({type(exc).__name__}); kept the base ranking."
        rankings = [r for r in rankings if r]
        if not rankings:
            return items, "LLM rerank returned no usable ranking; kept the base ranking."
        by_id = {item["item_id"]: item for item in items}
        return [by_id[i] for i in borda(rankings, [item["item_id"] for item in items])], None
