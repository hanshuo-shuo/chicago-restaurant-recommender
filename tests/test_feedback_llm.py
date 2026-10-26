"""Learning from logs (position-debiased multi-task ranker) and the optional LLM reranker."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen
import uuid

import numpy as np

from restaurant_recs.engine import Engine
from restaurant_recs.feedback import FEATURES, learned_position_bias, load_logs, train_feedback, train_from_log, utility
from restaurant_recs.llm_rerank import DEFAULT_MODEL, LLMReranker, borda
from restaurant_recs.server import create_server

from support import synthetic_dataset


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def simulated_logs(requests: int = 800, slots: int = 8, seed: int = 0) -> tuple[dict, np.ndarray]:
    """An old ranker ordered items by a feature unrelated to utility; clicks also fall with position."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(requests * slots, len(FEATURES))).astype(np.float32)
    true_utility = x[:, 0] + .8 * x[:, 4]
    confounder = x[:, 2]
    position = np.concatenate([np.argsort(np.argsort(-(confounder[r * slots:(r + 1) * slots] + .3 * rng.normal(size=slots)))) + 1
                               for r in range(requests)])
    bias = -1.2 * np.log(position)
    click = rng.random(len(x)) < 1 / (1 + np.exp(-(true_utility + bias - .5)))
    favorite = rng.random(len(x)) < 1 / (1 + np.exp(-(true_utility + .5 * bias - 2)))
    return ({"x": x, "position": position.astype(np.int64), "labels": np.stack([click, favorite], 1).astype(np.float32),
             "model": np.array(["baseline"] * len(x))}, true_utility)


class FeedbackTests(unittest.TestCase):
    def test_position_tower_absorbs_bias_that_a_naive_model_attributes_to_features(self):
        logs, _ = simulated_logs()
        debiased, mean, std = train_feedback(logs, position_tower=True, epochs=150)
        naive, naive_mean, naive_std = train_feedback(logs, position_tower=False, epochs=150)
        bias = learned_position_bias(debiased)[:8, 0]
        self.assertGreater(bias[0], bias[-1])
        self.assertGreater(np.corrcoef(bias, -np.log(np.arange(1, 9)))[0, 1], .9)  # recovers the simulated -log(position) shape
        fresh, truth = simulated_logs(requests=300, seed=1)
        good = rank_correlation(utility(debiased, mean, std, fresh["x"]), truth)
        bad = rank_correlation(utility(naive, naive_mean, naive_std, fresh["x"]), truth)
        self.assertGreater(good, bad)

    def test_logs_round_trip_through_the_server_and_too_little_traffic_is_refused(self):
        ds = synthetic_dataset("evanston", n_users=60, n_items=12)
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(ds, directory)
            db = Path(directory) / "events.sqlite3"
            server = create_server(engine, port=0, db_path=db)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = "http://127.0.0.1:%d" % server.server_address[1]
            session = str(uuid.uuid4())

            def post(path, body):
                with urlopen(Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=5) as r:
                    return json.load(r)
            try:
                result = post("/api/recommend", {"session_id": session, "k": 4, "preferences": {"cuisines": ["Thai", "Italian"]}})
                post("/api/events", {"session_id": session, "request_id": result["request_id"],
                                     "item_id": result["recommendations"][1]["item_id"], "type": "click"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            logs = load_logs(db, engine)
            self.assertEqual(logs["x"].shape, (4, len(FEATURES)))
            np.testing.assert_array_equal(logs["position"], [1, 2, 3, 4])
            np.testing.assert_array_equal(logs["labels"][:, 0], [0, 1, 0, 0])
            self.assertTrue(np.all(logs["x"][:, FEATURES.index("cuisine_requested")] == 1))
            outcome = train_from_log(db, engine, directory)
            self.assertFalse(outcome["trained"])
            self.assertFalse((Path(directory) / "feedback_ranker.pt").exists())


class FakeMessages:
    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.respond(kwargs)


def fake_client(respond):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(respond)))


def reply(ranking, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=json.dumps({"ranking": ranking, "top_reason": "fit"}))])


def candidates_in(kwargs):
    return [c["item_id"] for c in json.loads(kwargs["messages"][0]["content"])["candidates"]]


class LLMRerankTests(unittest.TestCase):
    items = [{"item_id": f"r{i}", "name": f"Place {i}", "cuisine": "Thai", "distance_km": float(i)} for i in range(5)]

    def test_request_shape(self):
        client = fake_client(lambda kwargs: reply(candidates_in(kwargs)))
        LLMReranker(client=client, permutations=1).rerank({"cuisines": ["Thai"]}, self.items)
        call = client.beta.messages.calls[0]
        self.assertEqual(call["model"], DEFAULT_MODEL)
        self.assertEqual(call["fallbacks"], "default")
        self.assertEqual(call["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(call["output_config"]["effort"], "low")
        self.assertNotIn("temperature", call)
        payload = json.loads(call["messages"][0]["content"])
        self.assertEqual(payload["preferences"], {"cuisines": ["Thai"]})
        self.assertEqual(len(payload["candidates"]), 5)

    def test_consistent_preference_survives_and_pure_position_bias_cancels(self):
        preferred = ["r3", "r1", "r4", "r0", "r2"]
        ranked, error = LLMReranker(client=fake_client(lambda kwargs: reply(preferred))).rerank({}, self.items)
        self.assertIsNone(error)
        self.assertEqual([i["item_id"] for i in ranked], preferred)
        echo = LLMReranker(client=fake_client(lambda kwargs: reply(candidates_in(kwargs))))  # always keeps the order it was shown
        ranked, _ = echo.rerank({}, self.items)
        self.assertEqual([i["item_id"] for i in ranked], [i["item_id"] for i in self.items])
        self.assertEqual(borda([["a", "b"], ["b", "a"]], ["a", "b"]), ["a", "b"])

    def test_bad_ids_refusals_and_errors_keep_the_base_order(self):
        messy = LLMReranker(client=fake_client(lambda kwargs: reply(["r4", "nope", "r4", "r2"])), permutations=1)
        ranked, _ = messy.rerank({}, self.items)
        self.assertEqual([i["item_id"] for i in ranked], ["r4", "r2", "r0", "r1", "r3"])
        refused, error = LLMReranker(client=fake_client(lambda kwargs: reply([], "refusal"))).rerank({}, self.items)
        self.assertEqual(refused, self.items)
        self.assertIn("no usable ranking", error)

        def boom(kwargs):
            raise ConnectionError("offline")
        failed, error = LLMReranker(client=fake_client(boom)).rerank({}, self.items)
        self.assertEqual(failed, self.items)
        self.assertIn("ConnectionError", error)

    def test_engine_serves_llm_as_its_own_attributable_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = synthetic_dataset("evanston", n_users=60, n_items=12)
            engine = Engine(ds, directory, llm_reranker=LLMReranker(client=fake_client(lambda kwargs: reply(candidates_in(kwargs)[::-1])), permutations=1))
            self.assertIn("llm", engine.models)
            base = engine.recommend({"cuisines": ["Thai", "Italian"]}, "baseline", 6)
            result = engine.recommend({"cuisines": ["Thai", "Italian"]}, "llm", 3)
            self.assertEqual(result["model"], "llm")
            self.assertTrue(result["model_version"].endswith("+llm:" + DEFAULT_MODEL))
            self.assertEqual([i["item_id"] for i in result["recommendations"]], [i["item_id"] for i in base["recommendations"]][::-1][:3])


if __name__ == "__main__":
    unittest.main()
