"""End-to-end training on synthetic data for both dataset profiles, then serving from the written artifacts."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from restaurant_recs.engine import Engine
from restaurant_recs.geo import haversine_km
from restaurant_recs.training import train

from support import FakeEncoder, synthetic_dataset


class ChicagoProfilePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.artifacts = Path(cls.temporary.name) / "chicago"
        cls.ds = synthetic_dataset("chicago", n_users=500)
        cls.report = train(cls.ds, cls.artifacts, quick=True, encoder=FakeEncoder(), log=lambda message: None)
        cls.engine = Engine(cls.ds, cls.artifacts)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_stage_is_reported_and_selection_uses_validation(self):
        report = self.report
        for name in ("popularity", "most_popular", "content_sentence", "content_tfidf", "content_ranked",
                     "two_tower_ranked", "sequential", "learned_ranker", "learned_ranker_no_geo"):
            self.assertIn(name, report["policies"])
        trials = report["two_tower"]["trials"]
        self.assertEqual(report["two_tower"]["served"], max(trials, key=lambda t: t["validation_ndcg"])["name"])
        backends = {b: report["policies"][f"content_{b}"]["validation"]["overall"]["ndcg_at_10"] for b in ("sentence", "tfidf")}
        self.assertEqual(report["text_backend_selected"], max(backends, key=backends.get))
        self.assertIsNone(report["policies"]["learned_ranker"]["validation"])  # fitted on validation, so not reported there
        for policy in report["policies"].values():
            self.assertTrue(0 <= policy["test"]["overall"]["ndcg_at_10"] <= 1)
        self.assertEqual(report["ann"][0]["method"], "exact")
        self.assertIn("share_of_test_targets_within_km", report["geo"])
        self.assertEqual(report["semantic_ids"]["items"], self.ds.n_items)
        self.assertEqual(report["served"]["learned"], "learned_ranker")
        selection = report["ranker"]["objective_selection"]
        self.assertEqual(set(selection), {"lambdarank", "binary", "rank_xendcg"})
        self.assertEqual(report["ranker"]["selected_objective"], max(selection, key=selection.get))

    def test_learned_models_beat_global_popularity_on_structured_data(self):
        policies = self.report["policies"]
        best = max(policies[name]["test"]["overall"]["ndcg_at_10"] for name in policies if name.startswith("two_tower["))
        self.assertGreater(best, policies["popularity"]["test"]["overall"]["ndcg_at_10"])

    def test_every_served_arm_respects_hard_constraints(self):
        location = {"latitude": self.ds.items[3]["latitude"], "longitude": self.ds.items[3]["longitude"]}
        preferences = {"cuisines": ["Italian", "Thai"], "location": location, "max_distance_km": 8., "max_price_level": 3}
        self.assertEqual(self.engine.models, ["baseline", "two_tower", "learned"])
        for model in self.engine.models:
            result = self.engine.recommend(preferences, model, 5)
            self.assertEqual(result["model"], model)
            for item in result["recommendations"]:
                self.assertIn(item["cuisine"], ("Italian", "Thai"))
                self.assertLessEqual(item["price_level"], 3)
                self.assertFalse(item["closed"])
                self.assertLessEqual(item["distance_km"], 8.)
                self.assertAlmostEqual(item["distance_km"], float(haversine_km(location["latitude"], location["longitude"],
                                                                               item["latitude"], item["longitude"])), places=1)
            scores = [item["ranking_score"] for item in result["recommendations"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_seed_restaurants_provide_a_location_and_are_excluded(self):
        seed = self.ds.items[5]["item_id"]
        result = self.engine.recommend({"seed_item_ids": [seed], "exclude_seen": True, "max_distance_km": 50.}, "learned", 5)
        self.assertEqual(result["location_source"], "seeds")
        self.assertNotIn(seed, [item["item_id"] for item in result["recommendations"]])
        with self.assertRaises(ValueError):
            self.engine.recommend({"max_distance_km": 2.}, "baseline", 5)  # no location at all

    def test_artifacts_refuse_a_different_dataset(self):
        with self.assertRaises(ValueError):
            Engine(replace(self.ds, fingerprint="something-else"), self.artifacts)

    def test_llm_arm_falls_back_when_not_enabled(self):
        result = self.engine.recommend({"cuisines": ["Thai"]}, "llm", 3)
        self.assertEqual(result["model"], "learned")
        self.assertIn("not enabled", result["fallback"])


class EvanstonProfilePipelineTests(unittest.TestCase):
    def test_small_catalog_profile_serves_baseline_and_two_tower(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = synthetic_dataset("evanston", n_users=150, n_items=20)
            report = train(ds, directory, quick=True, encoder=FakeEncoder(), log=lambda message: None)
            self.assertEqual(report["k"], 5)
            self.assertNotIn("learned_ranker", report["policies"])
            engine = Engine(ds, directory)
            self.assertEqual(engine.models, ["baseline", "two_tower"])
            fallback = engine.recommend({}, "learned", 3)
            self.assertEqual(fallback["model"], "two_tower")
            self.assertIsNotNone(fallback["fallback"])
            empty = engine.recommend({"cuisines": ["Thai"], "max_price_level": 1, "open_after_8pm": True,
                                      "location": {"latitude": 10., "longitude": 10.}, "max_distance_km": 1.}, "baseline", 3)
            self.assertEqual(empty["recommendations"], [])
            self.assertTrue(empty["message"])


class UntrainedEngineTests(unittest.TestCase):
    def test_baseline_works_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = synthetic_dataset("evanston", n_users=60, n_items=12)
            engine = Engine(ds, directory)
            self.assertEqual(engine.models, ["baseline"])
            self.assertEqual(engine.metrics()["status"], "not_trained")
            result = engine.recommend({"cuisines": ["Italian"]}, "two_tower", 3)
            self.assertEqual(result["model"], "baseline")
            self.assertTrue(all(item["cuisine"] == "Italian" for item in result["recommendations"]))
            self.assertEqual(engine.search("thai")[0]["cuisine"], "Thai")
            self.assertEqual(engine.search("  "), [])
            for bad in ({"budget": True}, {"cuisines": "Thai"}, {"seed_item_ids": ["missing"]}, {"exclude_seen": 1}):
                with self.assertRaises(ValueError):
                    engine.recommend(bad)
            with self.assertRaises(ValueError):
                engine.recommend({}, "baseline", 0)
            self.assertTrue(np.isfinite([item["score"] for item in result["recommendations"]]).all())


if __name__ == "__main__":
    unittest.main()
