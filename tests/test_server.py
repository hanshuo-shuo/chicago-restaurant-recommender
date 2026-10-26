"""HTTP contract checks, with a fake model to isolate service behavior."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from restaurant_recs.server import create_server


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.bad_result = False
        self.applied_model = None

    def catalog(self):
        return [
            {"item_id": "r_1", "name": "Example Cafe", "cuisine": "Cafe", "average_cost": 12, "open_after_8pm": False, "metadata_status": "catalog"},
            {"item_id": "r_2", "name": "Review Only", "cuisine": None, "average_cost": None, "open_after_8pm": None, "metadata_status": "review_only"},
        ]

    def metrics(self):
        return {"test": {"baseline_ranked": {"cold": {"queries": 3, "users": 3, "recall_at_5": .25, "ndcg_at_5": .15}}}}

    def info(self):
        return {"dataset": "test", "catalog_size": 2, "models": ["baseline"], "default_model": "baseline", "default_k": 5,
                "cuisines": ["Cafe"], "has_price_level": False, "has_budget": True, "locations": []}

    def search(self, text):
        return [{"item_id": i["item_id"], "name": i["name"]} for i in self.catalog() if text.casefold() in i["name"].casefold()]

    def recommend(self, preferences=None, model="baseline", k=5):
        self.calls.append({"preferences": preferences, "model": model, "k": k})
        rows = self.catalog()[:k]
        if preferences.get("budget") == 1:
            rows = []
        if self.bad_result:
            rows = [{"item_id": "foreign_item"}]
        return {"recommendations": rows, "candidate_count": len(rows), "model": self.applied_model or model, "model_version": "frozen-test-v1", "request_id": "engine-untrusted-id"}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.db = Path(self.temp.name) / "events.sqlite3"
        self.server = create_server(self.engine, port=0, db_path=self.db)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.session = str(uuid.uuid4())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, body=None, raw=None, headers=None):
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request_headers = {"Content-Type": "application/json"} if payload is not None else {}
        request_headers.update(headers or {})
        req = Request(self.url + path, data=payload, headers=request_headers)
        try:
            response = urlopen(req, timeout=3)
        except HTTPError as exc:
            response = exc
        content = response.read()
        result = content.decode() if "text/html" in response.headers.get("Content-Type", "") else (json.loads(content) if content else None)
        return response.status, result, response.headers

    def body(self, **changes):
        value = {"session_id": self.session, "model": "baseline", "k": 5, "preferences": {}}
        value.update(changes)
        return value

    def event(self, result, **changes):
        value = {"session_id": self.session, "request_id": result["request_id"], "item_id": "r_1", "type": "click"}
        value.update(changes)
        return value

    def test_static_page_and_read_only_routes(self):
        status, page, headers = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("Table for You", page)
        self.assertIn("Online metrics by model", page)
        self.assertNotRegex(page, "[\u4e00-\u9fff]")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("/api/catalog")[1], self.engine.catalog())
        self.assertEqual(self.request("/api/metrics")[1], self.engine.metrics())
        self.assertEqual(self.request("/api/health")[1]["log_counts"]["impressions"], 0)
        self.assertEqual(self.request("/not-found")[0], 404)

    def test_recommend_is_anonymous_and_logs_impressions(self):
        status, result, _ = self.request("/api/recommend", self.body(preferences={"cuisines": ["Cafe"], "budget": 15, "seed_item_ids": ["r_1"]}))
        self.assertEqual(status, 200)
        self.assertNotEqual(result["request_id"], "engine-untrusted-id")
        self.assertEqual(result["impressions_logged"], 2)
        self.assertGreaterEqual(result["latency_ms"], 0)
        with sqlite3.connect(str(self.db)) as connection:
            rows = connection.execute("SELECT session_id, position, model_version FROM impressions ORDER BY position").fetchall()
        self.assertEqual(rows, [(self.session, 1, "frozen-test-v1"), (self.session, 2, "frozen-test-v1")])

    def test_click_and_favorite_derive_impression_metadata(self):
        result = self.request("/api/recommend", self.body())[1]
        self.assertEqual(self.request("/api/events", self.event(result))[0], 201)
        self.assertEqual(self.request("/api/events", self.event(result, type="favorite"))[0], 201)
        with sqlite3.connect(str(self.db)) as connection:
            rows = connection.execute("SELECT type, position, model, model_version FROM events ORDER BY id").fetchall()
        self.assertEqual(rows, [("click", 1, "baseline", "frozen-test-v1"), ("favorite", 1, "baseline", "frozen-test-v1")])
        self.assertEqual(self.request("/api/health")[1]["log_counts"], {"impressions": 2, "clicks": 1, "favorites": 1})

    def test_event_must_match_original_request_session_and_item(self):
        result = self.request("/api/recommend", self.body())[1]
        for changes in ({"session_id": str(uuid.uuid4())}, {"request_id": str(uuid.uuid4())}, {"item_id": "unknown"}, {"type": "purchase"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.request("/api/events", self.event(result, **changes))[0], 400)
        self.assertEqual(self.request("/api/health")[1]["log_counts"]["clicks"], 0)

    def test_event_metadata_cannot_be_supplied_by_client(self):
        result = self.request("/api/recommend", self.body())[1]
        self.assertEqual(self.request("/api/events", self.event(result, model_version="fake", position=1))[0], 400)

    def test_fallback_logs_actual_applied_model(self):
        self.engine.applied_model = "baseline"
        result = self.request("/api/recommend", self.body(model="two_tower"))[1]
        self.assertEqual(self.request("/api/events", self.event(result))[0], 201)
        with sqlite3.connect(str(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT model FROM impressions LIMIT 1").fetchone()[0], "baseline")
            self.assertEqual(connection.execute("SELECT model FROM events LIMIT 1").fetchone()[0], "baseline")

    def test_online_metrics_attribute_ctr_per_model(self):
        self.assertEqual(self.request("/api/online-metrics")[1], {"by_model": []})
        baseline = self.request("/api/recommend", self.body())[1]
        self.request("/api/recommend", self.body(model="two_tower"))
        for event_type in ("click", "click", "favorite"):  # repeated clicks count once per impression
            self.assertEqual(self.request("/api/events", self.event(baseline, type=event_type))[0], 201)
        rows = {row["model"]: row for row in self.request("/api/online-metrics")[1]["by_model"]}
        self.assertEqual(rows["baseline"], {"model": "baseline", "requests": 1, "impressions": 2, "clicked_impressions": 1,
                                            "favorited_impressions": 1, "ctr": .5, "favorite_rate": .5})
        self.assertEqual((rows["two_tower"]["impressions"], rows["two_tower"]["ctr"]), (2, 0))

    def test_historical_identity_and_non_anonymous_sessions_rejected(self):
        self.assertEqual(self.request("/api/recommend", self.body(user_id="historical-reviewer"))[0], 400)
        for session in ("person@example.com", "reviewer_1", None, str(uuid.uuid1())):
            self.assertEqual(self.request("/api/recommend", self.body(session_id=session))[0], 400)
        self.assertEqual(len(self.engine.calls), 0)

    def test_k_and_model_validation(self):
        for k in (0, -1, 21, 1.5, True, "5", None):
            self.assertEqual(self.request("/api/recommend", self.body(k=k))[0], 400)
        for model in ("unknown", None, 12, {}):
            self.assertEqual(self.request("/api/recommend", self.body(model=model))[0], 400)
        self.assertEqual(self.request("/api/recommend", self.body(model="two_tower", k=1))[0], 200)

    def test_nonfinite_and_invalid_budgets(self):
        for budget in (float("nan"), float("inf"), float("-inf"), -1, 0, True, "20", 10001):
            self.assertEqual(self.request("/api/recommend", self.body(preferences={"budget": budget}))[0], 400)
        raw = json.dumps(self.body(preferences={"budget": 20})).replace('"budget": 20', '"budget": 1e309').encode()
        self.assertEqual(self.request("/api/recommend", raw=raw)[0], 400)
        self.assertEqual(len(self.engine.calls), 0)

    def test_preferences_checked_against_public_catalog(self):
        for preferences in ({"cuisines": ["Unknown"]}, {"seed_item_ids": ["unknown"]}, {"seed_item_ids": "r_1"}, {"cuisines": [12]}, {"open_after_8pm": "yes"}, {"exclude_seen": 1}, {"demographics": "anything"}, []):
            self.assertEqual(self.request("/api/recommend", self.body(preferences=preferences))[0], 400)
        self.assertEqual(self.request("/api/recommend", self.body(preferences={"seed_item_ids": ["r_2"]}))[0], 200)

    def test_malformed_json_content_type_and_oversized_payload(self):
        self.assertEqual(self.request("/api/recommend", raw=b"[]")[0], 400)
        self.assertEqual(self.request("/api/recommend", raw=b"{broken")[0], 400)
        self.assertEqual(self.request("/api/recommend", self.body(), headers={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.request("/api/recommend", raw=b"x" * 17000)[0], 400)

    def test_cross_origin_writes_rejected(self):
        self.assertEqual(self.request("/api/recommend", self.body(), headers={"Origin": "https://unrelated.example"})[0], 400)
        self.assertEqual(len(self.engine.calls), 0)

    def test_no_results_valid_without_fabricated_impressions(self):
        status, result, _ = self.request("/api/recommend", self.body(preferences={"budget": 1}))
        self.assertEqual(status, 200)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["impressions_logged"], 0)
        self.assertEqual(self.request("/api/health")[1]["log_counts"]["impressions"], 0)

    def test_model_cannot_log_unrecognized_items(self):
        self.engine.bad_result = True
        self.assertEqual(self.request("/api/recommend", self.body())[0], 500)
        self.assertEqual(self.request("/api/health")[1]["log_counts"]["impressions"], 0)

    def test_info_and_search_routes(self):
        self.assertEqual(self.request("/api/info")[1]["dataset"], "test")
        self.assertEqual([r["item_id"] for r in self.request("/api/search?q=cafe")[1]["results"]], ["r_1"])

    def test_location_distance_and_price_validation(self):
        good = {"location": {"latitude": 41.9, "longitude": -87.6}, "max_distance_km": 3, "max_price_level": 2, "include_closed": False}
        self.assertEqual(self.request("/api/recommend", self.body(preferences=good))[0], 200)
        self.assertEqual(self.engine.calls[-1]["preferences"]["location"], {"latitude": 41.9, "longitude": -87.6})
        for bad in ({"location": {"latitude": 91, "longitude": 0}}, {"location": {"lat": 1, "lon": 2}}, {"location": [41.9, -87.6]},
                    {"max_distance_km": 3}, {"location": {"latitude": 41.9, "longitude": -87.6}, "max_distance_km": 0},
                    {"max_price_level": 5}, {"max_price_level": 1.5}, {"include_closed": "no"}):
            with self.subTest(preferences=bad):
                self.assertEqual(self.request("/api/recommend", self.body(preferences=bad))[0], 400)
        self.assertEqual(self.request("/api/recommend", self.body(preferences={"seed_item_ids": ["r_1"], "max_distance_km": 3}))[0], 200)
        for model in ("learned", "llm"):
            self.assertEqual(self.request("/api/recommend", self.body(model=model))[0], 200)

    def test_request_context_is_logged_for_learning_from_feedback(self):
        result = self.request("/api/recommend", self.body(preferences={"cuisines": ["Cafe"]}))[1]
        with sqlite3.connect(str(self.db)) as connection:
            row = connection.execute("SELECT session_id, model, model_version, preferences FROM requests WHERE request_id=?", (result["request_id"],)).fetchone()
        self.assertEqual(row[:3], (self.session, "baseline", "frozen-test-v1"))
        self.assertEqual(json.loads(row[3])["cuisines"], ["Cafe"])

    def test_v1_event_database_is_migrated_in_place(self):
        legacy = Path(self.temp.name) / "legacy.sqlite3"
        with sqlite3.connect(str(legacy)) as connection:
            connection.execute("CREATE TABLE impressions (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, session_id TEXT NOT NULL, item_id TEXT NOT NULL, position INTEGER NOT NULL, model TEXT NOT NULL, model_version TEXT NOT NULL, timestamp TEXT NOT NULL, UNIQUE(request_id, session_id, item_id))")
            connection.execute("INSERT INTO impressions (request_id, session_id, item_id, position, model, model_version, timestamp) VALUES ('old', 's', 'r_1', 1, 'baseline', 'v1', 't')")
        server = create_server(self.engine, port=0, db_path=legacy)
        server.server_close()
        with sqlite3.connect(str(legacy)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(impressions)")}
            self.assertTrue({"retrieval_score", "ranking_score", "distance_km"} <= columns)
            self.assertEqual(connection.execute("SELECT count(*) FROM impressions").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
