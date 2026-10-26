"""End-to-end on the real processed data: trained artifacts behind the HTTP API, attribution and isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
import uuid

from restaurant_recs.dataset import load_dataset
from restaurant_recs.engine import Engine
from restaurant_recs.geo import haversine_km
from restaurant_recs.server import create_server

ROOT = Path(__file__).resolve().parents[1]
READY = {"evanston": "prepared.json", "chicago": "interactions.npz"}


def available(name: str) -> bool:
    return (ROOT / "data/processed" / name / READY[name]).exists() and (ROOT / "artifacts" / name / "model_config.json").exists()


class WorkflowTests(unittest.TestCase):
    def check(self, name: str) -> None:
        artifacts = ROOT / "artifacts" / name
        paths = sorted(p for p in artifacts.iterdir() if p.is_file())
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        engine = Engine(load_dataset(name, ROOT / "data/processed" / name), artifacts)
        info = engine.info()
        spot = info["locations"][0]
        location = {"latitude": spot["latitude"], "longitude": spot["longitude"]}
        cuisine = info["cuisines"][0]
        session = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "events.sqlite3"
            server = create_server(engine, port=0, db_path=db)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:%d" % server.server_address[1]

            def post(route, body):
                with urlopen(Request(base + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=10) as r:
                    return r.status, json.load(r)
            try:
                for model in engine.models:
                    status, result = post("/api/recommend", {"session_id": session, "model": model, "k": 5, "preferences": {
                        "cuisines": [cuisine], "location": location, "max_distance_km": 10.}})
                    self.assertEqual(status, 200)
                    self.assertEqual(result["model"], model)
                    self.assertTrue(result["recommendations"], f"{name}/{model} returned nothing")
                    for item in result["recommendations"]:
                        self.assertEqual(item["cuisine"], cuisine)
                        self.assertFalse(item["closed"])
                        self.assertLessEqual(float(haversine_km(spot["latitude"], spot["longitude"], item["latitude"], item["longitude"])), 10.01)
                    item = result["recommendations"][0]
                    for event_type in ("click", "favorite"):
                        status, event = post("/api/events", {"session_id": session, "request_id": result["request_id"],
                                                             "item_id": item["item_id"], "type": event_type})
                        self.assertEqual(status, 201)
                        self.assertEqual(event["model_version"], result["model_version"])
                with sqlite3.connect(str(db)) as connection:
                    rows = connection.execute("SELECT model, count(*) FROM events GROUP BY model").fetchall()
                    self.assertEqual(dict(rows), {model: 2 for model in engine.models})
                with urlopen(base + "/api/metrics", timeout=10) as response:
                    public = json.load(response)
                self.assertIsNone(re.search(r"u_[0-9a-f]{64}", json.dumps(public)), "Public metrics must omit user IDs")
                self.assertNotIn("exact_duplicate_audit", public["quality"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    @unittest.skipUnless(available("evanston"), "Run prepare/train --dataset evanston first")
    def test_evanston(self):
        self.check("evanston")

    @unittest.skipUnless(available("chicago"), "Run download/prepare/train --dataset chicago first")
    def test_chicago(self):
        self.check("chicago")


if __name__ == "__main__":
    unittest.main()
