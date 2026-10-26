"""Google Local ingestion: filtering, parsing, dedupe and k-core on tiny synthetic raw files."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from restaurant_recs.dataset import load_dataset
from restaurant_recs.google_local import _closing_minutes, _item, _open_after_8pm, _price_level, prepare_google_local


def meta(gmap_id, name, city, categories, **extra):
    row = {"gmap_id": gmap_id, "name": name, "address": f"{name}, 1 Main St, {city}, IL 60601",
           "latitude": 41.9, "longitude": -87.65, "category": categories, "price": None, "hours": None,
           "MISC": None, "state": None, "description": None, "avg_rating": 4.9, "num_of_reviews": 999}
    row.update(extra)
    return row


def ms(day: str) -> int:
    return int(np.datetime64(day, "ms").astype(np.int64))


class ParsingTests(unittest.TestCase):
    def test_hours_price_and_primary_cuisine(self):
        self.assertEqual(_closing_minutes("11AM–10PM"), 22 * 60)
        self.assertEqual(_closing_minutes("11AM–2:30PM, 5–9PM"), 21 * 60)
        self.assertEqual(_closing_minutes("5PM–2AM"), 26 * 60)
        self.assertEqual(_closing_minutes("7AM–12PM"), 12 * 60)
        self.assertIsNone(_closing_minutes("Closed"))
        self.assertTrue(_open_after_8pm([["Monday", "Closed"], ["Friday", "11AM–12AM"]]))
        self.assertFalse(_open_after_8pm([["Monday", "7AM–3PM"]]))
        self.assertIsNone(_open_after_8pm([["Monday", "Closed"]]))
        self.assertEqual([_price_level(p) for p in ("$", "$$$$", "₩₩", None, "$$$$$", "cheap")], [1, 4, 2, None, None, None])
        item = _item(meta("g1", "Spot", "Chicago", ["Restaurant", "Thai restaurant"],
                          MISC={"Atmosphere": ["Cozy"], "Health & safety": ["Mask required"]}, state="Permanently closed"), "Chicago")
        self.assertEqual(item["cuisine"], "Thai restaurant")
        self.assertEqual(item["attributes"], ["Cozy"])
        self.assertTrue(item["closed"])
        self.assertNotIn("avg_rating", item)  # all-time aggregates would leak the future


class PrepareTests(unittest.TestCase):
    def test_filters_dedupes_and_k_cores_without_keeping_raw_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            businesses = [
                meta("pizza", "Pizza A", "Chicago", ["Pizza restaurant"], price="$$", hours=[["Friday", "11AM–11PM"]]),
                meta("pizza", "Pizza A duplicate row", "Chicago", ["Pizza restaurant"]),
                meta("cafe", "Cafe B", "Evanston", ["Coffee shop"], price="₩"),
                meta("thai", "Thai C", "Chicago", ["Thai restaurant"], state="Permanently closed"),
                meta("toys", "Toy store", "Chicago", ["Toy store"]),
                meta("suburb", "Diner D", "Naperville", ["Diner"]),
                meta("rare", "Rare E", "Chicago", ["Mexican restaurant"]),
            ]
            with gzip.open(raw / "meta-Illinois.json.gz", "wt") as handle:
                handle.writelines(json.dumps(b) + "\n" for b in businesses)
            ratings = [("pizza", "111", 5, "2017-01-01"), ("cafe", "111", 4, "2017-02-01"), ("thai", "111", 2, "2017-03-01"),
                       ("pizza", "222", 3, "2017-01-05"), ("pizza", "222", 5, "2017-06-05"),  # repeat: the later one wins
                       ("cafe", "222", 5, "2017-02-05"), ("thai", "222", 4, "2017-03-05"),
                       ("pizza", "333", 4, "2018-01-01"), ("thai", "333", 5, "2018-02-01"), ("cafe", "333", 5, "2018-03-01"),
                       ("toys", "111", 5, "2017-04-01"), ("suburb", "222", 5, "2017-04-01"),   # out of scope businesses
                       ("pizza", "444", 5, "2015-06-01"), ("cafe", "444", 5, "2015-07-01"),     # before the window
                       ("rare", "555", 5, "2017-05-01")]                                        # removed by the k-core
            with gzip.open(raw / "rating-Illinois.csv.gz", "wt") as handle:
                handle.write("business,user,rating,timestamp\n")
                handle.writelines(f"{b},{u},{r},{ms(d)}\n" for b, u, r, d in ratings)
            out = Path(directory) / "processed"
            quality = prepare_google_local(raw, out, start="2016-01-01", end="2019-12-31", min_user=2, min_item=2)

            self.assertEqual(quality["funnel"]["duplicate_metadata_rows"], 1)
            self.assertEqual(quality["funnel"]["chicago_evanston_restaurants"], 4)
            self.assertEqual(quality["funnel"]["repeat_user_restaurant_ratings_collapsed"], 1)
            self.assertEqual(quality["catalog_items"], 3)
            self.assertEqual(quality["unique_users"], 3)
            self.assertEqual(quality["permanently_closed_items"], 1)
            ds = load_dataset("chicago", out)
            by_id = {it["item_id"]: it for it in ds.items}
            self.assertEqual(set(by_id), {"pizza", "cafe", "thai"})
            self.assertEqual(by_id["pizza"]["price_level"], 2)
            self.assertTrue(by_id["pizza"]["open_after_8pm"])
            self.assertEqual(by_id["cafe"]["city"], "Evanston")
            pizza = ds.index["pizza"]
            self.assertEqual(sorted(ds.rating[ds.item == pizza].tolist()), [4, 5, 5])  # 222's later rating (5) kept
            self.assertTrue(np.all(np.diff(ds.day) >= 0))
            stored = (out / "interactions.npz").read_bytes() + (out / "items.json").read_bytes()
            for raw_id in (b"111", b"222", b"333"):
                self.assertNotIn(raw_id, stored)


if __name__ == "__main__":
    unittest.main()
