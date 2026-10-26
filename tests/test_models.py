from __future__ import annotations

import unittest

import numpy as np
import torch

from restaurant_recs.ann import IVFIndex, benchmark
from restaurant_recs.dataset import Histories, make_queries, temporal_split
from restaurant_recs.evaluation import evaluate, ranking_metrics, top_k
from restaurant_recs.features import ItemFeatures, item_text
from restaurant_recs.geo import GeoEncoder, haversine_km, item_coordinates, user_location
from restaurant_recs.ranking import eligible_mask, ranking_scores
from restaurant_recs.retrieval import two_tower_inputs, two_tower_scores
from restaurant_recs.semantic_ids import prefix_ids, rq_kmeans
from restaurant_recs.sequential import SequentialRecommender, build_sequences, sequential_scores
from restaurant_recs.two_tower import TwoTower, build_examples

from support import FakeEncoder, synthetic_dataset, synthetic_items


class SplitAndQueryTests(unittest.TestCase):
    def setUp(self):
        self.ds = synthetic_dataset(n_users=120)
        self.train_end, self.val_end = temporal_split(self.ds.day)

    def test_split_is_roughly_80_10_10_with_three_nonempty_partitions(self):
        train = (self.ds.day <= self.train_end).mean()
        validation = ((self.ds.day > self.train_end) & (self.ds.day <= self.val_end)).mean()
        self.assertAlmostEqual(train, .8, delta=.08)
        self.assertGreater(validation, 0)
        self.assertGreater((self.ds.day > self.val_end).mean(), 0)
        with self.assertRaises(ValueError):
            temporal_split(np.array([1, 1, 2]))

    def test_history_is_strictly_before_the_query_day(self):
        histories = Histories(self.ds)
        queries = make_queries(self.ds, self.val_end, int(self.ds.day.max()), histories)
        self.assertGreater(len(queries), 0)
        for q in range(len(queries)):
            user, day = queries.user[q], queries.day[q]
            expected = (self.ds.user == user) & (self.ds.day < day)
            self.assertEqual(len(queries.history_items[q]), int(expected.sum()))
            same_day = self.ds.item[(self.ds.user == user) & (self.ds.day == day) & (self.ds.rating >= 4)]
            np.testing.assert_array_equal(queries.targets[q], np.unique(same_day))

    def test_frozen_histories_ignore_events_after_the_cutoff(self):
        frozen = Histories(self.ds, cutoff=self.train_end)
        for user in range(20):
            items, _ = frozen.before(user, 10**9)
            self.assertEqual(len(items), int(((self.ds.user == user) & (self.ds.day <= self.train_end)).sum()))

    def test_query_sampling_keeps_whole_users(self):
        histories = Histories(self.ds)
        full = make_queries(self.ds, self.train_end, int(self.ds.day.max()), histories)
        sample = make_queries(self.ds, self.train_end, int(self.ds.day.max()), histories, max_queries=30, seed=3)
        self.assertLess(len(sample), len(full))
        for user in np.unique(sample.user):
            self.assertEqual(int((sample.user == user).sum()), int((full.user == user).sum()))


class MetricTests(unittest.TestCase):
    def test_top_k_matches_stable_argsort_with_ties(self):
        rng = np.random.default_rng(0)
        scores = rng.integers(0, 4, size=(50, 30)).astype(float)
        np.testing.assert_array_equal(top_k(scores, 7), np.argsort(-scores, axis=1, kind="stable")[:, :7])

    def test_recall_ndcg_and_scorer_contract(self):
        recall, ndcg = ranking_metrics(["a", "b", "c"], ["a", "c"], 2)
        self.assertEqual(recall, .5)
        self.assertAlmostEqual(ndcg, 1 / (1 + 1 / np.log2(3)))
        ds = synthetic_dataset(n_users=60)
        queries = make_queries(ds, *temporal_split(ds.day), Histories(ds))
        with self.assertRaises(ValueError):
            evaluate(queries, ds.n_items, lambda q: np.zeros((len(q), ds.n_items - 1)), 5)
        with self.assertRaises(ValueError):
            evaluate(queries, ds.n_items, lambda q: np.full((len(q), ds.n_items), np.nan), 5)
        oracle = evaluate(queries, ds.n_items, lambda q: np.array([np.isin(np.arange(ds.n_items), t) for t in q.targets], float), 5)
        self.assertEqual(oracle["overall"]["ndcg_at_5"], 1.0)


class FeatureAndGeoTests(unittest.TestCase):
    def test_sentence_backend_roundtrips_from_saved_matrix(self):
        items = synthetic_items(12)
        fitted = ItemFeatures(items, text_backend="sentence", encoder=FakeEncoder())
        restored = ItemFeatures(items, fitted.config, fitted.text_matrix)
        np.testing.assert_allclose(restored.matrix, fitted.matrix)
        self.assertIn(items[0]["description"], item_text(items[0]))
        with self.assertRaises(ValueError):
            ItemFeatures(items, fitted.config)  # serving without the saved text matrix

    def test_tfidf_min_df_and_query_from_seeds_and_cuisines(self):
        items = synthetic_items(12)
        f = ItemFeatures(items, min_df=3)
        self.assertTrue(all(token in f.config["vocabulary"] for token in ("place", "spot")))
        query = f.query([], {"seed_item_ids": [items[0]["item_id"]], "cuisines": ["Thai"]})
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.)
        self.assertEqual(float(np.linalg.norm(f.query([], {}))), 0.)

    def test_geo_distance_encoder_and_user_location(self):
        self.assertAlmostEqual(float(haversine_km(41.8837, -87.6289, 42.0471, -87.6835)), 18.7, delta=.3)
        coords = item_coordinates(synthetic_items(30))
        geo = GeoEncoder(coords, 6, seed=0)
        a, b, far = geo.encode(np.array([coords[0], coords[0] + [.001, .001], [41.0, -88.5]]))
        self.assertGreater(a @ b, a @ far)
        self.assertTrue(np.all(geo.encode(np.array([[np.nan, np.nan]])) == 0))
        location = user_location(np.array([0, 1, 2]), np.array([5, 5, 1]), coords)
        np.testing.assert_allclose(location, np.median(coords[[0, 1]], axis=0))
        self.assertTrue(np.isnan(user_location(np.array([], int), np.array([], int), coords)).all())


class SemanticIdAndAnnTests(unittest.TestCase):
    def test_rq_kmeans_codes_and_dense_prefixes(self):
        x = np.random.default_rng(0).normal(size=(200, 8))
        codes, stats = rq_kmeans(x, levels=3, codebook_size=4, seed=0)
        self.assertEqual(codes.shape, (200, 3))
        self.assertTrue(stats["relative_reconstruction_error"][0] > stats["relative_reconstruction_error"][-1])
        prefixes = prefix_ids(codes, 4)
        for level in range(3):
            self.assertEqual(prefixes[:, level].max() + 1, len(np.unique(codes[:, :level + 1], axis=0)))
        same = np.all(codes[:, :2] == codes[0, :2], axis=1)
        self.assertTrue(np.all(prefixes[same, 1] == prefixes[0, 1]))

    def test_ivf_with_all_lists_probed_matches_exact(self):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(300, 16))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = IVFIndex(embeddings, rng.normal(size=300) * .1, .1, n_lists=10, seed=0)
        queries = rng.normal(size=(20, 16))
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        rows = benchmark(index, queries, 10, (1, 10))
        self.assertEqual(rows[-1]["recall_vs_exact"], 1.0)
        self.assertLess(rows[1]["items_scanned"], rows[-1]["items_scanned"])


class TwoTowerTests(unittest.TestCase):
    def test_examples_use_only_strictly_earlier_positive_history(self):
        ds = synthetic_dataset(n_users=50)
        train_end = temporal_split(ds.day)[0]
        coords = item_coordinates(ds.items)
        examples = build_examples(ds, train_end, coords, None)
        hist = Histories(ds, cutoff=train_end)
        row = 0
        for user in range(ds.n_users):
            lo, hi = hist.ptr[user], hist.ptr[user + 1]
            items, days, ratings = hist.item[lo:hi], hist.day[lo:hi], hist.rating[lo:hi]
            for day in np.unique(days[ratings >= 4]):
                for target in np.unique(items[(days == day) & (ratings >= 4)]):
                    history = examples.history[row][examples.history[row] < ds.n_items]
                    np.testing.assert_array_equal(history, items[(days < day) & (ratings >= 4)][-30:])
                    self.assertEqual(examples.target[row], target)
                    self.assertNotIn(target, examples.known[row])
                    row += 1
        self.assertEqual(row, len(examples))

    def test_numpy_serving_matches_torch_and_gradient_is_correct(self):
        torch.manual_seed(0)
        items = np.random.default_rng(0).normal(size=(9, 6))
        prefixes = np.stack([np.arange(9) % 3, np.arange(9) % 5, np.arange(9)], axis=1)
        model = TwoTower(items, 7, dim=4, hidden=5, temperature=.2, extra="sid", prefixes=prefixes).double()
        with torch.no_grad():
            model.item_bias.normal_()
        x = np.random.default_rng(1).normal(size=(3, 7))
        expected = (model.queries(torch.tensor(x)) @ model.items().T / .2 + model.item_bias).detach().numpy()
        np.testing.assert_allclose(two_tower_scores(model.export(), x), expected, rtol=1e-6, atol=1e-8)
        target = torch.tensor([2])
        loss = torch.nn.functional.cross_entropy(model.queries(torch.tensor(x[:1])) @ model.items().T / .2 + model.item_bias, target)
        loss.backward()
        weight = model.query_tower.layers[0].weight
        analytical, original, eps = weight.grad[0, 0].item(), weight[0, 0].item(), 1e-6
        values = []
        with torch.no_grad():
            for delta in (eps, -eps):
                weight[0, 0] = original + delta
                values.append(torch.nn.functional.cross_entropy(model.queries(torch.tensor(x[:1])) @ model.items().T / .2 + model.item_bias, target).item())
            weight[0, 0] = original
        self.assertAlmostEqual(analytical, (values[0] - values[1]) / (2 * eps), places=5)

    def test_query_inputs_include_geo_and_bias(self):
        coords = item_coordinates(synthetic_items(20))
        geo = GeoEncoder(coords, 4, seed=0)
        x = two_tower_inputs(np.ones((2, 3)), np.array([coords[0], [np.nan, np.nan]]), geo)
        self.assertEqual(x.shape, (2, 3 + geo.dim + 1))
        self.assertTrue(np.all(x[1, 3:3 + geo.dim] == 0))
        self.assertTrue(np.all(x[:, -1] == 1))


class SequentialTests(unittest.TestCase):
    def test_windows_and_causality(self):
        ds = synthetic_dataset(n_users=40)
        inputs, targets = build_sequences(ds, temporal_split(ds.day)[0], max_len=4)
        self.assertEqual(inputs.shape[1], 4)
        self.assertTrue(np.all((targets == -1) | (targets < ds.n_items)))
        torch.manual_seed(0)
        model = SequentialRecommender(np.random.default_rng(0).normal(size=(ds.n_items, 6)), dim=8, layers=1, heads=2,
                                      max_len=4, dropout=0.).eval()
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 9, 7]])
        with torch.no_grad():
            torch.testing.assert_close(model(a)[0, :2], model(b)[0, :2])  # later items never influence earlier positions
            self.assertFalse(torch.allclose(model(a)[0, 3], model(b)[0, 3]))

    def test_cold_queries_fall_back(self):
        ds = synthetic_dataset(n_users=40)
        queries = make_queries(ds, *temporal_split(ds.day), Histories(ds))
        model = SequentialRecommender(np.random.default_rng(0).normal(size=(ds.n_items, 6)), dim=8, layers=1, heads=2, max_len=5).eval()
        fallback = np.linspace(0, 1, ds.n_items)
        scores = sequential_scores(model, queries, fallback)
        for i, history in enumerate(queries.history_items):
            if len(history) == 0:
                np.testing.assert_array_equal(scores[i], fallback)


class ConstraintTests(unittest.TestCase):
    def test_hard_filters_never_match_unknown_metadata(self):
        items = synthetic_items(30)
        coords = item_coordinates(items)
        mask = eligible_mask(items, {"max_price_level": 2, "open_after_8pm": True}, set())
        for item, ok in zip(items, mask):
            if ok:
                self.assertTrue(item["price_level"] is not None and item["price_level"] <= 2)
                self.assertIs(item["open_after_8pm"], True)
                self.assertFalse(item["closed"])
        self.assertFalse(any(m for m, it in zip(eligible_mask(items, {}, set()), items) if it["closed"]))
        near = eligible_mask(items, {"max_distance_km": 2.}, set(), coords[0], coords)
        self.assertTrue(np.all(haversine_km(coords[0, 0], coords[0, 1], coords[near, 0], coords[near, 1]) <= 2.))
        with self.assertRaises(ValueError):
            eligible_mask(items, {"max_distance_km": 2.}, set(), np.array([np.nan, np.nan]), coords)

    def test_blend_is_row_wise(self):
        retrieval = np.array([[0., 2., 1.], [10., 30., 20.]])
        np.testing.assert_allclose(ranking_scores(retrieval, np.array([5., 1., 3.])), [[.15, .85, .5], [.15, .85, .5]])
        np.testing.assert_allclose(ranking_scores(np.ones(3), np.ones(3)), 0)


if __name__ == "__main__":
    unittest.main()
