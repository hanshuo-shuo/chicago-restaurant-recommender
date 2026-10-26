"""Offline pipeline: features -> retrievers -> rankers, every choice made on validation, test scored once.

For each dataset profile (``config.PROFILES``) this trains and evaluates:
popularity baselines, content retrieval per text backend, two-tower retrieval
(loss ablation, then ID / Semantic-ID item identity), a sequential Transformer,
the fixed blend ranker, a LightGBM LambdaRank ranker (with and without the
distance feature), and an IVF ANN index; then writes the served artifacts.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .ann import IVFIndex, benchmark
from .config import PROFILES, SEED
from .data import public_quality
from .dataset import Dataset, Histories, Queries, make_queries, temporal_split
from .evaluation import evaluate
from .features import ItemFeatures
from .geo import GeoEncoder, haversine_km, item_coordinates
from .ranker import FEATURES, GEO_FEATURES, ItemStats, LearnedRanker, candidates, features as ranker_features, source_scores
from .ranking import RANKING_POLICY, ranking_scores
from .retrieval import (content_batch, content_queries, interaction_counts, popularity_scores, query_locations,
                        save_two_tower, two_tower_inputs, two_tower_scores)
from .semantic_ids import prefix_ids, rq_kmeans

KNOWN_LIMITATIONS = {
    "evanston": ["Most users have a single rating, so most evaluation queries are cold-start; warm-user metrics rest on small samples.",
                 "The catalog is a single snapshot; item availability at each rating date is unknown.",
                 "User identities are name-based pseudonyms, not verified account IDs."],
    "chicago": ["Users and items are filtered to a 5/10-core, so cold-start is under-represented relative to production traffic.",
                "Catalog metadata (categories, price, hours, attributes) is a 2021 snapshot applied to 2016-2019 interactions.",
                "A Google review is a proxy for a visit; restaurants a user visited without reviewing are unobserved, not negatives.",
                "Evaluation uses a seeded sample of users per split (see protocol.max_queries)."],
}


def _quick(profile: dict) -> dict:
    """Small settings for smoke runs and tests; results are not meaningful."""
    profile.update(max_queries=1500, validation_sample=600)
    profile["two_tower"].update(max_epochs=2, losses=profile["two_tower"]["losses"][:1],
                                extras=tuple(e for e in profile["two_tower"]["extras"] if e in (None, "sid")))
    if profile["sequential"]:
        profile["sequential"].update(max_epochs=1, extras=(None,))
    if profile["ann"]:
        profile["ann"]["queries"] = 200
    return profile


def train(ds: Dataset, artifact_dir: str | Path, quick: bool = False, encoder=None,
          log: Callable[[str], None] = lambda message: print(message, flush=True)) -> dict:
    started = time.perf_counter()
    profile = copy.deepcopy(PROFILES[ds.name])
    if quick:
        profile = _quick(profile)
    k, n = profile["k"], ds.n_items
    metric = f"ndcg_at_{k}"
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_end, val_end = temporal_split(ds.day)
    in_train = ds.day <= train_end
    histories = Histories(ds)
    validation = make_queries(ds, train_end, val_end, histories, profile["max_queries"], seed=1)
    test = make_queries(ds, val_end, int(ds.day.max()), histories, profile["max_queries"], seed=2)
    tune = validation
    if profile["validation_sample"] and len(validation) > profile["validation_sample"]:
        rows = np.sort(np.random.default_rng(SEED).choice(len(validation), profile["validation_sample"], replace=False))
        tune = validation.subset(rows)
    log(f"[{ds.name}] {n} items, {ds.n_users} users, {len(ds.day)} interactions; train <= {ds.date(train_end)} "
        f"< validation <= {ds.date(val_end)} < test; queries: validation {len(validation)}, test {len(test)}")

    coords = item_coordinates(ds.items)
    popularity = popularity_scores(n, ds.item[in_train], ds.rating[in_train])
    counts = interaction_counts(n, ds.item[in_train])
    seen = counts[counts > 0]
    groups = {"new_item": counts == 0, "tail_item": (counts > 0) & (counts < (np.median(seen) if len(seen) else 0))}
    results: dict[str, dict] = {}

    def score(name: str, scorer: Callable[[Queries], np.ndarray], fitted_on_validation: bool = False) -> None:
        results[name] = {"validation": None if fitted_on_validation else evaluate(validation, n, scorer, k),
                         "test": evaluate(test, n, scorer, k, uncertainty=True, item_groups=groups)}
        val = results[name]["validation"]
        log(f"  {name:<34} validation NDCG@{k} {val['overall'][metric] if val else float('nan'):.4f}   "
            f"test NDCG@{k} {results[name]['test']['overall'][metric]:.4f}")

    def tune_ndcg(scorer: Callable[[Queries], np.ndarray]) -> float:
        return evaluate(tune, n, scorer, k)["overall"][metric]

    log("baselines")
    score("popularity", lambda q: np.tile(popularity, (len(q), 1)))
    if profile["ranker"]:
        most_popular = np.log1p(counts) + 1e-3 * popularity
        score("most_popular", lambda q: np.tile(most_popular, (len(q), 1)))

    log("content retrieval")
    features_by_backend = {}
    for backend in profile["text_backends"]:
        f = ItemFeatures(ds.items, text_backend=backend, encoder=encoder, min_df=profile["tfidf_min_df"],
                         min_category_items=profile["min_category_items"])
        features_by_backend[backend] = f
        score(f"content_{backend}", lambda q, f=f: content_batch(f, q, popularity))
    backend = max(profile["text_backends"], key=lambda b: results[f"content_{b}"]["validation"]["overall"][metric])
    features = features_by_backend[backend]
    features_by_backend.clear()
    log(f"  text backend selected on validation: {backend}")

    geo = GeoEncoder(coords, profile["geo_anchors"], seed=SEED) if profile["geo_anchors"] else None
    item_inputs = np.concatenate([features.matrix, geo.encode(coords)], axis=1) if geo else features.matrix

    def two_tower_scorer(weights: dict) -> Callable[[Queries], np.ndarray]:
        return lambda q: two_tower_scores(weights, two_tower_inputs(content_queries(features, q), query_locations(q, coords), geo))

    sid_stats, prefixes = None, None
    if profile["semantic_ids"]:
        log("semantic IDs (RQ-Kmeans)")
        codes, sid_stats = rq_kmeans(features.matrix, seed=SEED, **profile["semantic_ids"])
        prefixes = prefix_ids(codes, profile["semantic_ids"]["codebook_size"])
        np.save(artifact_dir / "semantic_ids.npy", codes)
        log(f"  {sid_stats['unique_ids']} unique IDs for {sid_stats['items']} items; utilization {sid_stats['codebook_utilization']}")

    log("two-tower retrieval")
    from .two_tower import build_examples, train_two_tower
    tt = profile["two_tower"]
    examples = build_examples(ds, train_end, coords, geo)
    target_counts = np.bincount(examples.target, minlength=n)
    trials, weights_by_name = [], {}

    def run_two_tower(base: dict, loss: str, extra: str | None, tag: str = "") -> dict:
        name = f"two_tower[{loss}{'+' + extra if extra else ''}]{tag}"
        log(f"  {name} ({len(examples)} training positives)")
        config = {**base, "loss": loss, "extra": extra, **{key: tt[key] for key in ("batch_size", "max_epochs", "validate_every", "patience")}}
        weights, trial = train_two_tower(examples, features.matrix, item_inputs, config,
                                         lambda w: tune_ndcg(two_tower_scorer(w)), target_counts, prefixes, log=log)
        trial.update(name=name, tag=tag)
        trials.append(trial)
        weights_by_name[name] = weights
        score(name, two_tower_scorer(weights))
        return trial

    tags = [f"#{i + 1}" if len(tt["search"]) > 1 else "" for i in range(len(tt["search"]))]
    loss_trials = [run_two_tower(base, loss, None, tag) for base, tag in zip(tt["search"], tags) for loss in tt["losses"]]
    best_loss = max(loss_trials, key=lambda t: t["validation_ndcg"])
    for extra in tt["extras"]:
        if extra is not None:
            run_two_tower({key: best_loss["configuration"][key] for key in tt["search"][0]}, best_loss["configuration"]["loss"], extra, best_loss["tag"])
    served_trial = max(trials, key=lambda t: t["validation_ndcg"])
    served_weights = weights_by_name[served_trial["name"]]
    log(f"  served two-tower selected on validation: {served_trial['name']}")
    examples = None  # release ~0.7 GB of training rows before the sequential stage

    sequential_trials = []
    if profile["sequential"]:
        log("sequential Transformer")
        import torch
        from .sequential import build_sequences, sequential_scores, train_sequential
        sp = profile["sequential"]
        inputs, targets = build_sequences(ds, train_end, sp["max_len"])
        best_state = None
        for extra in sp["extras"]:
            name = f"sequential{'[+' + extra + ']' if extra else ''}"
            log(f"  {name} ({len(inputs)} training windows)")
            model, trial = train_sequential(inputs, targets, item_inputs, {**sp, "extra": extra},
                                            lambda m: tune_ndcg(lambda q: sequential_scores(m, q, popularity)),
                                            prefixes, log=log)
            trial["name"] = name
            sequential_trials.append(trial)
            score(name, lambda q, m=model: sequential_scores(m, q, popularity))
            if best_state is None or trial["validation_ndcg"] > best_state[0]:
                best_state = (trial["validation_ndcg"], name, model.state_dict())
        torch.save({"name": best_state[1], "state_dict": best_state[2]}, artifact_dir / "sequential.pt")

    log("ranking")
    score("content_ranked", lambda q: ranking_scores(content_batch(features, q, popularity), popularity))
    tt_scorer = two_tower_scorer(served_weights)
    score("two_tower_ranked", lambda q: ranking_scores(tt_scorer(q), popularity))

    ranker_report = None
    if profile["ranker"]:
        stats = ItemStats.build(ds.items, ds.item, ds.day, train_end, popularity, coords)
        sizes = profile["ranker"]["sources"]
        popular = np.log1p(counts) + 1e-3 * popularity

        def rows_for(queries: Queries) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            cands, xs, labels = [], [], []
            for start in range(0, len(queries), 1024):
                batch = queries.subset(np.arange(start, min(start + 1024, len(queries))))
                content, two_tower = content_batch(features, batch, popularity), tt_scorer(batch)
                locations = query_locations(batch, coords)
                for j in range(len(batch)):
                    cand = candidates(source_scores(content[j], two_tower[j], popular, locations[j], coords), sizes)
                    cands.append(cand)
                    xs.append(ranker_features(cand, content[j], two_tower[j], batch.history_items[j],
                                              batch.history_ratings[j], locations[j], stats))
                    labels.append(np.isin(cand, batch.targets[j]).astype(float))
            return cands, xs, labels

        log("  learned ranker: candidate features")
        _, x_val, y_val = rows_for(validation)
        users = np.unique(validation.user)
        held_out = np.isin(validation.user, np.random.default_rng(SEED).choice(users, len(users) // 5, replace=False))
        useful = np.array([y.sum() > 0 for y in y_val])

        def pack(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            idx = np.flatnonzero(mask & useful)
            return (np.vstack([x_val[i] for i in idx]), np.concatenate([y_val[i] for i in idx]),
                    np.array([len(y_val[i]) for i in idx]))
        recall_ceiling = float(np.mean([y.sum() / len(t) for y, t in zip(y_val, validation.targets)]))
        ranker_report = {"candidates_per_query": round(float(np.mean([len(y) for y in y_val])), 1),
                         "candidate_recall_on_validation": round(recall_ceiling, 4),
                         "training_queries": int((~held_out & useful).sum()), "early_stopping_queries": int((held_out & useful).sum())}
        if min(ranker_report["training_queries"], ranker_report["early_stopping_queries"]) < 10:
            ranker_report["skipped"] = "fewer than 10 validation queries with a positive among the candidates"
            log(f"  learned ranker skipped: {ranker_report['skipped']}")
        else:
            fit_set, early_stop_set = pack(~held_out), pack(held_out)
        rankers = {}
        held_rows = np.flatnonzero(held_out)
        discounts = 1 / np.log2(np.arange(k) + 2)

        def held_out_ndcg(ranker: LearnedRanker) -> float:
            """NDCG@k on the held-out validation users, over all their queries (candidates first, as served)."""
            values = []
            for i in held_rows:
                top = np.argsort(-ranker.predict(x_val[i]), kind="stable")[:k]
                values.append((y_val[i][top] * discounts[:len(top)]).sum() / discounts[:min(k, len(validation.targets[i]))].sum())
            return float(np.mean(values))

        selected = "lambdarank"
        if "skipped" not in ranker_report:
            selection = {}
            for objective in profile["ranker"]["objectives"]:
                ranker = LearnedRanker(objective=objective)
                ranker.fit(fit_set, early_stop_set, k)
                selection[objective] = round(held_out_ndcg(ranker), 6)
                log(f"  ranker objective {objective:<12} held-out validation NDCG@{k} {selection[objective]:.4f}")
            selected = max(selection, key=selection.get)
            ranker_report.update(objective_selection=selection, selected_objective=selected)
        variants = () if "skipped" in ranker_report else (
            ("learned_ranker", FEATURES), ("learned_ranker_no_geo", tuple(f for f in FEATURES if f not in GEO_FEATURES)))
        for name, names in variants:
            ranker = LearnedRanker(feature_names=names, objective=selected)
            ranker_report[name] = ranker.fit(fit_set, early_stop_set, k)
            ranker_report[name]["held_out_validation_ndcg"] = round(held_out_ndcg(ranker), 6)
            rankers[name] = ranker

            def scorer(q: Queries, ranker=ranker) -> np.ndarray:
                content = content_batch(features, q, popularity)
                out = -1e6 + content
                two_tower = tt_scorer(q)
                locations = query_locations(q, coords)
                for j in range(len(q)):
                    cand = candidates(source_scores(content[j], two_tower[j], popular, locations[j], coords), sizes)
                    x = ranker_features(cand, content[j], two_tower[j], q.history_items[j], q.history_ratings[j], locations[j], stats)
                    out[j, cand] = ranker.predict(x)
                return out
            score(name, scorer, fitted_on_validation=True)
        if rankers:
            rankers["learned_ranker"].save(artifact_dir / "ranker.txt")

    ann_report = None
    if profile["ann"]:
        log("ANN benchmark")
        sample = test.subset(np.sort(np.random.default_rng(SEED).choice(len(test), min(profile["ann"]["queries"], len(test)), replace=False)))
        from .features import normalize
        from .retrieval import mlp
        query_vectors = normalize(mlp(served_weights["query_layers"], two_tower_inputs(
            content_queries(features, sample), query_locations(sample, coords), geo)))
        index = IVFIndex(served_weights["item_embeddings"], served_weights["item_bias"], served_weights["temperature"], seed=SEED)
        ann_report = benchmark(index, query_vectors, profile["ann"]["k"], profile["ann"]["n_probes"])

    geo_report = None
    if geo is not None:
        locations = query_locations(test, coords)
        distances = np.concatenate([haversine_km(loc[0], loc[1], coords[t, 0], coords[t, 1])
                                    for loc, t in zip(locations, test.targets) if not np.isnan(loc).any()])
        geo_report = {"anchors": geo.dim, "bandwidth_km": round(geo.bandwidth, 3),
                      "test_target_distance_km_percentiles": {str(p): round(float(np.percentile(distances, p)), 2)
                                                              for p in (25, 50, 75, 90)},
                      "share_of_test_targets_within_km": {str(r): round(float(np.mean(distances <= r)), 4) for r in (1, 2, 5, 10)}}

    save_two_tower(artifact_dir / "two_tower.npz", served_weights)
    if features.config["text_backend"] == "sentence":
        np.save(artifact_dir / "text_matrix.npy", features.text_matrix)
    ranker_path = artifact_dir / "ranker.txt"
    if ranker_path.exists() and not (ranker_report and "skipped" not in ranker_report):
        ranker_path.unlink()  # never serve a stale ranker from an earlier run
    digest = hashlib.sha256((artifact_dir / "two_tower.npz").read_bytes()
                            + (ranker_path.read_bytes() if ranker_path.exists() else b"")).hexdigest()[:12]
    model_version = f"{ds.name}_v2_{digest}"
    served = {"baseline": "content_ranked", "two_tower": "two_tower_ranked"}
    if ranker_path.exists():
        served["learned"] = "learned_ranker"
    model_config = {"version": 2, "dataset": ds.name, "k": k, "item_ids": features.ids, "features": features.config,
                    "geo": geo.config if geo else None, "train_end_inclusive": train_end, "prepared_sha256": ds.fingerprint,
                    "model_version": model_version, "served": served, "two_tower": served_trial["configuration"],
                    "ranker_sources": profile["ranker"]["sources"] if profile["ranker"] else None}
    report = {"version": 2, "dataset": ds.name, "model_version": model_version, "seed": SEED, "k": k, "quick": quick,
              "split": {"train_end_inclusive": ds.date(train_end), "validation_end_inclusive": ds.date(val_end),
                        "counts": {"train": int(in_train.sum()), "validation": int(((ds.day > train_end) & (ds.day <= val_end)).sum()),
                                   "test": int((ds.day > val_end).sum())},
                        "queries": {"validation": len(validation), "test": len(test)}},
              "protocol": {"split": "global chronological 80/10/10 by interactions; whole days on one side",
                           "positive_label": "rating >= 4", "unit": "one (user, day) query with every positive that day as targets",
                           "history": "all of the user's interactions strictly before the query day (features fresh, models frozen)",
                           "training_data": "training split only; item statistics from training only",
                           "candidates": "full catalog", "metrics": f"macro Recall@{k}, NDCG@{k}, coverage; recall on new/tail-item targets",
                           "selection": "text backend, two-tower loss/identity, sequential variant and ranker early stopping use validation only",
                           "max_queries": profile["max_queries"], "uncertainty": "95% user-cluster bootstrap on test",
                           "item_groups": {"new_item": "no training interactions", "tail_item": "below the median training count of seen items"},
                           "ranking_policy": RANKING_POLICY},
              "served": served, "policies": results, "text_backend_selected": backend,
              "two_tower": {"trials": trials, "served": served_trial["name"]},
              "sequential": {"trials": sequential_trials}, "semantic_ids": sid_stats, "ranker": ranker_report,
              "ann": ann_report, "geo": geo_report, "quality": public_quality(ds.quality),
              "limitations": KNOWN_LIMITATIONS[ds.name], "training_seconds": round(time.perf_counter() - started, 1),
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (artifact_dir / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2) + "\n")
    (artifact_dir / "offline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    log(f"done in {report['training_seconds']}s -> {artifact_dir}")
    return report
