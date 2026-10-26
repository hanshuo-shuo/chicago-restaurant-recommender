"""Command-line entry point: download -> prepare -> train -> report / bench -> serve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen
import uuid

from .config import SEED
from .dataset import DATASETS

GOOGLE_LOCAL_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/"
GOOGLE_LOCAL_FILES = ("meta-Illinois.json.gz", "rating-Illinois.csv.gz")
CITATION = """Google Local Data (2021), UCSD McAuley Lab. The dataset page asks that you cite:
  J. Li, J. Shang, J. McAuley. UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining. ACL 2022.
  A. Yan, Z. He, J. Li, T. Zhang, J. McAuley. Personalized Showcases: Generating Multi-Modal Explanations for Recommendations. SIGIR 2023."""
WARMUP_REQUESTS = 50


def _source_hash(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _download(raw_dir: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(CITATION + "\n")
    for name in GOOGLE_LOCAL_FILES:
        target = raw_dir / name
        if target.exists():
            print(f"{target} already exists; skipping")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        print(f"downloading {GOOGLE_LOCAL_URL + name} ...", flush=True)
        with urlopen(GOOGLE_LOCAL_URL + name, timeout=60) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.rename(target)
        print(f"  saved {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return 0


def _format_report(report: dict) -> str:
    k = report["k"]
    split = report["split"]
    lines = [f"{report['dataset']}  model_version: {report.get('model_version', '-')}",
             f"split: train <= {split['train_end_inclusive']} < validation <= {split['validation_end_inclusive']} < test; "
             f"queries: validation {split['queries']['validation']}, test {split['queries']['test']}",
             f"served: {report['served']}", "",
             f"{'policy':<34}{'val NDCG':>10}{'Recall@' + str(k):>11}{'NDCG@' + str(k):>9}{'  95% CI':<20}{'Cov.':>7}{'tail R':>8}"]
    for name, result in report["policies"].items():
        test, val = result["test"], result["validation"]
        ci = test["overall"].get("ndcg_95pct_user_bootstrap") or [float("nan")] * 2
        tail = test.get(f"recall_at_{k}_tail_item_targets", {}).get("recall")
        lines.append(f"{name:<34}{(val['overall'][f'ndcg_at_{k}'] if val else float('nan')):>10.4f}"
                     f"{test['overall'][f'recall_at_{k}']:>11.4f}{test['overall'][f'ndcg_at_{k}']:>9.4f}"
                     f"  [{ci[0]:.4f}, {ci[1]:.4f}]  {test['overall'][f'catalog_coverage_at_{k}']:>6.3f}"
                     f"{(tail if tail is not None else float('nan')):>8.4f}")
    return "\n".join(lines)


def _percentiles(timings: list[float]) -> dict:
    ordered = sorted(timings)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"p50_ms": round(pick(.5), 3), "p95_ms": round(pick(.95), 3), "p99_ms": round(pick(.99), 3)}


def _bench(engine, requests: int) -> dict:
    """In-process engine latency and end-to-end HTTP latency on random preference queries, per served model."""
    from .server import create_server
    rng = random.Random(SEED)
    info = engine.info()
    queries = [{"cuisines": rng.sample(info["cuisines"], rng.randint(0, 2)),
                "seed_item_ids": rng.sample(engine.features.ids, rng.randint(0, 2))} for _ in range(requests)]
    for query in queries[::2]:
        if info["locations"]:
            spot = rng.choice(info["locations"])
            query.update(location={"latitude": spot["latitude"], "longitude": spot["longitude"]}, max_distance_km=5.)
    results = {"requests_per_model": requests, "catalog_size": len(engine.features.ids)}
    models = [m for m in engine.models if m != "llm"]  # never spend API calls on a latency benchmark
    for model in models:
        for preferences in queries[:WARMUP_REQUESTS]:
            engine.recommend(preferences=preferences, model=model, k=10)
        timings = []
        for preferences in queries:
            started = time.perf_counter()
            engine.recommend(preferences=preferences, model=model, k=10)
            timings.append((time.perf_counter() - started) * 1000)
        results[f"engine_{model}"] = _percentiles(timings)
    with tempfile.TemporaryDirectory() as directory:
        server = create_server(engine, port=0, db_path=Path(directory) / "bench.sqlite3")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:%d/api/recommend" % server.server_address[1]
        session = str(uuid.uuid4())

        def post(model: str, preferences: dict) -> None:
            body = json.dumps({"session_id": session, "model": model, "k": 10, "preferences": preferences}).encode()
            with urlopen(Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=10) as response:
                response.read()
        try:
            for model in models:
                for preferences in queries[:WARMUP_REQUESTS]:
                    post(model, preferences)
                timings = []
                for preferences in queries:
                    started = time.perf_counter()
                    post(model, preferences)
                    timings.append((time.perf_counter() - started) * 1000)
                results[f"http_{model}"] = _percentiles(timings)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="recsys", description="Restaurant recommender: data, training, evaluation and serving")
    parser.add_argument("command", choices=["download", "prepare", "train", "report", "bench", "serve", "demo", "train-feedback"],
                        help="demo = prepare (if needed) + train + serve")
    parser.add_argument("--dataset", choices=DATASETS, default="chicago")
    parser.add_argument("--source", type=Path, help="raw data: the Evanston workbook, or the Google Local folder")
    parser.add_argument("--data-dir", type=Path, help="processed data (default data/processed/<dataset>)")
    parser.add_argument("--artifact-dir", type=Path, help="model artifacts (default artifacts/<dataset>)")
    parser.add_argument("--events-db", type=Path, default=Path("data/logs/events.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--requests", type=int, default=500, help="queries per model for bench")
    parser.add_argument("--quick", action="store_true", help="small, fast training run (smoke test; numbers not meaningful)")
    parser.add_argument("--llm-rerank", action="store_true", help="serve the 'llm' arm (needs the anthropic package and API credentials)")
    args = parser.parse_args(argv)
    default_source = {"evanston": Path("data/raw/RestaurantReviews.xlsx"), "chicago": Path("data/raw/google_local")}
    source = args.source or default_source[args.dataset]
    data_dir = args.data_dir or Path("data/processed") / args.dataset
    artifact_dir = args.artifact_dir or Path("artifacts") / args.dataset
    ready = data_dir / ("prepared.json" if args.dataset == "evanston" else "interactions.npz")

    if args.command == "download":
        if args.dataset != "chicago":
            print("Only the Chicago dataset (Google Local) is downloadable; place the Evanston workbook at data/raw/.", file=sys.stderr)
            return 2
        return _download(source)

    if args.command == "prepare" or (args.command == "demo" and not ready.exists()):
        if args.dataset == "evanston":
            if not source.is_file():
                print(f"Workbook not found: {source}. Place it there or pass --source.", file=sys.stderr)
                return 2
            from .data import prepare
            digest_before = _source_hash(source)
            prepared = prepare(source, data_dir)
            assert _source_hash(source) == digest_before, "Source workbook unexpectedly changed"
            quality = prepared["quality"]
            print(f"Prepared {len(prepared['items'])} items and {len(prepared['interactions'])} interactions "
                  f"from {quality['unique_users']} users -> {data_dir}")
        else:
            if not all((source / name).exists() for name in GOOGLE_LOCAL_FILES):
                print(f"Google Local files not found in {source}. Run: recsys download --dataset chicago", file=sys.stderr)
                return 2
            from .google_local import prepare_google_local
            quality = prepare_google_local(source, data_dir)
            print(json.dumps({key: quality[key] for key in ("funnel", "catalog_items", "unique_users", "retained_interactions")}, indent=2))
        if args.command == "prepare":
            return 0

    if not ready.exists():
        print(f"No prepared data in {data_dir}. Run 'prepare --dataset {args.dataset}' first.", file=sys.stderr)
        return 2
    from .dataset import load_dataset
    ds = load_dataset(args.dataset, data_dir)
    if args.command in {"train", "demo"}:
        try:
            from .training import train
        except ImportError as exc:
            print(f"Training needs extra packages ({exc.name} missing). Install with: pip install -e '.[train]'", file=sys.stderr)
            return 2
        report = train(ds, artifact_dir, quick=args.quick)
        print("\n" + _format_report(report))
        if args.command == "train":
            return 0

    from .engine import Engine
    llm = None
    if args.llm_rerank:
        try:
            from .llm_rerank import LLMReranker
            llm = LLMReranker()
        except ImportError:
            print("The 'llm' arm needs the anthropic package: pip install -e '.[llm]'", file=sys.stderr)
            return 2
    engine = Engine(ds, artifact_dir, llm_reranker=llm)
    if args.command == "train-feedback":
        from .feedback import train_from_log
        print(json.dumps(train_from_log(args.events_db, engine, artifact_dir), indent=2))
        return 0
    if args.command == "report":
        if not engine.report:
            print("No offline report. Run 'train' first.", file=sys.stderr)
            return 2
        print(_format_report(engine.report))
        return 0
    if args.command == "bench":
        print(json.dumps(_bench(engine, args.requests), indent=2))
        return 0
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    from .server import serve
    try:
        serve(engine, host=args.host, port=args.port, db_path=args.events_db)
    except OSError as exc:
        print(f"Could not start the server on {args.host}:{args.port}: {exc}. Try another --port.", file=sys.stderr)
        return 2
    print("\nServer stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
