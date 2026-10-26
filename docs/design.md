# Design notes

This document covers the decisions behind each stage, the alternatives considered, and what would change at production scale. Headline numbers and citations are in the [README](../README.md); bracketed numbers like [16] refer to its reference list.

## 1. Problem framing

**Task.** Given a visitor's stated preferences (cuisines, a neighborhood or location, price, late opening, restaurants they like) and any history, return a short, explainable list of restaurants that satisfies every hard constraint.

**Two datasets, two regimes:**

| | Chicago (Google Local) | Evanston (academic workbook) |
|---|---|---|
| Scale | 7,581 restaurants, 106k users, 1.33M ratings (2016–2019) | 66 restaurants, 1k users, 1.4k ratings |
| Users | 5-core: median 8 ratings, so warm users with histories | 92% have one rating, so almost everything is cold-start |
| What it tests | Retrieval at scale, negatives, Semantic IDs, sequence models, learned ranking, ANN, geography | Cold-start, content features, evaluation discipline on tiny samples |

**Constraints that shape the design:**

- **Explicit ratings with no exposure logs.** We know what people reviewed, not what they were shown. Unreviewed means unobserved, not disliked. CTR can't be estimated offline, which is why the serving path logs impressions.
- **Location dominates in local services.** In the Chicago test split, the median liked restaurant is about 4 km from the user's usual area (§ 6), so geography has to be a first-class signal in features, candidates and rules.
- **Snapshot metadata.** Categories, price, hours and the "permanently closed" flag were collected in 2021. Using them as features for 2016–2019 behavior is a known approximation, and the closed flag specifically would leak the future (§ 6).

## 2. Data contracts

### Chicago: Google Local Data (2021)

`google_local.py` turns the Illinois files into `items.json` plus `interactions.npz`, and writes a data-quality funnel.

| Rule | Decision | Why |
|---|---|---|
| Scope | Businesses whose address is in Chicago or Evanston and whose categories match a food pattern | Gives a coherent local market; Evanston items join the two datasets. |
| Window | 2016-01-01 to 2019-12-31 | Keeps COVID-era behavior (2020–21) out of the test period. |
| Repeat ratings of one restaurant by one user (32k) | Keep the latest | Google keeps one review per user and place; earlier rows are superseded edits. |
| Duplicate metadata rows | First row wins (13 duplicates) | Same `gmap_id`. |
| k-core | Users ≥ 5 ratings, items ≥ 10, applied iteratively | Gives users histories to learn from. The cost is under-represented cold-start, which the Evanston set covers. |
| User identity | Raw Google account IDs dropped; users become anonymous integer indices | Nothing downstream needs an identifier. |
| `avg_rating`, `num_of_reviews` | Not stored | Computed over all time, so they leak the future. Popularity is always recomputed from the training split. |
| Attributes | Offerings, Atmosphere, Popular for, Dining options, Highlights, Crowd, Planning, Service options kept as item text; Health & safety, Payments, Accessibility dropped | The dropped groups are COVID-era or not taste-related. |
| Price | `$`–`$$$$` (and `₩` variants) map to level 1–4; anything else is unknown | Unknown never satisfies a price filter. |
| Hours | Closing time of the last shift per day; past-midnight closes count as late | "Open after 8 pm" is true if any day closes after 20:00. |

### Evanston: workbook

| Situation | Decision | Rationale |
|---|---|---|
| Exact full-row duplicates (8 rows) | Removed | Byte-identical rows are ingestion artifacts. |
| Unparseable dates (15 rows) | Kept, flagged and excluded from temporal evaluation | Guessing a date would silently move an event across the train/test boundary. |
| Restaurant in ratings but not in catalog (3) | Placeholder item with an empty description | Dropping them would lose ratings; placeholder prose would leak into TF-IDF (§ 9). |
| Reviewer names | Keyed HMAC-SHA256 pseudonyms (key stored with mode 0600) | Stable across rebuilds, not reversible without the key. |
| Review text and demographics | Dropped at ingestion | Not needed for pre-interaction recommendation; review text is post-hoc. |

Both sources load into one columnar `Dataset`: parallel arrays sorted by (day, user, item), plus the catalog. Every downstream stage is source-agnostic.

## 3. Evaluation protocol

The goal is to simulate what the system would have known at serving time.

- **Global chronological split.** 80/10/10 by interactions, with whole days kept on one side [14]. A random or leave-one-out split would train on the future.
- **Models frozen, features fresh.** Models and item statistics use only the training split. Each validation or test query sees the user's full history strictly before the query day, as a deployed system would. (Version 1 froze histories at the training cutoff too. The change makes Evanston's numbers not directly comparable to the old README.)
- **Query unit = (user, day).** Every positive (rating ≥ 4) that day is a target.
- **Full catalog, no sampled negatives** [15]. Recall@K, NDCG@K, catalog coverage, and recall on targets that are new items (no training interactions) or tail items (below the median training count).
- **Sampling.** Chicago evaluates a seeded sample of whole users per split (up to 20k queries), so per-user clustering is preserved.
- **Uncertainty.** A 95% percentile bootstrap that resamples users, not queries.
- **Selection discipline.** The text backend, two-tower loss and identity, sequential variant, early stopping and ranker early stopping all use validation (a 4k-query subsample for fast in-loop checks on Chicago). The test split is scored once per policy. The learned ranker is fitted on validation, so its validation score is not reported.

## 4. Features

`ItemFeatures` builds one L2-normalized content vector per item from metadata only, never from interactions:

- **Text:** a frozen pretrained sentence encoder (all-MiniLM-L6-v2 [21, 22], 384-d), or TF-IDF, over name, categories, description and attributes. The backend is chosen on validation, and the sentence encoder won on both datasets. The encoded matrix is saved with the artifacts, so serving never loads the encoder.
- **Categories:** multi-hot over categories used by at least N items, weighted 0.8/√k.
- **Scalars:** log cost, price level and opening hours, each with a "known" indicator, so missing differs from zero.
- **Geography (Chicago):** Gaussian RBF features over 32 k-means anchors of restaurant locations. The bandwidth is the median anchor nearest-neighbor distance (about 3 km). A user's location is the median of their liked restaurants. Kept separate from the content vector so "similar" doesn't silently mean "nearby" for the content baseline.

## 5. Retrieval

**Popularity.** A Bayesian average rating, `(sum + m·μ)/(n + m)` with m = 5, is a quality prior, not a retriever: it surfaces niche five-star places. "Most popular" (training count) is the real popularity baseline.

**Content.** Cosine similarity between item vectors and the query vector. The query is the weighted mean of liked history (weight = rating − 3), seed restaurants (weight 2) and the requested-cuisine centroid (weight 2).

**Two-tower.**

```
query tower: MLP([content-space query vector ; geo RBF(user location) ; 1])     -> L2-normalized, d = 64
item tower:  MLP([item content vector ; geo RBF(item location)]) (+ ID or SID)  -> L2-normalized
score      = <q, i> / τ + b_item            (τ = 0.05 on Chicago)
```

- **No user-ID embedding.** The query tower reads the same content-space vector the content baseline uses, so the model is inductive: it serves cold users and unseen items.
- **Three losses, trained identically:**
  - `full`: exact softmax over all 7.6k items, with the user's other known positives masked.
  - `in_batch`: softmax over the other targets in the mini-batch of 1,024. Duplicates and known positives are masked.
  - `in_batch_logq`: in-batch with the correction of Yi et al. [16], `logit_j − log q_j`, where `q_j` is item j's share of training positives.

  In-batch negatives are sampled in proportion to popularity, so without the correction the model learns to push popular restaurants down.
- **Item identity variants** follow Singh et al. [2]:
  - `+id` adds a learned per-item embedding (memorization).
  - `+sid` adds the sum of embeddings of the item's Semantic-ID prefixes, so items that share a prefix share parameters.

**Semantic IDs.** RQ-Kmeans, as in OneRec [6]: three residual levels × 64 codes over the content vector, giving 2,890 distinct IDs for 7,581 items. Restaurants sharing a full ID are mostly chain branches (up to about 150 items share one ID), whose content is identical. Embedding tables get one row per prefix that occurs, so the deepest level has 2,890 rows, not 64³ = 262k.

**Sequential Transformer.** SASRec-style [18]: learned positions and causal self-attention over the user's last 50 interactions, with item inputs = projection(content ⊕ geo) (+ ID or SID). It is trained autoregressively, the generative next-item objective of HSTU [4], with a full softmax over the catalog and loss only on positive targets. Training windows are non-overlapping, right-padded and length-bucketed. Bucketing halves epoch time, because the median history is 8 but the window is 50. Users with no history fall back to popularity.

## 6. Ranking and business rules

**Fixed blend** (both datasets): `0.85 · minmax(retrieval) + 0.15 · minmax(Bayesian rating)`, row-wise. The weights were fixed before test evaluation.

**Learned ranker** (Chicago): LightGBM [19] over the union of four candidate sources; its loss (LambdaRank [20], pointwise logistic or XE-NDCG) is selected on validation:

| Source | Top-n | Why |
|---|---|---|
| Two-tower | 100 | Strongest single retriever. |
| Content | 50 | Taste similarity independent of the learned model. |
| Most popular | 50 | Reliable fallback. |
| Nearby | 50 | Closest restaurants to the user's location; the classic local-services generator. |

The four sources together contain about 30% of validation targets (the candidate-recall ceiling).

- **Features** (14):
  - retrieval scores and log-ranks (content, two-tower)
  - log training count and log count in the last 90 days of training
  - Bayesian rating
  - distance from the user's location
  - cuisine affinity: the share of liked history with the same cuisine
  - price level and the gap to the user's usual price level
  - open late
  - history length and mean rating
- **Excluded feature:** the 2021 "permanently closed" flag. A restaurant that closed in 2020 was open in 2019, so the flag would tell the model which 2019 restaurants were about to fail. It is only a serving filter.
- **Training:** fitted on validation-period queries (retrievers frozen at the training cutoff), with early stopping on 20% of validation users held out. Evaluated once on test.
- **Objective as a hyperparameter:** LambdaRank, pointwise binary logistic and XE-NDCG share the same trees and features. The one with the best NDCG@10 on the held-out validation users is served (§ 9).
- **Ablation:** the same ranker without the distance feature quantifies the value of geography.
- **Train/serve skew to watch:** offline, ranker "history" is real interactions; online, it is the seed restaurants, treated as 5-star ratings.

**Hard constraints** run before candidate truncation, so a filter never empties a list that the full catalog could fill:

- cuisine, maximum price level, budget, open late, maximum distance, and permanently closed (excluded by default)
- unknown metadata never satisfies a filter
- an empty result returns an explanation, never a silently relaxed filter

## 7. ANN retrieval

`ann.py` implements an inverted-file index, the coarse quantizer behind IVF-ADC [24]. It runs k-means over item embeddings (√n ≈ 87 lists); a query scans only the `nprobe` closest lists. The item bias is folded into the vectors, `[v/τ, b] · [q, 1]`, so the index returns exactly the two-tower score for each item it visits. The benchmark reports recall of the IVF top-100 against exact top-100, items scanned and per-query latency. At 7.6k items exact scoring takes about 1 ms, so serving still uses exact search; the index and its recall/latency curve are what you'd switch to as the catalog grows.

## 8. Serving, logging and learning from logs

**Artifacts.** The served artifacts are:

- the feature config and saved sentence-encoder matrix
- the geo anchors
- two-tower weights, with the item embeddings precomputed
- the LightGBM model, plus its feature list
- `model_config.json`: the data fingerprint, item order, training cutoff and model version (a hash of the weights)
- `offline_report.json`

The engine refuses to start if the prepared data's fingerprint or item order differs from the training run. Serving needs NumPy and LightGBM, not PyTorch or the encoder.

**API hygiene:**

- strict schemas that reject unknown fields, including any user ID
- range checks on location, distance and price
- a 16 KiB body limit
- a same-origin check on writes
- CSP and `nosniff` headers
- structured JSON access logs that never include request bodies

**Logging.** Every recommendation writes a `requests` row (preferences, model, version) and one `impressions` row per returned item, with position, retrieval score, ranking score and distance. A click or favorite is accepted only if `(request_id, session_id, item_id)` matches a logged impression; its position and model version are copied from that impression, never taken from the client. `GET /api/online-metrics` reports per-arm CTR and favorite rate.

**Learning from logs (`feedback.py`).** A multi-task MMoE [23] network predicts click and favorite from logged features. A shallow position tower, following Zhao et al. [17], adds a per-position bias logit during training and is dropped at serving, so ranking reflects estimated utility rather than slot.

- **Simulation test.** An old ranker placed items by a feature unrelated to utility, and the true position bias is −1.2·log(position). The naive model learns to trust that feature. The debiased model recovers the decreasing position curve (correlation above 0.9 with the true curve) and ranks fresh items closer to true utility.
- **Traffic gate.** The CLI refuses to train with fewer than 2,000 impressions and 100 clicks of real traffic.

**LLM reranker arm (`llm_rerank.py`).** An optional `llm` arm passes the best arm's top 20 to Claude Opus 5.5 with the user's stated preferences.

- **Request:** structured JSON output (ranking of the given IDs) at low effort, with server-side refusal fallbacks enabled.
- **Position bias:** LLM rankers are order-sensitive [11], so the candidates are ranked in two orders (as retrieved, and reversed) and merged with a Borda count. A purely order-following LLM therefore cancels out.
- **Failure handling:** any failure (credentials, network, refusal, malformed output) keeps the base order.
- **Attribution:** the arm logs its own model version (`…+llm:claude-opus-5-5`), so its CTR is an A/B readout. It is off by default; it costs an API call per request and needs credentials.

## 9. Lessons from building it

- **Missing values should be missing, not prose.** Evanston's ratings-only restaurants first got a stock description that flowed into TF-IDF and made them artificially similar. Emptying it changed the two-tower's selected configuration while leaving the content model unchanged.
- **Sampling-bias correction is not optional with in-batch negatives.** On Chicago, plain in-batch training reached test NDCG@10 0.012; the logQ correction lifted it to 0.030, close to the exact full softmax (0.032), with overlapping intervals and a fraction of the cost per step.
- **A pretrained text encoder beats TF-IDF for sparse content.** On Evanston, content retrieval rose from NDCG@5 0.057 to 0.088 (+55%), with tail-item recall up from 0.11 to 0.17. On Chicago the two tie for pure content retrieval (0.0037 vs 0.0038 NDCG@10), where location and collaborative signal dominate.
- **Per-request work over the full catalog adds up.** The first Chicago benchmark showed 40 ms p50 over HTTP against 5 ms in the engine, because validation rebuilt the 7.6k-item catalog and the filters looped in Python on every request. Precomputing the catalog sets and vectorizing the filters brought it to 8 ms.
- **Validation and test can disagree on small data.** On Evanston, the two-tower configuration that won validation (NDCG@5 0.125) scored 0.026 on test, while the content model was stable. With about 70–90 queries per split, report intervals and ship the simpler model.
- **A learned ranker can lose to its own input feature.** The first Chicago ranker (LambdaRank) scored below simply ordering its candidates by two-tower score, on held-out validation (0.0286 vs 0.0297 NDCG@10) as well as on test. With one or two positives among about 220 candidates, pairwise NDCG gradients are sparse and noisy, and it early-stopped at 60 trees. A pointwise logistic loss beat the two-tower order on the held-out validation users, so the objective became a hyperparameter chosen there. The general rule: always report the ranker next to its strongest single feature.
- **Geography matters, and where it enters matters too.** Sixty-one percent of liked restaurants are within 5 km of the user's usual area. Distance is the ranker's second feature by gain, yet removing it costs only 3% NDCG, because the two-tower's geo features and the nearby candidate source already carry location.

## 10. What changes at scale

| Concern | Here (7.6k items, 106k users) | At 10M users / 1M items |
|---|---|---|
| Retrieval | Exact scoring; IVF index benchmarked | IVF-PQ / HNSW (FAISS, ScaNN); multiple generators merged to about 1k candidates |
| Two-tower training | In-batch + logQ with static frequency counts | Streaming frequency estimation [16], mixed negatives, larger batches across devices |
| Item identity | Content + Semantic-ID prefix embeddings | The same, plus generative retrieval over SIDs (TIGER [1]) with constrained beam search |
| Sequence modeling | 2-layer SASRec-style Transformer, 50 items | HSTU-style long-sequence models [4]; MTGR-style DLRM features alongside [8] |
| Ranking | LightGBM (validation-selected loss) on validation queries | Multi-task deep ranker on impressions with position debiasing (this repo's `feedback.py`, scaled) |
| Features | Computed per request from artifacts | Feature store with offline/online parity; streaming session features |
| Logging | Synchronous SQLite | Async event bus (Kafka) to a warehouse; impression and event joins in the pipeline |
| Evaluation | Temporal split, bootstrap CIs, per-arm CTR | Online A/B tests or interleaving with guardrails and CUPED variance reduction |
| Deployment | Single process | Stateless replicas behind a load balancer; shadow traffic and canaries per model version |
