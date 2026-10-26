# data/

Nothing in this folder except this file is committed.

| Path | Produced by | Contents |
|---|---|---|
| `raw/google_local/meta-Illinois.json.gz`, `rating-Illinois.csv.gz` | `make download` | Google Local Data (2021), UCSD McAuley Lab: Illinois business metadata and (business, user, rating, timestamp) tuples. A public academic release; cite Li et al. (ACL 2022) and Yan et al. (SIGIR 2023) and do not redistribute. |
| `raw/RestaurantReviews.xlsx` | you | Evanston ratings workbook (`Restaurants` and `Reviews` sheets), for `DATASET=evanston`. |
| `processed/chicago/items.json` | `make data` | Chicago and Evanston restaurants (categories, coordinates, price level, hours, attributes, closed flag). |
| `processed/chicago/interactions.npz` | `make data` | user (anonymous integer), item, rating, day; 5-core users, 10-core items, 2016–2019. Raw Google account IDs are dropped. |
| `processed/<dataset>/data_quality.json` | `make data` | Filtering funnel, duplicates, quarantined rows and missing-metadata counts. |
| `processed/evanston/prepared.json` | `make data DATASET=evanston` | Validated items and pseudonymized interactions (no names, review text or demographics). |
| `processed/evanston/.hmac_secret` | `make data DATASET=evanston` | 32-byte key for user pseudonyms (mode 0600). Keep it private. |
| `logs/events.sqlite3` | `make serve` | Requests, impressions, clicks and favorites from the API. |
