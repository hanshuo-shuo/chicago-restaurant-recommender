# Table for You — public hosted adapter

This English portfolio site preserves the content baseline, trained Two Tower inference, hard constraints, offline experiment report and anonymous feedback workflow of the local Evanston restaurant project.

The Worker uses exported restaurant metadata features, precomputed trained item embeddings and the trained query projection. It performs real inference for visitor preferences. The hosted adapter matches the local Python engine in 12 fixed parity cases, with score tolerance 0.000002. Training continues in the local Python project; this website does not train on visitor events.

## Public data boundary

Only restaurant facts, content-feature matrices, learned aggregate parameters and aggregate offline metrics are included. Original review text, workbook, historical user interactions, HMAC identities, local secrets and local SQLite logs are excluded. Course restaurant descriptions are replaced with short factual summaries. Dataset provenance and redistribution rights remain unconfirmed; historical metadata is not a current dining guide.

## Feedback

Consent is required before logging. D1 stores impressions and attributed click/favorite events under anonymous UUID sessions. Every event is marked synthetic=1 / source=portfolio_demo and excluded from offline evaluation. Reporting uses the last seven days; older records are removed on subsequent recommendation requests. Event records do not contain names, email, IP or precise location. Favorites shown in the interface are device-local preferences.

## Evaluation accuracy

The Two Tower configuration and epoch were selected on validation retrieval NDCG@5. The UI baseline is a fixed demonstration default, not the validation winner. Validation ranked NDCG@5 was 0.084403 for Two Tower and 0.067137 for baseline; test ranked NDCG@5 was 0.035067 and 0.050194 respectively. Test includes only 67 queries from 56 pseudonymous users, including 12 warm queries from eight users. No business uplift is claimed.

## Local verification

```sh
npm run build
npm run validate
node scripts/verify.mjs
npm run dev
```

The preview uses Node built-in SQLite. Production uses the Sites-managed D1 binding and the committed Drizzle migration. Database schema edits require a new generated migration. The production Worker has no third-party runtime dependencies.

A page tool can stage visible preferences where WebMCP is supported; it does not consent to logging or submit recommendations.
