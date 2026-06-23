# Deploying the matchup model to Google Cloud Run

The inference server (`POST /predict`, `GET /health`) is a stateless FastAPI app.
The trained checkpoint is **baked into the image** (see `Dockerfile`) because
Cloud Run has no writable volume mounts. Updating the model = rebuild + redeploy.

## Prerequisites

- `gcloud` CLI authenticated: `gcloud auth login` and `gcloud config set project <PROJECT_ID>`
- A region, e.g. `europe-west1` (close to the Supabase `eu-west-3` / Vercel region).
- A fresh `artifacts/matchup-model.pt` in the repo (the one the image bakes in).

## Build & deploy (Cloud Build → Cloud Run)

```bash
# From the repo root (rigged-royale-matchup-ml/).
gcloud run deploy rigged-matchup \
  --source . \
  --region europe-west1 \
  --port 8080 \
  --cpu 1 --memory 512Mi \
  --min-instances 1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_NAME=symmetric-matchup \
  --set-secrets PREDICT_API_KEY=rigged-matchup-key:latest
```

Notes:
- **`--port 8080`**: Cloud Run sets `$PORT`; the shell-form `CMD` binds it.
- **`--min-instances 1`**: keeps one instance warm. Without it the first request
  after idle pays a cold start (container boot + checkpoint load). The site also
  warms the service via `/health` before its prediction fan-out
  (`riggedroyale/src/lib/ml-inference.ts:warmUpModel`), so `--min-instances 0` is
  viable to cut cost — expect the first analysis after idle to be slower.
- **`PREDICT_API_KEY`**: optional but recommended so only the site can call
  `/predict`. Create the secret first:
  ```bash
  printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create rigged-matchup-key --data-file=-
  ```
  The same value must go into Vercel (below). Omit `--set-secrets` to run open.
- `--allow-unauthenticated` because Vercel calls it over plain HTTPS with the
  `X-API-Key` header; the WAF/key is the gate. For tighter control, use Cloud Run
  IAM + an ID token instead.

After deploy, `gcloud` prints the service URL, e.g.
`https://rigged-matchup-xxxx.europe-west1.run.app`. Smoke-test:

```bash
curl -s https://rigged-matchup-xxxx.europe-west1.run.app/health
curl -s -X POST https://rigged-matchup-xxxx.europe-west1.run.app/predict \
  -H 'Content-Type: application/json' -H 'X-API-Key: <KEY>' \
  -d '{"team_card_ids":[26000000,26000001,26000002,26000003,26000004,26000005,26000006,26000007],
       "opponent_card_ids":[26000008,26000009,26000010,26000011,26000012,26000013,26000014,26000015],
       "mode_key":"ladder","team_trophies":8000}'
```

## Wire the site (Vercel)

Set in the Vercel project (Production + Preview):

| Env var | Value |
| --- | --- |
| `ML_INFERENCE_URL` | the Cloud Run service URL (no trailing slash) |
| `PREDICT_API_KEY` | the same secret as the Cloud Run `PREDICT_API_KEY` |
| `ML_INFERENCE_TIMEOUT_MS` | optional; default 6000 is fine once warm |

```bash
vercel env add ML_INFERENCE_URL production   # paste the Cloud Run URL
vercel env add PREDICT_API_KEY production     # paste the same key
```

The site auto-detects the model (`mlInferenceConfigured()` is true once
`ML_INFERENCE_URL` is set) and uses it as the primary matchup engine, falling
back to the empirical matrix per-battle only when a prediction is missing — and
the player page now surfaces which engine produced the verdict.

## Updating the model

```bash
# Replace artifacts/matchup-model.pt with a newer checkpoint, then:
gcloud run deploy rigged-matchup --source . --region europe-west1
```

Newer checkpoints embed `data_config` (trophy buckets) so segment resolution is
self-describing; the serving code falls back to the config defaults for older
checkpoints.
