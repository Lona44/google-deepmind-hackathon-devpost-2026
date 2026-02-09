# Deployment Guide

Deploy the G1 Alignment dashboard for hackathon judges: **Vercel** (frontend) + **Cloud Run** (backend) + **GCS** (experiment data).

## Prerequisites

- Google Cloud project with billing (`modelproof-platform`)
- `gcloud` CLI authenticated: `gcloud auth login`
- Vercel account linked to the GitHub repo
- Node.js 18+ and Docker installed locally

## Architecture

```
Judges → Vercel (Next.js dashboard)
           ↓ /api/chat → Gemini 3 Pro (Vertex AI)
           ↓ tool calls → Cloud Run (FastAPI backend)
                            ↓ paper search (Vertex AI Search)
                            ↓ web search (Google Search grounding)
                            ↓ video analysis (Gemini vision + GCS)
                            ↓ video streaming (GCS signed URLs)
```

## Step 1: Upload Experiment Data to GCS

Upload the 223MB `extractions/` directory to GCS so Cloud Run can serve it.

```bash
# Ensure the bucket exists
gsutil ls gs://g1-experiment-videos/ || gsutil mb -l us-central1 gs://g1-experiment-videos/

# Upload extractions (incremental — skips unchanged files)
./scripts/upload_extractions_to_gcs.sh g1-experiment-videos

# Verify
gsutil ls gs://g1-experiment-videos/extractions/barrels_corrupt/ | head -5
```

## Step 2: Deploy FastAPI Backend to Cloud Run

### Build and push Docker image

```bash
cd gcp/web

# Build with Cloud Build (no local Docker needed)
gcloud builds submit --tag gcr.io/modelproof-platform/g1-web:latest .

# Or build locally and push
docker build -t gcr.io/modelproof-platform/g1-web:latest .
docker push gcr.io/modelproof-platform/g1-web:latest
```

### Deploy to Cloud Run

```bash
gcloud run deploy g1-web \
  --image gcr.io/modelproof-platform/g1-web:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "\
GOOGLE_CLOUD_PROJECT=modelproof-platform,\
GOOGLE_CLOUD_LOCATION=us-central1,\
GCS_BUCKET_NAME=g1-experiment-videos,\
VERTEX_SEARCH_DATASTORE_ID=<YOUR_DATASTORE_ID>,\
FRONTEND_URL=https://<YOUR_VERCEL_DOMAIN>"
```

Note the Cloud Run URL from the output (e.g., `https://g1-web-xxxxx-uc.a.run.app`).

### Verify backend health

```bash
curl https://g1-web-xxxxx-uc.a.run.app/health
# → {"status":"healthy"}

curl https://g1-web-xxxxx-uc.a.run.app/api/search/papers/catalog
# → {"total_papers": 12, "papers": [...]}
```

## Step 3: Deploy Next.js Frontend to Vercel

### Option A: Vercel CLI

```bash
cd template-dashboard-oss
npx vercel --prod
```

When prompted:
- **Root directory**: `template-dashboard-oss` (if deploying from repo root)
- **Framework**: Next.js (auto-detected)
- **Build command**: `next build`
- **Output directory**: `.next`

### Option B: Vercel Dashboard

1. Import the GitHub repo at [vercel.com/new](https://vercel.com/new)
2. Set **Root Directory** to `template-dashboard-oss`
3. Add environment variables (see below)
4. Deploy

### Environment Variables (Vercel)

Set these in **Vercel Dashboard → Settings → Environment Variables**:

| Variable | Value | Notes |
|----------|-------|-------|
| `AUTH_PASSWORD_HASH` | `0b93c01c1dff729d8e4a04dab257594a115810ee049e1348c018fdd165993d42` | SHA-256 of judge password |
| `AUTH_SECRET` | *(generate with `openssl rand -hex 32`)* | Session signing secret |
| `GOOGLE_VERTEX_PROJECT` | `modelproof-platform` | Vertex AI project |
| `GOOGLE_VERTEX_LOCATION` | `global` | Vertex AI region |
| `BACKEND_URL` | `https://g1-web-xxxxx-uc.a.run.app` | Cloud Run URL from Step 2 |
| `OPENAI_API_KEY` | `sk-proj-...` | For GPT-5 eval display |
| `MOONSHOT_API_KEY` | `sk-...` | For Kimi K2.5 eval display |

**Important**: For Vertex AI to work from Vercel, you need to set up Workload Identity Federation or use a service account key:

```bash
# Create a service account key for Vercel
gcloud iam service-accounts keys create vercel-sa-key.json \
  --iam-account=g1-web-app@modelproof-platform.iam.gserviceaccount.com

# Base64 encode it for Vercel env var
cat vercel-sa-key.json | base64
```

Then add to Vercel:
| Variable | Value |
|----------|-------|
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | *(base64-encoded service account key)* |

Or use the `GOOGLE_GENERATIVE_AI_API_KEY` env var for direct Gemini API access (simpler but fewer features):
| Variable | Value |
|----------|-------|
| `GOOGLE_GENERATIVE_AI_API_KEY` | `AIza...` | Direct Gemini API key |

## Step 4: Update CORS

After Vercel deploys, update Cloud Run with the actual Vercel domain:

```bash
gcloud run services update g1-web \
  --region us-central1 \
  --update-env-vars "FRONTEND_URL=https://your-project.vercel.app"
```

## Step 5: Verify End-to-End

1. Open `https://your-project.vercel.app`
2. Log in with the judge password
3. Navigate to **Research** — send a message, verify streaming works
4. Try: "What papers are in the database?" (tests paper catalog tool)
5. Try: "Search the web for AI deception research 2026" (tests web search tool)
6. Navigate to **Settings** — verify all credentials show as configured
7. Check backend status shows "Online"

## Troubleshooting

### "Backend not running" in Research chat
- Check Cloud Run URL is correct in `BACKEND_URL`
- Verify Cloud Run service is deployed: `gcloud run services list`
- Check Cloud Run logs: `gcloud run services logs read g1-web --region us-central1`

### Gemini API errors
- Vertex AI mode: Ensure the service account has `roles/aiplatform.user`
- Direct API mode: Check `GOOGLE_GENERATIVE_AI_API_KEY` is set

### Video streaming returns 404
- Verify extractions uploaded: `gsutil ls gs://g1-experiment-videos/extractions/`
- Check `GCS_BUCKET_NAME` is set on Cloud Run
- Ensure service account has `roles/storage.objectViewer` on the bucket

### CORS errors in browser console
- Verify `FRONTEND_URL` on Cloud Run matches your Vercel domain exactly
- The backend also allows `*.vercel.app` by default

## Cost Estimate

| Service | Spec | Monthly Cost |
|---------|------|-------------|
| Cloud Run (backend) | 0-10 instances, 1 CPU / 512MB | ~$5-15 |
| GCS (extractions) | 223MB storage + egress | <$1 |
| Vertex AI (Gemini 3) | Pay-per-token | ~$5-20 |
| Vercel (frontend) | Hobby/Pro plan | Free-$20 |
| **Total** | | **~$10-55/mo** |

For a hackathon demo period (1-2 weeks), total cost is negligible.
