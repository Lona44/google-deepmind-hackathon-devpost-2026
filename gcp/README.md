# G1 Alignment Platform - GCP Deployment

GCP-based research platform for running and reviewing AI alignment experiments with interactive 3D playback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GCP Platform                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐  │
│  │  Cloud Run   │     │  Cloud Run Jobs  │     │  Cloud Storage  │  │
│  │  (Web App)   │────▶│  (Workers)       │────▶│  (Results)      │  │
│  │  FastAPI     │     │  MuJoCo + AI     │     │  - trajectory   │  │
│  └──────────────┘     └──────────────────┘     │  - video.mp4    │  │
│         │                                       └─────────────────┘  │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │  Firestore   │                                                    │
│  │  - Users     │                                                    │
│  │  - API Keys  │                                                    │
│  │  - Experiments│                                                   │
│  └──────────────┘                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ Browser downloads trajectory
                              ▼
                ┌──────────────────────────────────┐
                │              Browser              │
                │  ┌────────────────────────────┐  │
                │  │  mujoco_wasm + Three.js    │  │
                │  │  - Load G1 robot model     │  │
                │  │  - Playback with camera    │  │
                │  │  - Pause, rewind, speed    │  │
                │  └────────────────────────────┘  │
                └──────────────────────────────────┘
```

## Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| Web API | `web/` | FastAPI app with Firebase Auth |
| Worker | `worker/` | Runs MuJoCo experiments, saves trajectories |
| Frontend | `frontend/` | 3D playback viewer (mujoco_wasm + Three.js) |
| Terraform | `terraform/` | Infrastructure as code |

## Quick Start

### 1. Set Up GCP Project

```bash
# Create project and enable billing
gcloud projects create g1-alignment --name="G1 Alignment"
gcloud config set project g1-alignment

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com
```

### 2. Deploy Infrastructure

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

terraform init
terraform plan
terraform apply
```

### 3. Deploy Web App

```bash
cd web
gcloud run deploy g1-web --source .
```

### 4. Deploy Worker

```bash
cd worker
gcloud builds submit --tag gcr.io/YOUR_PROJECT/g1-worker
gcloud run jobs create g1-worker --image gcr.io/YOUR_PROJECT/g1-worker
```

### 5. Deploy Frontend

```bash
cd frontend
npm install
# Deploy to Firebase Hosting
firebase deploy --only hosting
```

## Development

### Run Web App Locally

```bash
cd web
pip install -r requirements.txt
uvicorn main:app --reload
```

### Run Frontend Locally

```bash
cd frontend
npm install
npx five-server
# Open http://localhost:5500
```

### Test Playback

1. Open the frontend in browser
2. Drag and drop `assets/sample_trajectory.json` onto the viewer
3. Use playback controls:
   - Space: Play/Pause
   - ←/→: Step frames
   - [/]: Change speed
   - R: Reset to start
   - Mouse: Rotate/zoom camera

## API Endpoints

### Authentication
- `POST /api/auth/verify` - Verify Firebase token

### API Keys
- `GET /api/keys` - List vendor API keys
- `POST /api/keys` - Add/update API key
- `DELETE /api/keys/{vendor}` - Remove API key

### Experiments
- `GET /api/experiments` - List experiments
- `POST /api/experiments` - Start new experiment
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/experiments/{id}/trajectory` - Get trajectory download URL
- `GET /api/experiments/{id}/viewer` - Get 3D viewer URL
- `DELETE /api/experiments/{id}` - Cancel experiment

## Environment Variables

### Web App
| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID |
| `RESULTS_BUCKET` | Cloud Storage bucket for results |
| `VIEWER_URL` | Frontend viewer URL |

### Worker
| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID |
| `RESULTS_BUCKET` | Cloud Storage bucket for results |

## Cost Estimate

| Component | Spec | Monthly Cost |
|-----------|------|--------------|
| Cloud Run (web) | 1 vCPU, 512MB | ~$15 |
| Cloud Run Jobs | On-demand | ~$5-20 |
| Firestore | 1GB, 50K ops/day | ~$5 |
| Cloud Storage | 50GB | ~$1 |
| **Total** | | **~$25-40/mo** |

## Security

- Firebase Auth handles Google OAuth
- API keys encrypted at rest (Firestore + base64, KMS in prod)
- Signed URLs for trajectory access (1-hour expiry)
- Service accounts with minimal permissions
