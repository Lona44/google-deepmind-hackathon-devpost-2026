# Vertex AI Integration Plan

## Overview

Add dual-mode support for the G1 Research Assistant:
- **Free tier**: Direct Gemini API (API key only)
- **Vertex AI tier**: Full features (video analysis, paper RAG, Google Search)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    G1 Research Assistant                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User: "Does Kimi's behavior match deceptive alignment?"         │
│                                                                  │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Experiment  │    Video     │   Research   │     Google        │
│  Data (JSON) │   Analysis   │   Papers     │     Search        │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│  ✅ All      │  ⚡ Vertex   │  ⚡ Vertex   │   ⚡ Vertex       │
│    users     │    only      │    Search    │    Grounding      │
└──────────────┴──────────────┴──────────────┴───────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      Gemini 3 Flash/Pro       │
              │  Synthesizes with citations   │
              └───────────────────────────────┘
```

## Feature Matrix

| Feature | Free (API Key) | Vertex AI |
|---------|----------------|-----------|
| Chat about experiments | ✅ | ✅ |
| Load trajectories, control viewer | ✅ | ✅ |
| Basic experiment analysis | ✅ | ✅ |
| **Video analysis** (watch MP4s) | ❌ | ✅ |
| **Research paper RAG** | ❌ | ✅ |
| **Google Search grounding** | ❌ | ✅ |

## Implementation Phases

### Phase 1: Backend Infrastructure (2-3 hours)

#### New Files
- `gcp/web/services/vertex_client.py` - Unified client with mode detection
- `gcp/web/api/capabilities.py` - Expose available features
- `gcp/web/services/gcs_client.py` - Video upload to GCS

#### Key Code: Vertex Client
```python
# gcp/web/services/vertex_client.py
import os
from google import genai

class VertexClient:
    def __init__(self):
        self.mode = self._detect_mode()
        self.client = self._init_client()

    def _detect_mode(self) -> str:
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            return "vertex"
        elif os.getenv("GEMINI_API_KEY"):
            return "free"
        raise ValueError("No credentials found")

    def _init_client(self) -> genai.Client:
        if self.mode == "vertex":
            return genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location="global"
            )
        else:
            return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
```

#### Capabilities Endpoint
```python
# GET /api/capabilities
{
  "mode": "vertex",
  "features": {
    "chat": true,
    "video_analysis": true,
    "paper_rag": true,
    "google_search": true
  },
  "models": {
    "chat": "gemini-3.1-pro-preview",
    "vision": "gemini-3-flash-preview"
  }
}
```

### Phase 2: Video Analysis (2-3 hours)

#### New Endpoint: `/api/chat/video`
```python
# POST /api/chat/video
{
  "run_id": "2026-02-06T04-28_kimi-k2.5",
  "question": "What went wrong in this run?"
}

# Response
{
  "analysis": {
    "summary": "Robot collided with barrel #2 at 23.4s...",
    "key_moments": [
      {"time": 23.4, "event": "First barrel contact"}
    ]
  }
}
```

#### Video Flow
1. Check if video exists: `extractions/*/media/full_run.mp4`
2. Upload to GCS bucket (cached)
3. Call Gemini 3 Flash with video URI
4. Return structured analysis

### Phase 3: Paper RAG with Vertex AI Search (3-4 hours)

#### Setup Steps
1. Enable Discovery Engine API
2. Create data store for PDFs
3. Upload AI safety papers to GCS
4. Import into data store

#### SDK Requirements
```bash
pip install google-cloud-discoveryengine
```

#### Create Data Store
```python
from google.cloud import discoveryengine_v1

def create_paper_datastore(project_id: str):
    client = discoveryengine_v1.DataStoreServiceClient()

    data_store = discoveryengine_v1.DataStore(
        display_name="AI Safety Papers",
        solution_type=discoveryengine_v1.SolutionType.SOLUTION_TYPE_SEARCH,
    )

    parent = f"projects/{project_id}/locations/global/collections/default_collection"
    request = discoveryengine_v1.CreateDataStoreRequest(
        parent=parent,
        data_store=data_store,
        data_store_id="ai-safety-papers",
    )

    operation = client.create_data_store(request=request)
    return operation.result()
```

#### Search with Grounding
```python
from google.genai.types import Tool, Retrieval, VertexAISearch

def search_papers(query: str, datastore_id: str):
    tool = Tool(
        retrieval=Retrieval(
            vertex_ai_search=VertexAISearch(
                datastore=f"projects/{PROJECT}/locations/global/collections/default_collection/dataStores/{datastore_id}"
            )
        )
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=query,
        config=GenerateContentConfig(tools=[tool])
    )
    return response
```

### Phase 4: Frontend Integration (2-3 hours)

#### New Tools in agentTools.js
```javascript
// Add to TOOL_DECLARATIONS
{
  name: "analyze_video",
  description: "Watch and analyze an experiment video",
  parameters: {
    type: "object",
    properties: {
      run_id: { type: "string" },
      question: { type: "string" }
    }
  }
},
{
  name: "search_papers",
  description: "Search AI safety research papers",
  parameters: {
    type: "object",
    properties: {
      query: { type: "string" }
    }
  }
},
{
  name: "web_search",
  description: "Search Google for recent research",
  parameters: {
    type: "object",
    properties: {
      query: { type: "string" }
    }
  }
}
```

#### Capability Detection
```javascript
// gcp/frontend/src/services/capabilities.js
export async function getCapabilities() {
  const response = await fetch('/api/capabilities');
  return response.json();
}

export function isFeatureEnabled(capabilities, feature) {
  return capabilities?.features?.[feature] ?? false;
}
```

#### UI Updates (ChatPanel.js)
- Add mode indicator badge (Free/Vertex AI)
- Show/hide tools based on capabilities
- Add video upload UI when available

## Environment Variables

### Free Tier (minimal)
```bash
GEMINI_API_KEY=AIza...
```

### Vertex AI Tier (full features)
```bash
# GCP
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global

# Vertex AI Search
VERTEX_SEARCH_DATASTORE_ID=ai-safety-papers

# GCS (for video uploads)
GCS_BUCKET_NAME=g1-experiment-videos
```

## Files to Create

| File | Purpose |
|------|---------|
| `gcp/web/services/vertex_client.py` | Dual-mode Gemini client |
| `gcp/web/services/gcs_client.py` | Video upload to GCS |
| `gcp/web/api/capabilities.py` | Feature detection endpoint |
| `gcp/web/api/video.py` | Video analysis endpoint |
| `gcp/web/api/search.py` | Paper RAG + Google Search |
| `gcp/frontend/src/services/capabilities.js` | Frontend capability check |

## Files to Modify

| File | Changes |
|------|---------|
| `gcp/web/main.py` | Register new routers |
| `gcp/web/api/chat.py` | Add Vertex mode support |
| `gcp/frontend/src/agentTools.js` | Add new tools |
| `gcp/frontend/src/geminiAgent.js` | Capability-aware client |
| `gcp/frontend/src/components/ChatPanel.js` | Mode indicator, feature badges |

## Demo Script (3 min)

1. **Free Mode** (30s)
   - "Show me the worst safety run"
   - Agent loads trajectory, explains scores

2. **Video Analysis** (60s)
   - "Watch the video - what exactly went wrong?"
   - Agent analyzes MP4, describes key moments

3. **Paper RAG** (60s)
   - "Does this match deceptive alignment patterns?"
   - Agent searches papers, returns citations

4. **Google Search** (30s)
   - "Find recent papers on AI deception"
   - Agent searches web, returns arxiv links

5. **Synthesis** (30s)
   - "What's your assessment?"
   - Agent combines all sources

## Cost Estimate (Using $1,725 Credits)

| Feature | Monthly Usage | Cost |
|---------|---------------|------|
| Paper indexing | ~100 PDFs | ~$5 |
| RAG queries | ~1000/month | ~$10 |
| Google Search | ~500/month | ~$17.50 |
| **Total** | | **~$30/month** |

Credits last: **~4+ years** at this rate

## Prerequisites

1. Enable APIs:
```bash
gcloud services enable discoveryengine.googleapis.com --project=your-project-id
gcloud services enable storage.googleapis.com --project=your-project-id
```

2. Create GCS bucket:
```bash
gsutil mb -p your-project-id gs://g1-experiment-videos
```

3. Install dependencies:
```bash
pip install google-cloud-discoveryengine google-cloud-storage
```

## Testing Checklist

### Free Mode
- [ ] Chat works with only GEMINI_API_KEY
- [ ] Video/RAG tools hidden in UI
- [ ] Capabilities endpoint returns correct flags

### Vertex Mode
- [ ] Video upload + analysis works
- [ ] Paper search returns citations
- [ ] Google search works
- [ ] Mode indicator shows "Vertex AI"
