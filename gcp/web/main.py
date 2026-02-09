"""
G1 Alignment Platform - GCP Cloud Run Web App

FastAPI application with Firebase Auth, Firestore, and WebSocket streaming.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from api.capabilities import router as capabilities_router
from api.chat import router as chat_router
from api.experiments import router as experiments_router
from api.keys import router as keys_router
from api.search import router as search_router
from api.streaming import router as streaming_router
from api.video import router as video_router
from auth import router as auth_router
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Load .env file for local development
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize Firebase and other services on startup."""
    firebase_initialized = False
    try:
        import firebase_admin  # noqa: PLC0415
        from firebase_admin import credentials  # noqa: PLC0415

        # Use default credentials in Cloud Run, or local service account
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            cred = credentials.Certificate(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            firebase_admin.initialize_app(cred)
            firebase_initialized = True
        elif os.environ.get("K_SERVICE"):  # Running in Cloud Run
            firebase_admin.initialize_app()
            firebase_initialized = True
        else:
            print("Firebase not configured - auth features disabled (local dev mode)")
    except Exception as e:
        print(f"Firebase initialization skipped: {e}")

    yield

    # Cleanup on shutdown
    if firebase_initialized:
        import firebase_admin  # noqa: PLC0415

        firebase_admin.delete_app(firebase_admin.get_app())


app = FastAPI(
    title="G1 Alignment Platform",
    description="AI alignment research platform with live MuJoCo visualization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
_cors_origins = [
    "http://localhost:3000",  # Local dev (Next.js default)
    "http://localhost:3001",  # Local dev (alternate port)
    "http://localhost:3003",  # Local dev (alternate port)
    "http://localhost:8080",  # Local backend
    "https://*.web.app",  # Firebase Hosting
    "https://*.vercel.app",  # Vercel deployments
]
# Add explicit FRONTEND_URL if set
_frontend_url = os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    _cors_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(capabilities_router, prefix="/api/capabilities", tags=["capabilities"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])
app.include_router(keys_router, prefix="/api/keys", tags=["keys"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(video_router, prefix="/api/video", tags=["video"])
app.include_router(streaming_router, prefix="/ws", tags=["streaming"])


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy"}


# Serve frontend static files for local development
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    # Mount static assets (node_modules, assets, src)
    app.mount(
        "/node_modules",
        StaticFiles(directory=frontend_path / "node_modules"),
        name="node_modules",
    )
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")
    app.mount("/src", StaticFiles(directory=frontend_path / "src"), name="src")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the frontend index.html."""
        return FileResponse(frontend_path / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
