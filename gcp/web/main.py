"""
G1 Alignment Platform - GCP Cloud Run Web App

FastAPI application with Firebase Auth, Firestore, and WebSocket streaming.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from auth import router as auth_router
from api.experiments import router as experiments_router
from api.keys import router as keys_router
from api.streaming import router as streaming_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Firebase and other services on startup."""
    # Initialize Firebase Admin SDK
    import firebase_admin
    from firebase_admin import credentials

    # Use default credentials in Cloud Run, or local service account
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        cred = credentials.Certificate(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        firebase_admin.initialize_app(cred)
    else:
        # Cloud Run provides default credentials
        firebase_admin.initialize_app()

    yield

    # Cleanup on shutdown
    firebase_admin.delete_app(firebase_admin.get_app())


app = FastAPI(
    title="G1 Alignment Platform",
    description="AI alignment research platform with live MuJoCo visualization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local dev
        "https://*.web.app",  # Firebase Hosting
        os.environ.get("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])
app.include_router(keys_router, prefix="/api/keys", tags=["keys"])
app.include_router(streaming_router, prefix="/ws", tags=["streaming"])


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main dashboard page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>G1 Alignment Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <div id="app">
            <h1>G1 Alignment Platform</h1>
            <p>Loading...</p>
        </div>
        <script type="module" src="/static/app.js"></script>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
