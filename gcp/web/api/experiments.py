"""
Experiments API for G1 Platform.

Handles starting experiments, checking status, and retrieving results.
"""

import uuid
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from google.cloud import firestore, pubsub_v1
from pydantic import BaseModel, Field

from auth import User, get_current_user, get_db

router = APIRouter()

# Pub/Sub client for job queue
_publisher: Optional[pubsub_v1.PublisherClient] = None

# Maximum concurrent experiments per user
MAX_CONCURRENT_EXPERIMENTS = 4


def get_publisher() -> pubsub_v1.PublisherClient:
    """Get Pub/Sub publisher client (singleton)."""
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


class ExperimentCreate(BaseModel):
    """Request to create a new experiment."""

    scenario: Literal["barrels_lo", "barrels_hi", "open_field"] = "barrels_lo"
    model: Literal["robotics", "pro", "flash"] = "robotics"
    reasoning: Literal["none", "low", "medium", "high"] = "high"


class Experiment(BaseModel):
    """Experiment model."""

    id: str
    user_id: str
    scenario: str
    model: str
    reasoning: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results_url: Optional[str] = None
    error: Optional[str] = None


@router.get("")
async def list_experiments(
    user: User = Depends(get_current_user),
    limit: int = 20,
):
    """List user's experiments."""
    db = get_db()
    experiments_ref = (
        db.collection("users")
        .document(user.uid)
        .collection("experiments")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )

    experiments = []
    for doc in experiments_ref.stream():
        exp_data = doc.to_dict()
        exp_data["id"] = doc.id
        experiments.append(exp_data)

    return {"experiments": experiments}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    request: ExperimentCreate,
    user: User = Depends(get_current_user),
):
    """
    Start a new experiment.

    Enforces max concurrent experiments per user.
    """
    db = get_db()

    # Check concurrent experiment limit
    running_count = (
        db.collection("users")
        .document(user.uid)
        .collection("experiments")
        .where("status", "in", ["pending", "running"])
        .count()
        .get()[0][0]
        .value
    )

    if running_count >= MAX_CONCURRENT_EXPERIMENTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum {MAX_CONCURRENT_EXPERIMENTS} concurrent experiments allowed",
        )

    # Check if user has API keys configured
    keys_ref = db.collection("users").document(user.uid).collection("api_keys")
    if not any(keys_ref.limit(1).stream()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please configure at least one API key before running experiments",
        )

    # Create experiment record
    experiment_id = str(uuid.uuid4())
    experiment_data = {
        "user_id": user.uid,
        "scenario": request.scenario,
        "model": request.model,
        "reasoning": request.reasoning,
        "status": "pending",
        "created_at": firestore.SERVER_TIMESTAMP,
    }

    db.collection("users").document(user.uid).collection("experiments").document(
        experiment_id
    ).set(experiment_data)

    # Publish job to Pub/Sub queue
    publisher = get_publisher()
    topic_path = publisher.topic_path("g1-alignment", "experiment-jobs")

    import json

    job_data = {
        "experiment_id": experiment_id,
        "user_id": user.uid,
        "scenario": request.scenario,
        "model": request.model,
        "reasoning": request.reasoning,
    }

    publisher.publish(topic_path, json.dumps(job_data).encode("utf-8"))

    return {
        "id": experiment_id,
        "status": "pending",
        "message": "Experiment queued successfully",
    }


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    user: User = Depends(get_current_user),
):
    """Get experiment details."""
    db = get_db()
    exp_ref = (
        db.collection("users")
        .document(user.uid)
        .collection("experiments")
        .document(experiment_id)
    )
    exp_doc = exp_ref.get()

    if not exp_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    exp_data = exp_doc.to_dict()
    exp_data["id"] = experiment_id

    return exp_data


@router.delete("/{experiment_id}")
async def cancel_experiment(
    experiment_id: str,
    user: User = Depends(get_current_user),
):
    """Cancel a pending or running experiment."""
    db = get_db()
    exp_ref = (
        db.collection("users")
        .document(user.uid)
        .collection("experiments")
        .document(experiment_id)
    )
    exp_doc = exp_ref.get()

    if not exp_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    exp_data = exp_doc.to_dict()
    if exp_data["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only pending or running experiments can be cancelled",
        )

    exp_ref.update(
        {
            "status": "failed",
            "error": "Cancelled by user",
            "completed_at": firestore.SERVER_TIMESTAMP,
        }
    )

    return {"message": "Experiment cancelled"}
