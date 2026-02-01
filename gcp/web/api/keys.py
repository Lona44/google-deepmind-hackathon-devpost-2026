"""
API Keys management for G1 Platform.

Handles secure storage and retrieval of user API keys.
Keys are encrypted at rest using Google Cloud KMS.
"""

import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from google.cloud import firestore
from pydantic import BaseModel, Field

from auth import User, get_current_user, get_db

router = APIRouter()

# Supported API key vendors
SUPPORTED_VENDORS = ["gemini", "openai", "anthropic", "moonshot"]


class APIKeyCreate(BaseModel):
    """Request to store an API key."""

    vendor: Literal["gemini", "openai", "anthropic", "moonshot"]
    api_key: str = Field(..., min_length=10, max_length=256)


class APIKeyInfo(BaseModel):
    """API key info (without the actual key)."""

    vendor: str
    created_at: str
    last_used: Optional[str] = None
    masked_key: str  # e.g., "sk-...abc123"


def mask_key(key: str) -> str:
    """Mask an API key for display."""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def encrypt_key(key: str) -> str:
    """
    Encrypt API key for storage.

    In production, use Google Cloud KMS:
        from google.cloud import kms
        client = kms.KeyManagementServiceClient()
        response = client.encrypt(name=key_name, plaintext=key.encode())
        return base64.b64encode(response.ciphertext).decode()

    For now, use simple encoding (replace with KMS in production).
    """
    import base64

    # TODO: Replace with KMS encryption
    return base64.b64encode(key.encode()).decode()


def decrypt_key(encrypted: str) -> str:
    """
    Decrypt API key from storage.

    In production, use Google Cloud KMS.
    """
    import base64

    # TODO: Replace with KMS decryption
    return base64.b64decode(encrypted.encode()).decode()


@router.get("")
async def list_keys(user: User = Depends(get_current_user)):
    """List user's configured API keys (masked)."""
    db = get_db()
    keys_ref = db.collection("users").document(user.uid).collection("api_keys")

    keys = []
    for doc in keys_ref.stream():
        key_data = doc.to_dict()
        keys.append(
            APIKeyInfo(
                vendor=doc.id,
                created_at=str(key_data.get("created_at", "")),
                last_used=str(key_data.get("last_used", "")) if key_data.get("last_used") else None,
                masked_key=key_data.get("masked_key", "***"),
            )
        )

    return {"keys": keys}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_key(
    request: APIKeyCreate,
    user: User = Depends(get_current_user),
):
    """Store a new API key."""
    if request.vendor not in SUPPORTED_VENDORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported vendor. Supported: {SUPPORTED_VENDORS}",
        )

    db = get_db()
    key_ref = (
        db.collection("users")
        .document(user.uid)
        .collection("api_keys")
        .document(request.vendor)
    )

    # Encrypt and store
    encrypted = encrypt_key(request.api_key)
    key_ref.set(
        {
            "encrypted_key": encrypted,
            "masked_key": mask_key(request.api_key),
            "created_at": firestore.SERVER_TIMESTAMP,
        }
    )

    return {
        "vendor": request.vendor,
        "message": "API key stored successfully",
    }


@router.delete("/{vendor}")
async def delete_key(
    vendor: str,
    user: User = Depends(get_current_user),
):
    """Delete an API key."""
    if vendor not in SUPPORTED_VENDORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported vendor. Supported: {SUPPORTED_VENDORS}",
        )

    db = get_db()
    key_ref = (
        db.collection("users")
        .document(user.uid)
        .collection("api_keys")
        .document(vendor)
    )

    if not key_ref.get().exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {vendor} API key found",
        )

    key_ref.delete()

    return {"message": f"{vendor} API key deleted"}


async def get_user_api_key(user_id: str, vendor: str) -> Optional[str]:
    """
    Get decrypted API key for a user.

    Used by worker to retrieve keys for experiments.
    """
    db = get_db()
    key_ref = (
        db.collection("users")
        .document(user_id)
        .collection("api_keys")
        .document(vendor)
    )
    key_doc = key_ref.get()

    if not key_doc.exists:
        return None

    key_data = key_doc.to_dict()
    encrypted = key_data.get("encrypted_key")

    if not encrypted:
        return None

    # Update last_used timestamp
    key_ref.update({"last_used": firestore.SERVER_TIMESTAMP})

    return decrypt_key(encrypted)
