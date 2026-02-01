"""
Firebase Authentication for G1 Platform.

Handles Google OAuth via Firebase Auth and token verification.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth
from google.cloud import firestore
from pydantic import BaseModel

router = APIRouter()
security = HTTPBearer()

# Firestore client (initialized lazily)
_db: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """Get Firestore client (singleton)."""
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


class User(BaseModel):
    """User model."""

    uid: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None


class TokenVerifyRequest(BaseModel):
    """Request to verify Firebase ID token."""

    id_token: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    Verify Firebase ID token and return current user.

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"email": user.email}
    """
    token = credentials.credentials

    try:
        # Verify the Firebase ID token
        decoded_token = firebase_auth.verify_id_token(token)

        return User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email", ""),
            name=decoded_token.get("name"),
            picture=decoded_token.get("picture"),
        )
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
        )


@router.post("/verify")
async def verify_token(request: TokenVerifyRequest):
    """
    Verify a Firebase ID token and return user info.

    Called by frontend after Google Sign-In to validate the session.
    """
    try:
        decoded_token = firebase_auth.verify_id_token(request.id_token)

        user = User(
            uid=decoded_token["uid"],
            email=decoded_token.get("email", ""),
            name=decoded_token.get("name"),
            picture=decoded_token.get("picture"),
        )

        # Create/update user in Firestore
        db = get_db()
        user_ref = db.collection("users").document(user.uid)
        user_ref.set(
            {
                "email": user.email,
                "name": user.name,
                "picture": user.picture,
                "last_login": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {
            "valid": True,
            "user": user.model_dump(),
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""
    db = get_db()
    user_ref = db.collection("users").document(user.uid)
    user_doc = user_ref.get()

    user_data = user.model_dump()
    if user_doc.exists:
        stored_data = user_doc.to_dict()
        user_data["experiment_count"] = stored_data.get("experiment_count", 0)
        user_data["created_at"] = stored_data.get("created_at")

    return user_data
