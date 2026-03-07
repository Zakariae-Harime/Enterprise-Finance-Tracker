"""
JWT access token creation and verification.

Two-token strategy:
  - Access token:  JWT, signed, 15 min, NO database lookup to verify
  - Refresh token: opaque random string, 7 days, stored in DB (revocable)
"""
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from jose import JWTError, jwt
from fastapi import HTTPException, status

# Config
# SECRET_KEY signs every token. If leaked, attackers can forge tokens.
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


# Access Token

def create_access_token(user_id: UUID, org_id: UUID, role: str) -> str:
    """
    Create a signed JWT containing user identity + tenant + role.
    No database call needed to verify — the signature proves validity.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),       # subject — who this token belongs to
        "org_id": str(org_id),     # tenant — which organization
        "role": role,              # RBAC — owner / admin / finance / employee
        "exp": expire,             # expiry — jose validates this automatically
        "iat": datetime.now(timezone.utc),  # issued-at — for audit
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Verify signature and expiry, return payload.
    Raises HTTP 401 on any failure — invalid, expired, tampered.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Verify required claims are present
        if payload.get("sub") is None or payload.get("org_id") is None:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception


# Refresh Token

def generate_refresh_token() -> str:
    """
    Opaque random token — NOT a JWT.
    Stored in database so it can be revoked (logout, stolen token).
    """
    return secrets.token_urlsafe(32)   # 32 bytes = 256 bits of randomness


def get_refresh_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
