"""
Authentication Routes

POST /auth/register  — create account + organization, return tokens
POST /auth/login     — verify credentials, return tokens
POST /auth/refresh   — exchange refresh token for new access token
POST /auth/logout    — revoke refresh token
"""
import hashlib
import secrets
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_db_pool
from src.api.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    MessageResponse,
)
from src.auth.jwt import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    get_refresh_token_expiry,
)
from src.auth.password import hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash_token(raw: str) -> str:
    """SHA-256 hash of a refresh token for safe DB storage."""
    return hashlib.sha256(raw.encode()).hexdigest()


# REGISTER

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db_pool=Depends(get_db_pool),
) -> TokenResponse:
    """
    Create a new user + organization in one atomic transaction.
    The registering user becomes the organization owner.
    """
    async with db_pool.acquire() as conn:
        # Check email not already taken
        existing = await conn.fetchval(
            "SELECT id FROM users WHERE email = $1", request.email
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )

        # Everything in one transaction — user + org + membership
        # If any step fails, the whole thing rolls back
        async with conn.transaction():
            # 1. Create user
            user_id = await conn.fetchval(
                """
                INSERT INTO users (email, hashed_password, full_name)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                request.email,
                hash_password(request.password),
                request.full_name,
            )

            # 2. Create organization
            org_id = await conn.fetchval(
                """
                INSERT INTO organizations (name, slug, plan)
                VALUES ($1, $2, 'pro')
                RETURNING id
                """,
                request.org_name,
                request.org_name.lower().replace(" ", "-"),
            )

            # 3. Add user as owner of the organization
            await conn.execute(
                """
                INSERT INTO organization_members (organization_id, user_id, role)
                VALUES ($1, $2, 'owner')
                """,
                org_id,
                user_id,
            )

            # 4. Issue refresh token
            raw_refresh = generate_refresh_token()
            await conn.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
                """,
                user_id,
                _hash_token(raw_refresh),
                get_refresh_token_expiry(),
            )

    return TokenResponse(
        access_token=create_access_token(user_id, org_id, role="owner"),
        refresh_token=raw_refresh,
    )


# LOGIN

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db_pool=Depends(get_db_pool),
) -> TokenResponse:
    async with db_pool.acquire() as conn:
        # Fetch user by email
        user = await conn.fetchrow(
            "SELECT id, hashed_password, is_active FROM users WHERE email = $1",
            request.email,
        )

        # Same error for wrong email OR wrong password — never reveal which
        auth_error = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

        if not user:
            raise auth_error
        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account is disabled")
        if not verify_password(request.password, user["hashed_password"]):
            raise auth_error

        # Get organization membership
        member = await conn.fetchrow(
            """
            SELECT organization_id, role
            FROM organization_members
            WHERE user_id = $1
            LIMIT 1
            """,
            user["id"],
        )
        if not member:
            raise HTTPException(status_code=403, detail="User has no organization")

        # Issue refresh token
        raw_refresh = generate_refresh_token()
        await conn.execute(
            """
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            """,
            user["id"],
            _hash_token(raw_refresh),
            get_refresh_token_expiry(),
        )

    return TokenResponse(
        access_token=create_access_token(user["id"], member["organization_id"], member["role"]),
        refresh_token=raw_refresh,
    )


#  REFRESH 

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db_pool=Depends(get_db_pool),
) -> TokenResponse:
    token_hash = _hash_token(request.refresh_token)

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, expires_at, revoked
            FROM refresh_tokens
            WHERE token_hash = $1
            """,
            token_hash,
        )

        if not row or row["revoked"] or row["expires_at"] < __import__("datetime").datetime.now(__import__("datetime").timezone.utc):
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

        # Token rotation — delete old, insert new
        async with conn.transaction():
            await conn.execute("DELETE FROM refresh_tokens WHERE id = $1", row["id"])

            raw_refresh = generate_refresh_token()
            await conn.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
                """,
                row["user_id"],
                _hash_token(raw_refresh),
                get_refresh_token_expiry(),
            )

        member = await conn.fetchrow(
            "SELECT organization_id, role FROM organization_members WHERE user_id = $1 LIMIT 1",
            row["user_id"],
        )

    return TokenResponse(
        access_token=create_access_token(row["user_id"], member["organization_id"], member["role"]),
        refresh_token=raw_refresh,
    )


# LOGOUT

@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: LogoutRequest,
    db_pool=Depends(get_db_pool),
) -> MessageResponse:
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = $1",
            _hash_token(request.refresh_token),
        )
    return MessageResponse(message="Logged out successfully")
