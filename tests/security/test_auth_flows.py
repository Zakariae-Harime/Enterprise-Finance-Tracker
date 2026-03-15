"""
Auth Flow Security Tests

Proves that:
  - Refresh token replay after logout is rejected
  - Refresh token single-use (rotation) — old token invalid after use
  - Expired refresh token is rejected
  - Brute force login is possible (documents rate-limiting gap — marked xfail)
  - Error messages don't reveal whether email exists (anti-enumeration)
"""
import os
import pytest
import hashlib
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

os.environ.setdefault("SECRET_KEY",     "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from src.auth.password import hash_password

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client(db_pool):
    app.state.db_pool = db_pool
    app.state.categorizer = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _seed_user(db_pool, email: str, password: str, role: str = "owner") -> dict:
    """Insert a real user + org into the test DB. Returns {user_id, org_id}."""
    user_id = uuid4()
    org_id  = uuid4()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, hashed_password, full_name) VALUES ($1, $2, $3, $4)"
            " ON CONFLICT (email) DO NOTHING",
            user_id, email, hash_password(password), "Test User",
        )
        slug = f"test-org-{org_id.hex[:8]}"
        await conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            org_id, "Test Org", slug,
        )
        await conn.execute(
            "INSERT INTO organization_members (user_id, organization_id, role) VALUES ($1, $2, $3)"
            " ON CONFLICT DO NOTHING",
            user_id, org_id, role,
        )
    return {"user_id": user_id, "org_id": org_id}


async def test_refresh_token_invalidated_after_logout(client, db_pool):
    """
    Flow: register → login → logout → try to refresh with revoked token → 401.
    The refresh_tokens table has revoked=TRUE after logout.
    """
    email = f"logout-test-{uuid4().hex[:8]}@test.no"
    await _seed_user(db_pool, email, "TestPass123!")

    # Login
    login = await client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123!"})
    assert login.status_code == 200, f"Login failed: {login.text}"
    refresh_token = login.json()["refresh_token"]

    # Logout
    logout = await client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout.status_code == 200

    # Try to refresh with the now-revoked token
    refresh = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh.status_code == 401, "Revoked refresh token was accepted after logout!"


async def test_refresh_token_single_use_rotation(client, db_pool):
    """
    Flow: login → use refresh token → try to use the SAME token again → 401.
    Token rotation deletes the old row on first use.
    """
    email = f"rotation-test-{uuid4().hex[:8]}@test.no"
    await _seed_user(db_pool, email, "TestPass123!")

    login = await client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123!"})
    original_refresh = login.json()["refresh_token"]

    # First refresh — succeeds, deletes original token, issues new one
    first = await client.post("/api/v1/auth/refresh", json={"refresh_token": original_refresh})
    assert first.status_code == 200

    # Second refresh with the original token — must fail (it was deleted)
    second = await client.post("/api/v1/auth/refresh", json={"refresh_token": original_refresh})
    assert second.status_code == 401, "Replayed refresh token was accepted!"


async def test_expired_refresh_token_rejected(client, db_pool):
    """
    Insert a refresh token with expires_at in the past directly into the DB.
    Attempting to use it must return 401.
    """
    email = f"expired-rt-{uuid4().hex[:8]}@test.no"
    ids = await _seed_user(db_pool, email, "TestPass123!")

    raw_token    = "expired-token-value-for-test"
    token_hash   = hashlib.sha256(raw_token.encode()).hexdigest()
    expired_time = datetime.now(timezone.utc) - timedelta(days=1)

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at) VALUES ($1, $2, $3, $4)",
            uuid4(), ids["user_id"], token_hash, expired_time,
        )

    resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": raw_token})
    assert resp.status_code == 401, "Expired refresh token was accepted!"


async def test_wrong_password_returns_401(client, db_pool):
    """Standard wrong-password rejection."""
    email = f"wrongpass-{uuid4().hex[:8]}@test.no"
    await _seed_user(db_pool, email, "CorrectPass123!")

    resp = await client.post("/api/v1/auth/login", json={"email": email, "password": "WrongPassword!"})
    assert resp.status_code == 401


async def test_nonexistent_email_same_error_as_wrong_password(client, db_pool):
    """
    Anti-enumeration: both "wrong password" and "email not found" must return
    the same HTTP status and same error message.
    Prevents attackers from discovering which emails are registered.
    """
    real_email = f"real-{uuid4().hex[:8]}@test.no"
    await _seed_user(db_pool, real_email, "SomePass123!")

    wrong_pass = await client.post(
        "/api/v1/auth/login",
        json={"email": real_email, "password": "WrongPassword!"},
    )
    no_user = await client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@test.no", "password": "AnyPassword!"},
    )

    assert wrong_pass.status_code == no_user.status_code == 401
    assert wrong_pass.json().get("detail") == no_user.json().get("detail"), (
        "Different error messages reveal whether email exists (enumeration risk)"
    )


@pytest.mark.xfail(reason="Known gap: no rate limiting on /auth/login — brute force is possible. Fix: add slowapi.")
async def test_brute_force_login_is_rate_limited(client, db_pool):
    """
    10 rapid login attempts with wrong passwords should trigger rate limiting.
    Currently NO rate limiting exists — all 10 return 401 freely.

    Fix:
      pip install slowapi
      from slowapi import Limiter
      from slowapi.util import get_remote_address
      limiter = Limiter(key_func=get_remote_address)

      @router.post("/login")
      @limiter.limit("5/minute")
      async def login(...): ...

    Until then, this test is xfail (documents the gap without blocking CI).
    """
    email = f"bruteforce-{uuid4().hex[:8]}@test.no"
    await _seed_user(db_pool, email, "RealPass123!")

    rate_limited = False
    for _ in range(10):
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "WrongPassword!"},
        )
        if resp.status_code == 429:
            rate_limited = True
            break

    assert rate_limited, "10 rapid failed logins were all accepted — no rate limiting"
