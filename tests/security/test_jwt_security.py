"""
JWT Security Tests

Proves that the auth layer correctly rejects:
  - Tampered payloads (signature mismatch)
  - Expired tokens
  - None algorithm attack
  - Wrong signing key
  - Missing / malformed Authorization header
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from jose import jwt
from uuid import UUID

os.environ.setdefault("SECRET_KEY",     "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from src.auth.jwt import SECRET_KEY, ALGORITHM
from tests.security.conftest import make_tampered_token, bearer, ORG_A, USER_A

pytestmark = pytest.mark.asyncio

# Any protected GET endpoint — we just need a 401, not actual data
PROTECTED = "/api/v1/integrations/"


@pytest.fixture
async def client():
    # JWT tests are rejected at auth middleware — never reach DB.
    # db_pool is not needed; None is safe here.
    app.state.db_pool = None
    app.state.categorizer = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_tampered_payload_rejected(client):
    """
    Attacker changes org_id in the JWT payload but cannot update the signature.
    Must be rejected with 401 — not 200 or 403.
    """
    bad_token = make_tampered_token(USER_A, ORG_A, "owner")
    resp = await client.get(PROTECTED, headers=bearer(bad_token))
    assert resp.status_code == 401, f"Tampered token accepted! Got {resp.status_code}"


async def test_expired_token_rejected(client):
    """Token with exp in the past must be rejected with 401."""
    payload = {
        "sub":    str(USER_A),
        "org_id": str(ORG_A),
        "role":   "owner",
        "exp":    datetime.now(timezone.utc) - timedelta(seconds=1),
        "iat":    datetime.now(timezone.utc) - timedelta(minutes=16),
    }
    expired_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    resp = await client.get(PROTECTED, headers=bearer(expired_token))
    assert resp.status_code == 401, f"Expired token accepted! Got {resp.status_code}"


async def test_none_algorithm_attack_rejected(client):
    """
    CVE pattern: set alg='none', omit signature.
    python-jose rejects this because we pass algorithms=['HS256'] to jwt.decode().
    """
    payload = {
        "sub":    str(USER_A),
        "org_id": str(ORG_A),
        "role":   "owner",
        "exp":    datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    # Manually craft a token with alg=none
    import base64, json
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps({**payload, "exp": int(payload["exp"].timestamp())}).encode()
    ).rstrip(b"=").decode()
    none_token = f"{header}.{body}."  # empty signature

    resp = await client.get(PROTECTED, headers=bearer(none_token))
    assert resp.status_code == 401, f"None-algorithm token accepted! Got {resp.status_code}"


async def test_wrong_signing_key_rejected(client):
    """Token signed with a different secret must be rejected."""
    payload = {
        "sub":    str(USER_A),
        "org_id": str(ORG_A),
        "role":   "owner",
        "exp":    datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    wrong_key_token = jwt.encode(payload, "completely-different-secret-key!", algorithm=ALGORITHM)
    resp = await client.get(PROTECTED, headers=bearer(wrong_key_token))
    assert resp.status_code == 401


async def test_missing_authorization_header_returns_401(client):
    """No Authorization header → 401, not 500 (route is properly guarded)."""
    resp = await client.get(PROTECTED)
    assert resp.status_code == 401


async def test_malformed_bearer_value_rejected(client):
    """Authorization: Bearer not-a-jwt → 401."""
    resp = await client.get(PROTECTED, headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


async def test_missing_sub_claim_rejected(client):
    """Token without 'sub' claim is structurally invalid — must be rejected."""
    payload = {
        # "sub" intentionally missing
        "org_id": str(ORG_A),
        "role":   "owner",
        "exp":    datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    bad_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    resp = await client.get(PROTECTED, headers=bearer(bad_token))
    assert resp.status_code == 401


async def test_missing_org_id_claim_rejected(client):
    """Token without 'org_id' claim is structurally invalid — must be rejected."""
    payload = {
        "sub":  str(USER_A),
        # "org_id" intentionally missing
        "role": "owner",
        "exp":  datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    bad_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    resp = await client.get(PROTECTED, headers=bearer(bad_token))
    assert resp.status_code == 401


@pytest.mark.xfail(reason="Known gap: SECRET_KEY falls back to 'change-me-in-production' if env var not set")
async def test_default_secret_key_is_insecure(client):
    """
    Documents a known security gap: if SECRET_KEY env var is not set,
    jwt.py falls back to 'change-me-in-production' — a public, known string.
    Any attacker can forge valid tokens with this key.

    Fix: replace os.environ.get(..., fallback) with:
         SECRET_KEY = os.environ["SECRET_KEY"]  (raises KeyError if missing)
    or:  raise RuntimeError("SECRET_KEY must be set") at module load time.
    """
    import os
    original = os.environ.pop("SECRET_KEY", None)
    try:
        # Reload the module to pick up the missing env var
        import importlib
        import src.auth.jwt as jwt_module
        importlib.reload(jwt_module)

        # With the fallback key, an attacker can sign arbitrary tokens
        payload = {
            "sub":    str(UUID("deadbeef-dead-beef-dead-beefdeadbeef")),
            "org_id": str(ORG_A),
            "role":   "owner",
            "exp":    datetime.now(timezone.utc) + timedelta(hours=24),
        }
        attacker_token = jwt.encode(payload, "change-me-in-production", algorithm="HS256")
        resp = await client.get(PROTECTED, headers=bearer(attacker_token))

        # This SHOULD fail (token accepted = vulnerability confirmed = xfail passes)
        assert resp.status_code == 200
    finally:
        if original:
            os.environ["SECRET_KEY"] = original
        import importlib, src.auth.jwt as jwt_module
        importlib.reload(jwt_module)
