"""
Security test fixtures.

Security tests do NOT bypass auth via dependency_overrides.
They use real JWTs so the actual auth stack is exercised.
"""
import base64
import json
import os
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from uuid import UUID

os.environ.setdefault("SECRET_KEY",        "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY",    "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from src.auth.jwt import create_access_token

# Two distinct orgs — used for IDOR tests
ORG_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
ORG_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
USER_A = UUID("11111111-1111-1111-1111-111111111111")
USER_B = UUID("22222222-2222-2222-2222-222222222222")


def make_token(user_id: UUID, org_id: UUID, role: str) -> str:
    """Create a real signed JWT using the test SECRET_KEY."""
    return create_access_token(user_id=user_id, org_id=org_id, role=role)


def make_tampered_token(user_id: UUID, org_id: UUID, role: str) -> str:
    """
    Create a valid token then swap org_id in the payload WITHOUT re-signing.
    Simulates an attacker who tries to access another org's data.
    The signature will no longer match the modified payload.
    """
    real_token = make_token(user_id, org_id, role)
    header_b64, payload_b64, signature = real_token.split(".")

    # Decode payload, change org_id to a different org
    padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    payload["org_id"] = str(ORG_B if org_id == ORG_A else ORG_A)

    # Re-encode payload WITHOUT updating the signature
    new_payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()

    return f"{header_b64}.{new_payload_b64}.{signature}"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture(scope="session", autouse=True)
async def seed_security_orgs(db_pool):
    """
    Pre-seed ORG_A and ORG_B into the organizations table.
    IDOR tests insert integrations with these org IDs — FK requires them to exist.
    Session-scoped so it runs once and persists for all security tests.
    """
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            ORG_A, "Org Alpha", "org-alpha",
        )
        await conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            ORG_B, "Org Beta", "org-beta",
        )


@pytest_asyncio.fixture
async def security_client(db_pool):
    """
    HTTP client with REAL auth (no dependency_overrides).
    Routes get the real get_current_user dependency — JWT must be valid.
    """
    app.state.db_pool = db_pool
    app.state.categorizer = None
    # No dependency_overrides — auth is real

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
def owner_token():
    return make_token(USER_A, ORG_A, "owner")


@pytest_asyncio.fixture
def employee_token():
    return make_token(USER_A, ORG_A, "employee")


@pytest_asyncio.fixture
def finance_token():
    return make_token(USER_A, ORG_A, "finance")


@pytest_asyncio.fixture
def org_b_owner_token():
    return make_token(USER_B, ORG_B, "owner")
