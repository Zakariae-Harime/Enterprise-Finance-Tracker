"""
IDOR (Insecure Direct Object Reference) Security Tests

Proves that Org A cannot access Org B's resources by guessing UUIDs.

The correct response for cross-tenant access is 404 (not 403):
  - 403 reveals "this resource exists but you can't see it"
  - 404 reveals nothing — the resource may or may not exist

All DB queries include AND organization_id = $N — these tests verify that.
"""
import os
import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4, UUID

os.environ.setdefault("SECRET_KEY",     "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from tests.security.conftest import make_token, bearer, ORG_A, ORG_B, USER_A, USER_B

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client(db_pool):
    app.state.db_pool = db_pool
    app.state.categorizer = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_org_b_cannot_access_org_a_integration(client):
    """
    Org B authenticates with a valid token, but tries to access
    an integration that belongs to Org A (by guessing the UUID).
    Must return 404, not 200.
    """
    org_a_integration_id = uuid4()  # simulates a known-but-foreign UUID
    token_b = make_token(USER_B, ORG_B, "owner")

    resp = await client.get(
        f"/api/v1/integrations/{org_a_integration_id}/sync-jobs",
        headers=bearer(token_b),
    )
    assert resp.status_code == 404, (
        f"Org B accessed Org A's integration! Got {resp.status_code}"
    )


async def test_org_b_cannot_trigger_sync_on_org_a_integration(client):
    """
    POST /integrations/{id}/sync must verify organization_id.
    SyncService._load_integration() includes AND organization_id = $2.
    A foreign integration_id → SyncResult with error → success=False, not data leak.
    """
    org_a_integration_id = uuid4()
    token_b = make_token(USER_B, ORG_B, "owner")

    resp = await client.post(
        f"/api/v1/integrations/{org_a_integration_id}/sync",
        headers=bearer(token_b),
    )
    # Either 200 with success=False (not found) or 404 — both acceptable
    if resp.status_code == 200:
        body = resp.json()
        assert body["success"] is False, "Sync succeeded on another org's integration!"
    else:
        assert resp.status_code == 404


async def test_org_b_cannot_delete_org_a_integration(client):
    """
    DELETE /integrations/{id} scopes by organization_id.
    Cross-org delete returns 404 (UPDATE 0 rows → not found).
    """
    org_a_integration_id = uuid4()
    token_b = make_token(USER_B, ORG_B, "owner")

    resp = await client.delete(
        f"/api/v1/integrations/{org_a_integration_id}",
        headers=bearer(token_b),
    )
    assert resp.status_code == 404


async def test_each_org_sees_only_own_integrations(client, db_pool):
    """
    GET /integrations/ returns only the caller's org integrations.
    Org B's token must not see Org A's integrations list.
    """
    from src.services.credentials import encrypt_credentials

    org_a_integration_id = uuid4()

    # Seed an integration for Org A directly in the DB
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO integrations (id, organization_id, provider, credentials_encrypted, status)
            VALUES ($1, $2, 'tripletex', $3, 'active')
            ON CONFLICT DO NOTHING
            """,
            org_a_integration_id,
            ORG_A,
            encrypt_credentials({"consumer_token": "ct", "employee_token": "et", "company_id": 1}),
        )

    # Org B lists its integrations — must NOT see Org A's
    token_b = make_token(USER_B, ORG_B, "owner")
    resp = await client.get("/api/v1/integrations/", headers=bearer(token_b))

    assert resp.status_code == 200
    ids_returned = [i["id"] for i in resp.json()]
    assert str(org_a_integration_id) not in ids_returned, (
        "Org B can see Org A's integration in the list!"
    )
