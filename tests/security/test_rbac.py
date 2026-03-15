"""
RBAC (Role-Based Access Control) Security Tests

Proves that role guards work correctly:
  - Lower-privilege roles cannot reach higher-privilege endpoints
  - Correct roles get through
  - 403 is returned (not 404 or 500) for insufficient permissions
"""
import os
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

os.environ.setdefault("SECRET_KEY",     "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from tests.security.conftest import (
    make_token, bearer,
    ORG_A, USER_A,
)

pytestmark = pytest.mark.asyncio

CONNECT_ERP   = "/api/v1/integrations/"
DELETE_ERP    = f"/api/v1/integrations/{uuid4()}"
TRIGGER_SYNC  = f"/api/v1/integrations/{uuid4()}/sync"
LIST_MEMBERS  = "/api/v1/organizations/members"
INVITE_MEMBER = "/api/v1/organizations/members/invite"


def _make_mock_pool():
    """Minimal asyncpg pool mock — supports `async with pool.acquire() as conn:`.

    fetchrow returns None  → callers treat as "not found" and handle gracefully.
    fetch    returns []    → callers iterate over empty list safely.
    execute  returns str   → asyncpg execute() returns a command-status string.
    """
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    conn.fetch.return_value = []
    conn.execute.return_value = "UPDATE 0"
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = cm
    return pool


@pytest.fixture
async def client():
    # Role-guard tests: denied roles are rejected before DB access.
    # "Can reach" tests pass the guard and hit the DB layer — provide a mock
    # pool so the route returns an HTTP response rather than raising AttributeError.
    app.state.db_pool = _make_mock_pool()
    app.state.categorizer = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ── Connect ERP: requires owner | admin ───────────────────────────────────────

async def test_employee_cannot_connect_erp(client):
    token = make_token(USER_A, ORG_A, "employee")
    resp = await client.post(
        CONNECT_ERP,
        json={"provider": "tripletex", "credentials": {"consumer_token": "x", "employee_token": "y", "company_id": 1}},
        headers=bearer(token),
    )
    assert resp.status_code == 403


async def test_finance_cannot_connect_erp(client):
    token = make_token(USER_A, ORG_A, "finance")
    resp = await client.post(
        CONNECT_ERP,
        json={"provider": "tripletex", "credentials": {"consumer_token": "x", "employee_token": "y", "company_id": 1}},
        headers=bearer(token),
    )
    assert resp.status_code == 403


async def test_admin_can_reach_connect_erp_endpoint(client):
    """Admin passes the role guard — gets 422 (unknown provider) but NOT 403.

    We send an invalid provider so the route raises 422 before any DB access,
    keeping this test self-contained (no real DB or complex mock needed).
    """
    token = make_token(USER_A, ORG_A, "admin")
    resp = await client.post(
        CONNECT_ERP,
        json={"provider": "not_a_real_erp", "credentials": {}},
        headers=bearer(token),
    )
    assert resp.status_code != 403, f"Admin blocked by RBAC — should not happen"


# ── Delete ERP: requires owner | admin ────────────────────────────────────────

async def test_employee_cannot_delete_integration(client):
    token = make_token(USER_A, ORG_A, "employee")
    resp = await client.delete(DELETE_ERP, headers=bearer(token))
    assert resp.status_code == 403


async def test_finance_cannot_delete_integration(client):
    token = make_token(USER_A, ORG_A, "finance")
    resp = await client.delete(DELETE_ERP, headers=bearer(token))
    assert resp.status_code == 403


# ── Trigger sync: requires owner | admin | finance ────────────────────────────

async def test_employee_cannot_trigger_sync(client):
    token = make_token(USER_A, ORG_A, "employee")
    resp = await client.post(TRIGGER_SYNC, headers=bearer(token))
    assert resp.status_code == 403


async def test_finance_can_reach_sync_endpoint(client):
    """Finance passes the role guard — may get 404 (integration not found) but NOT 403."""
    token = make_token(USER_A, ORG_A, "finance")
    resp = await client.post(TRIGGER_SYNC, headers=bearer(token))
    assert resp.status_code != 403


# ── Invite member: requires owner | admin ─────────────────────────────────────

async def test_employee_cannot_invite_members(client):
    token = make_token(USER_A, ORG_A, "employee")
    resp = await client.post(
        INVITE_MEMBER,
        json={"email": "evil@corp.no", "role": "admin"},
        headers=bearer(token),
    )
    assert resp.status_code == 403


async def test_finance_cannot_invite_members(client):
    token = make_token(USER_A, ORG_A, "finance")
    resp = await client.post(
        INVITE_MEMBER,
        json={"email": "newguy@corp.no", "role": "employee"},
        headers=bearer(token),
    )
    assert resp.status_code == 403


# ── Read endpoints: any authenticated user ────────────────────────────────────

async def test_employee_can_list_integrations(client):
    """Employee token passes auth (not 401) and RBAC (not 403) for GET /integrations/."""
    token = make_token(USER_A, ORG_A, "employee")
    resp = await client.get(CONNECT_ERP, headers=bearer(token))
    assert resp.status_code not in (401, 403), (
        f"Employee blocked by auth/RBAC on read endpoint — got {resp.status_code}"
    )


async def test_unauthenticated_request_to_protected_route(client):
    """No token at all → 401, not 403 (must distinguish unauthenticated from unauthorized)."""
    resp = await client.get(CONNECT_ERP)
    assert resp.status_code == 401
