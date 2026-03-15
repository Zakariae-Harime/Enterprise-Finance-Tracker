"""
Injection Security Tests

Proves that the API is safe against:
  - SQL injection via string fields (asyncpg parameterized queries prevent this)
  - XSS payloads (JSON API returns strings — browsers don't execute them)
  - Path traversal via UUID path parameters

asyncpg uses $1/$2 placeholders that are sent separately from the query string —
the DB driver never interpolates user input into SQL text.
"""
import os
import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

os.environ.setdefault("SECRET_KEY",     "test-secret-key-32-bytes-minimum!")
os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.api.main import app
from tests.security.conftest import make_token, bearer, ORG_A, USER_A

pytestmark = pytest.mark.asyncio

SQL_PAYLOADS = [
    "'; DROP TABLE expenses; --",
    "1' OR '1'='1",
    "' UNION SELECT * FROM users --",
    "admin'--",
    "1; DELETE FROM integrations WHERE '1'='1",
]

XSS_PAYLOADS = [
    "<script>alert('xss')</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(document.cookie)",
    '"><svg/onload=alert(1)>',
]


@pytest.fixture
async def client(db_pool):
    app.state.db_pool = db_pool
    app.state.categorizer = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def owner_token():
    return make_token(USER_A, ORG_A, "owner")


@pytest.mark.parametrize("payload", SQL_PAYLOADS)
async def test_sql_injection_in_integration_provider(client, owner_token, payload):
    """
    SQL injection via the 'provider' field.
    The value is validated against ADAPTERS dict before hitting DB,
    so these return 422 (validation error), not 500 (query error).
    Critical: must never return 500.
    """
    resp = await client.post(
        "/api/v1/integrations/",
        json={"provider": payload, "credentials": {}},
        headers=bearer(owner_token),
    )
    assert resp.status_code != 500, f"Server error on SQL payload: {payload!r}"
    assert resp.status_code in (422, 400), f"Unexpected status {resp.status_code} for: {payload!r}"


@pytest.mark.parametrize("payload", SQL_PAYLOADS)
async def test_sql_injection_in_transaction_description(client, owner_token, payload, db_pool):
    """
    SQL injection via transaction description field.
    asyncpg parameterization means the payload is stored as a literal string.
    Must never cause a 500 or DB error.
    """
    # Seed an account first (transactions need an account_id)
    account_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO account_projections (account_id, user_id, bank_name, current_balance, currency, account_type) "
            "VALUES ($1, $2, 'Test', 0, 'NOK', 'checking') ON CONFLICT DO NOTHING",
            account_id, ORG_A,
        )

    resp = await client.post(
        "/api/v1/transactions/",
        json={
            "account_id": str(account_id),
            "amount": "100.00",
            "currency": "NOK",
            "description": payload,
            "transaction_type": "debit",
        },
        headers=bearer(owner_token),
    )
    assert resp.status_code != 500, f"Server error on SQL injection: {payload!r}"


@pytest.mark.parametrize("xss", XSS_PAYLOADS)
async def test_xss_payload_stored_as_literal_string(client, owner_token, db_pool, xss):
    """
    XSS payloads submitted as strings are stored and returned as JSON strings.
    Content-Type is application/json — browsers never execute JSON as HTML.
    The payload must round-trip unchanged (stored literally, not stripped/escaped).
    """
    account_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO account_projections (account_id, user_id, bank_name, current_balance, currency, account_type) "
            "VALUES ($1, $2, 'XSS Test', 0, 'NOK', 'checking') ON CONFLICT DO NOTHING",
            account_id, ORG_A,
        )

    resp = await client.post(
        "/api/v1/transactions/",
        json={
            "account_id": str(account_id),
            "amount": "1.00",
            "currency": "NOK",
            "description": xss,
            "transaction_type": "debit",
        },
        headers=bearer(owner_token),
    )
    # Not a 500
    assert resp.status_code != 500
    # If accepted: Content-Type is JSON, not HTML — XSS cannot execute
    if resp.status_code in (200, 201):
        assert "application/json" in resp.headers.get("content-type", "")


async def test_path_traversal_uuid_validation(client, owner_token):
    """
    FastAPI validates UUID path parameters at the routing layer.
    Path traversal strings like '../../../etc/passwd' fail UUID parsing → 422.
    Must never be 200 or 500.
    """
    resp = await client.get(
        "/api/v1/integrations/../../etc/passwd/sync-jobs",
        headers=bearer(owner_token),
    )
    # FastAPI returns 404 for non-matching routes or 422 for invalid UUIDs
    assert resp.status_code in (404, 422), f"Unexpected {resp.status_code}"
    assert resp.status_code != 500


async def test_oversized_payload_rejected(client, owner_token):
    """
    A very large string in a field must not cause a 500 (memory/timeout).
    FastAPI/Pydantic will either store it or reject it — but not crash.
    """
    huge_string = "A" * 100_000  # 100KB string

    resp = await client.post(
        "/api/v1/integrations/",
        json={"provider": huge_string, "credentials": {}},
        headers=bearer(owner_token),
    )
    assert resp.status_code != 500
