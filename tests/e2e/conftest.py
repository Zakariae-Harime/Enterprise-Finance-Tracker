"""
Shared fixtures for E2E API tests.

api_client:    in-process HTTP client — no running server needed.
auth_override: replaces get_current_user with a fixed UserContext — no JWT needed in tests.
"""
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from uuid import UUID

from src.api.main import app
from src.auth.dependencies import get_current_user, UserContext

# Fixed test identity — same tenant as the test DB seed data
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
TEST_ORG_ID  = UUID("00000000-0000-0000-0000-000000000001")

def _fake_user() -> UserContext:
    """Returns a hardcoded UserContext — bypasses JWT entirely in tests."""
    return UserContext(
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        role="owner",
    )


@pytest_asyncio.fixture
async def api_client(db_pool):
    """
    Injects the test DB pool into app state + overrides auth dependency.

    dependency_overrides is a dict on the FastAPI app:
      key   = the real dependency function
      value = the replacement callable (same return type, no Depends())

    FastAPI checks this dict before calling the real dependency.
    Every route that does Depends(get_current_user) gets _fake_user() instead.
    """
    app.state.db_pool = db_pool
    app.state.categorizer = None
    app.dependency_overrides[get_current_user] = _fake_user  # <-- the fix

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()  # clean up after each test
