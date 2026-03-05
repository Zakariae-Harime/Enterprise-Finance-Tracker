"""
Shared fixtures for E2E API tests.

api_client: in-process HTTP client — no running server needed.
  - ASGITransport sends requests directly to the FastAPI ASGI app
  - app.state.db_pool is set to the test pool (skips lifespan entirely)
  - No Kafka required — routes only use db_pool and event_store
"""
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app

@pytest_asyncio.fixture
async def api_client(db_pool):
    """
    Injects the test DB pool into the FastAPI app state, then yields
    an AsyncClient that sends requests in-process via ASGITransport.

    Why set app.state.db_pool directly?
      get_db_pool()       → reads request.app.state.db_pool
      get_event_store()   → reads request.app.state.db_pool → EventStore(pool)
    Both dependencies resolve correctly once state is set.
    """
    app.state.db_pool = db_pool
    app.state.categorizer = None  # Skip 170MB model load in tests — auto-categorization disabled

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
