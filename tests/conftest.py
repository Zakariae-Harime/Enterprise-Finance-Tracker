"""
conftest.py - Shared fixtures for ALL tests.

Fixtures defined here are available to ALL tests automatically.
pytest discovers this file and injects fixtures by name.
"""
import pytest
import pytest_asyncio
import asyncpg
import json
from uuid import uuid4
from datetime import datetime, timezone


# Database URL for test environment (same as Docker service)
TEST_DB_URL = "postgresql://postgres:postgres@localhost:5432/finance_tracker"


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """
    Database connection pool shared across all tests.
    Created once, reused everywhere, closed at end.
    """
    pool = await asyncpg.create_pool(
        TEST_DB_URL,
        min_size=2,
        max_size=10
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture()
async def db_conn(db_pool):
    """
    Single database connection with auto-rollback.
    Each test gets a clean slate - changes are undone after test.

    Uses explicit acquire/release instead of 'async with' to avoid
    conflict between the context manager and pytest-asyncio's fixture teardown.
    """
    conn = await db_pool.acquire()
    tr = conn.transaction()
    await tr.start()
    yield conn
    await tr.rollback()
    await db_pool.release(conn)


@pytest.fixture
def make_event():
    """Factory fixture - creates test events with sensible defaults."""
    def _make_event(
        event_type: str = "TestEvent",
        aggregate_type: str = "test",
        aggregate_id=None,
        event_data=None,
        version: int = 1
    ):
        return {
            "event_id": uuid4(),
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id or uuid4(),
            "event_type": event_type,
            "event_data": event_data or {"test": "data"},
            "version": version,
            "created_at": datetime.now(timezone.utc)
        }
    return _make_event


@pytest.fixture
def make_dlq_message():
    """Factory fixture - creates test DLQ messages."""
    def _make_dlq(
        error_message: str = "Test error",
        error_category: str = "transient",
        consumer_name: str = "test_consumer"
    ):
        return {
            "event_id": str(uuid4()),
            "consumer_name": consumer_name,
            "error_message": error_message,
            "error_category": error_category,
            "original_event": json.dumps({"test": "data"}),
            "original_topic": "test-topic"
        }
    return _make_dlq
