"""
Integration tests for EventStore → Outbox atomicity.
REQUIRES: docker-compose up -d db

Tests the core guarantee: append_events() writes to BOTH
the events table AND the outbox table in one transaction.
If this guarantee breaks, Kafka never receives the event
and projections silently fall out of sync.
"""
import pytest
from uuid import uuid4
from decimal import Decimal

from src.domain.events_store import EventStore, ConcurrencyError
from src.domain import EventMetadata, AccountCreated, Currency, AccountType

pytestmark = pytest.mark.asyncio


class TestEventStoreAtomicity:
    """
    Proves the outbox pattern guarantee:
    events table write + outbox table write = one atomic transaction.
    """

    async def test_append_event_creates_outbox_row(self, db_pool):
        """
        After append_events(), an outbox row must exist.
        This row is what OutboxRelay reads to publish to Kafka.
        No outbox row = no Kafka message = broken projection pipeline.
        """
        # Arrange
        event_store = EventStore(db_pool)
        account_id = uuid4()
        tenant_id = uuid4()

        event = AccountCreated(
            aggregate_id=account_id,
            metadata=EventMetadata(),
            account_name="REMA Business Account",
            account_type=AccountType.CHECKING,
            currency=Currency.NOK,
            initial_balance=Decimal("10000.00"),
        )

        # Act
        await event_store.append_events(
            aggregate_id=account_id,
            aggregate_type="Account",
            new_events=[event],
            expected_version=0,
            tenant_id=tenant_id,
        )

        # Assert — query outbox directly to confirm row was written
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM outbox WHERE aggregate_id = $1",
                account_id,
            )

        assert row is not None
        assert row["event_type"] == "AccountCreated"
        assert row["aggregate_type"] == "Account"
        assert row["published_at"] is None   # not yet picked up by OutboxRelay

    async def test_outbox_row_has_correct_aggregate_type(self, db_pool):
        """
        OutboxRelay derives the Kafka topic from aggregate_type:
        "Account" -> "finance.account.events"
        "Transaction" -> "finance.transaction.events"
        The outbox row must store aggregate_type exactly as passed.
        """
        event_store = EventStore(db_pool)
        account_id = uuid4()
        tenant_id = uuid4()

        event = AccountCreated(
            aggregate_id=account_id,
            metadata=EventMetadata(),
            account_name="Test Account",
            account_type=AccountType.SAVINGS,
            currency=Currency.NOK,
        )

        await event_store.append_events(
            aggregate_id=account_id,
            aggregate_type="Account",
            new_events=[event],
            expected_version=0,
            tenant_id=tenant_id,
        )

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT aggregate_type FROM outbox WHERE aggregate_id = $1",
                account_id,
            )

        assert row["aggregate_type"] == "Account"

    async def test_concurrency_error_creates_no_outbox_row(self, db_pool):
        """
        If expected_version is wrong (optimistic concurrency conflict),
        the entire transaction rolls back — NO event row, NO outbox row.
        This proves atomicity: it's all or nothing.
        """
        event_store = EventStore(db_pool)
        account_id = uuid4()
        tenant_id = uuid4()

        event = AccountCreated(
            aggregate_id=account_id,
            metadata=EventMetadata(),
            account_name="Conflict Test",
            account_type=AccountType.SAVINGS,
            currency=Currency.NOK,
        )

        # First append succeeds (version 0 -> 1)
        await event_store.append_events(
            aggregate_id=account_id,
            aggregate_type="Account",
            new_events=[event],
            expected_version=0,
            tenant_id=tenant_id,
        )

        # Second append with expected_version=0 again — should fail
        with pytest.raises(ConcurrencyError):
            await event_store.append_events(
                aggregate_id=account_id,
                aggregate_type="Account",
                new_events=[event],
                expected_version=0,   # wrong — current version is 1
                tenant_id=tenant_id,
            )

        # Only 1 outbox row should exist (from the first successful append)
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM outbox WHERE aggregate_id = $1",
                account_id,
            )

        assert count == 1
