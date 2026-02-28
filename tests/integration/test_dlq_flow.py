"""
Integration tests for idempotent consumer deduplication + DLQ messages.
REQUIRES: docker-compose up -d db

Tests:
  - processed_events table tracks what each consumer has handled
  - Duplicate event_id + consumer_name is rejected (composite PK)
  - DLQ messages are stored with correct defaults
"""
import pytest
from uuid import uuid4

pytestmark = pytest.mark.asyncio


class TestProcessedEvents:
    """
    The processed_events table is what makes consumers idempotent.
    (event_id, consumer_name) composite PK = same event processed by
    two different consumers is fine, same consumer twice is rejected.
    """

    async def test_can_record_processed_event(self, db_conn):
        """
        After a consumer processes an event, it INSERTs into processed_events.
        This is the mechanism that prevents re-processing on Kafka redeliver.
        """
        event_id = uuid4()

        await db_conn.execute(
            """
            INSERT INTO processed_events (event_id, consumer_name, processed_at)
            VALUES ($1, $2, NOW())
            """,
            event_id,
            "account_projection_service",
        )

        row = await db_conn.fetchrow(
            "SELECT * FROM processed_events WHERE event_id = $1",
            event_id,
        )

        assert row is not None
        assert row["consumer_name"] == "account_projection_service"

    async def test_same_event_two_consumers_both_allowed(self, db_conn):
        """
        SAME event_id processed by TWO different consumers = allowed.
        composite PK is (event_id, consumer_name) not just event_id.

        Real scenario: AccountCreated processed by BOTH
        account_projection_service AND email_service independently.
        """
        event_id = uuid4()

        # Both consumers process the same event
        await db_conn.execute(
            "INSERT INTO processed_events (event_id, consumer_name, processed_at) VALUES ($1, $2, NOW())",
            event_id, "account_projection_service",
        )
        await db_conn.execute(
            "INSERT INTO processed_events (event_id, consumer_name, processed_at) VALUES ($1, $2, NOW())",
            event_id, "email_service",
        )

        count = await db_conn.fetchval(
            "SELECT COUNT(*) FROM processed_events WHERE event_id = $1", event_id
        )
        assert count == 2

    async def test_same_event_same_consumer_rejected(self, db_conn):
        """
        SAME event_id + SAME consumer = rejected by composite PK.
        This is the deduplication guarantee — Kafka can redeliver,
        but the consumer will never process it twice.
        """
        event_id = uuid4()

        await db_conn.execute(
            "INSERT INTO processed_events (event_id, consumer_name, processed_at) VALUES ($1, $2, NOW())",
            event_id, "account_projection_service",
        )

        with pytest.raises(Exception):   # asyncpg raises on PK violation
            await db_conn.execute(
                "INSERT INTO processed_events (event_id, consumer_name, processed_at) VALUES ($1, $2, NOW())",
                event_id, "account_projection_service",  # same consumer, same event
            )


class TestDLQMessages:
    """
    DLQ messages are written when an event fails max_retries times.
    Tests that the table structure supports all the fields the consumers write.
    """

    async def test_dlq_message_stored_with_pending_status(self, db_conn, make_dlq_message):
        """
        DLQ messages start as 'pending' — waiting for a human or
        automated processor to decide: retry, replay, or discard.
        """
        dlq = make_dlq_message(
            error_message="asyncpg.exceptions.ConnectionFailureError: connection refused",
            error_category="transient",
            consumer_name="account_projection_service",
        )

        await db_conn.execute(
            """
            INSERT INTO dlq_messages
                (event_id, consumer_name, error_message, error_category, original_event, original_topic)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            dlq["event_id"], dlq["consumer_name"],
            dlq["error_message"], dlq["error_category"],
            dlq["original_event"], dlq["original_topic"],
        )

        row = await db_conn.fetchrow(
            "SELECT * FROM dlq_messages WHERE event_id = $1", dlq["event_id"]
        )

        assert row["status"] == "pending"
        assert row["error_category"] == "transient"
        assert row["retries_counter"] == 0

    async def test_dlq_unique_per_event_consumer(self, db_conn, make_dlq_message):
        """
        (event_id, consumer_name) is a UNIQUE INDEX on dlq_messages.
        The same failed event from the same consumer can't be inserted twice.
        """
        dlq = make_dlq_message()

        await db_conn.execute(
            "INSERT INTO dlq_messages (event_id, consumer_name, error_message, error_category, original_event, original_topic) VALUES ($1, $2, $3, $4, $5, $6)",
            dlq["event_id"], dlq["consumer_name"],
            dlq["error_message"], dlq["error_category"],
            dlq["original_event"], dlq["original_topic"],
        )

        with pytest.raises(Exception):   # UNIQUE constraint violation
            await db_conn.execute(
                "INSERT INTO dlq_messages (event_id, consumer_name, error_message, error_category, original_event, original_topic) VALUES ($1, $2, $3, $4, $5, $6)",
                dlq["event_id"], dlq["consumer_name"],   # same event + consumer
                "second error", "permanent",
                dlq["original_event"], dlq["original_topic"],
            )
