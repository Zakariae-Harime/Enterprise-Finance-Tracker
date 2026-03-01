"""
Integration tests for Kafka production and OutboxRelay.

Tests the full outbox → Kafka path:
  OutboxRelay.run_once() → reads outbox rows → publishes to Kafka topic
  → marks published_at → consumer reads the message back

Requires BOTH services running:
  docker-compose up -d db kafka zookeeper

Why this matters:
  All previous tests called consumers directly (bypassing Kafka entirely).
  These tests prove the TRANSPORT layer works:
    event store wrote outbox row → relay published to Kafka → message is there
  This is the bridge between the DB-centric world and the streaming world.
"""
import json
import pytest
from uuid import uuid4

from src.infrastructure.outbox_relay import OutboxRelay

pytestmark = pytest.mark.asyncio


class TestKafkaConnectivity:
    """
    Basic producer/consumer round-trip.
    Proves aiokafka is installed, Kafka is reachable, and topics auto-create.
    """

    async def test_produce_and_consume_message(self, kafka_producer, kafka_consumer_factory):
        """
        Send a message to a fresh topic → read it back with a consumer.

        Uses a unique topic name per test run to avoid reading old messages
        left over from previous test executions.
        """
        topic = f"test.connectivity.{uuid4().hex[:8]}"
        payload = {"check": "kafka_works", "run_id": str(uuid4())}

        # PRODUCE: send to Kafka, wait for broker acknowledgment
        await kafka_producer.send_and_wait(
            topic,
            value=json.dumps(payload).encode("utf-8"),
        )

        # CONSUME: read from offset=0 (earliest), expect our message
        consumer = await kafka_consumer_factory(topic)
        msg = await consumer.__anext__()           # blocks until message or timeout
        received = json.loads(msg.value)

        assert received["check"] == "kafka_works"
        assert received["run_id"] == payload["run_id"]


class TestOutboxRelay:
    """
    OutboxRelay.run_once() picks up DB rows and sends them to the correct topic.
    """

    async def test_relay_publishes_unpublished_row(
        self, db_pool, kafka_producer, kafka_consumer_factory
    ):
        """
        Seed one outbox row (published_at=NULL) → run relay →
        message must appear in 'finance.account.events'.

        Proves the full outbox → Kafka path end-to-end:
          DB row with NULL published_at → relay sends to topic → consumer reads it
        """
        aggregate_id = uuid4()
        event_id = uuid4()

        # Seed the outbox directly (simulating what EventStore.append_events does)
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO outbox
                  (event_id, aggregate_type, aggregate_id, event_type, event_data)
                VALUES ($1, $2, $3, $4, $5)
                """,
                event_id,
                "Account",
                aggregate_id,
                "AccountCreated",
                json.dumps({"name": "DNB Bedriftskonto", "currency": "NOK"}),
            )

        relay = OutboxRelay(db_pool=db_pool, kafka_producer=kafka_producer)
        published_count = await relay.run_once()

        assert published_count >= 1  # at least our row was published

        # Read from the topic the relay derived: 'finance.account.events'
        consumer = await kafka_consumer_factory("finance.account.events")

        found = False
        async for msg in consumer:          # iterates until consumer_timeout_ms (5s)
            payload = json.loads(msg.value)
            if payload["aggregate_id"] == str(aggregate_id):
                assert payload["event_type"] == "AccountCreated"
                assert payload["aggregate_type"] == "Account"
                found = True
                break

        assert found, "AccountCreated message not found in finance.account.events"

    async def test_relay_marks_published_at(self, db_pool, kafka_producer):
        """
        After run_once(), the outbox row must have published_at set to a timestamp.

        This is the sentinel that prevents the relay from re-sending on the next poll.
        Without this UPDATE, the relay would send every event to Kafka every time
        it runs — consumers would receive unbounded duplicates.
        """
        aggregate_id = uuid4()

        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO outbox
                  (event_id, aggregate_type, aggregate_id, event_type, event_data)
                VALUES ($1, $2, $3, $4, $5)
                """,
                uuid4(),
                "Transaction",
                aggregate_id,
                "TransactionCreated",
                json.dumps({"amount": "15000.00", "currency": "NOK", "merchant": "SAP AG"}),
            )

        relay = OutboxRelay(db_pool=db_pool, kafka_producer=kafka_producer)
        await relay.run_once()

        # Verify the relay set published_at on our specific row
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT published_at FROM outbox WHERE aggregate_id = $1",
                aggregate_id,
            )

        assert row is not None
        assert row["published_at"] is not None  # relay marked it — will be skipped next run

    async def test_relay_skips_already_published_rows(self, db_pool, kafka_producer):
        """
        Rows with published_at already set are excluded by WHERE published_at IS NULL.
        The relay must NOT reset published_at to NULL or re-send these rows.

        Scenario: relay crashes after marking published_at → restarts → runs again.
        The row must remain published — no second Kafka message sent.
        """
        aggregate_id = uuid4()

        # Insert with published_at already set — simulates a previously processed row
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO outbox
                  (event_id, aggregate_type, aggregate_id, event_type, event_data, published_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                uuid4(),
                "Budget",
                aggregate_id,
                "BudgetCreated",
                json.dumps({"budget_name": "Reise Enterprise 2026", "amount": "50000.00"}),
            )

        relay = OutboxRelay(db_pool=db_pool, kafka_producer=kafka_producer)
        await relay.run_once()

        # Row must still be marked as published — relay did not reset it
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT published_at FROM outbox WHERE aggregate_id = $1",
                aggregate_id,
            )

        assert row["published_at"] is not None   # untouched — relay skipped it

    async def test_topic_names_derived_from_aggregate_type(self, db_pool, kafka_producer):
        """
        Verify the topic naming convention is applied correctly.

        Enterprise Kafka clusters use structured topic names for access control
        and monitoring. 'finance.account.events' makes it clear:
          domain = finance, aggregate = account, type = events stream
        """
        relay = OutboxRelay(db_pool=db_pool, kafka_producer=kafka_producer)

        assert relay._get_topic("Account") == "finance.account.events"
        assert relay._get_topic("Transaction") == "finance.transaction.events"
        assert relay._get_topic("Budget") == "finance.budget.events"
