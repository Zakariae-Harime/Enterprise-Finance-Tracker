"""
OutboxRelay — Polls the outbox table and publishes events to Kafka.

The outbox pattern (Transactional Outbox) solves the dual-write problem:
  WRONG approach:  write to DB  →  send to Kafka   (two separate operations)
                   crash here ↑    message is lost
  CORRECT:         write DB + outbox atomically (EventStore does this)
                   relay polls outbox → publishes to Kafka → marks published_at

Guarantee: At-least-once delivery to Kafka.
  If the relay crashes AFTER publishing but BEFORE marking published_at,
  the row is still NULL → sent again on next poll.
  This is acceptable because consumers use IdempotentConsumer for deduplication.

Topic naming convention:
  aggregate_type 'Account'     → topic 'finance.account.events'
  aggregate_type 'Transaction' → topic 'finance.transaction.events'
  aggregate_type 'Budget'      → topic 'finance.budget.events'
"""
import json
import asyncpg
from aiokafka import AIOKafkaProducer


class OutboxRelay:
    """
    Polls outbox for unpublished rows and sends them to Kafka.

    run_once() is designed to be called in a loop (e.g., every 100ms by a scheduler).
    Each call fetches ALL unpublished rows, publishes, then marks them done.
    """

    def __init__(self, db_pool: asyncpg.Pool, kafka_producer: AIOKafkaProducer):
        self.db_pool = db_pool
        self.kafka_producer = kafka_producer

    def _get_topic(self, aggregate_type: str) -> str:
        """
        Derive Kafka topic from aggregate type.
        Convention: lowercase, prefixed with domain name.
        """
        return f"finance.{aggregate_type.lower()}.events"

    async def run_once(self) -> int:
        """
        Fetch all unpublished outbox rows, publish each to Kafka, mark as published.

        ORDER BY created_at ASC guarantees chronological order within a topic.
        This matters for event sourcing: AccountCreated must reach Kafka before
        TransactionCreated for the same account, so consumers replay correctly.

        Returns:
            Number of messages successfully published in this run.
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, event_id, aggregate_type, aggregate_id,
                       event_type, event_data
                FROM outbox
                WHERE published_at IS NULL
                ORDER BY created_at ASC
                """
            )

        published = 0
        for row in rows:
            topic = self._get_topic(row["aggregate_type"])

            # asyncpg returns JSONB as a dict, but guard against string just in case
            event_data = row["event_data"]
            if isinstance(event_data, str):
                event_data = json.loads(event_data)

            # Kafka message envelope — consumers parse this to get the domain event
            message = {
                "event_id": str(row["event_id"]) if row["event_id"] else None,
                "aggregate_type": row["aggregate_type"],
                "aggregate_id": str(row["aggregate_id"]),
                "event_type": row["event_type"],
                "event_data": event_data,
            }

            # key = aggregate_id ensures all events for the same aggregate go to
            # the same Kafka partition → ordering preserved per aggregate
            await self.kafka_producer.send_and_wait(
                topic,
                value=json.dumps(message).encode("utf-8"),
                key=str(row["aggregate_id"]).encode("utf-8"),
            )

            # Mark as published immediately after successful send.
            # If crash here, row stays NULL → safely resent on next run.
            # Consumers handle duplicates via IdempotentConsumer.
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE outbox SET published_at = NOW() WHERE id = $1",
                    row["id"],
                )

            published += 1

        return published
