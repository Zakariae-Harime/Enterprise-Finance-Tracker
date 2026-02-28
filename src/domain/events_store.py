#EventStore with Outbox Pattern
"""sumary_line
  This module handles:
  1. Saving events to the database (event sourcing)
  2. Writing to outbox table (guaranteed Kafka delivery)
  3. Optimistic concurrency control (version checking)
  4. Loading event history for an aggregate
  """
from aiokafka import AIOKafkaProducer # For Kafka integration - aiokafka = Async I/O Kafka- Works with async/await (matches our EventStore)- Non-blocking: can send messages while doing other work
import json
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID
from typing import Optional
import asyncpg
  # Import our domain events from __init__.py
from src.domain import DomainEvent, EventMetadata
# CUSTOM JSON ENCODER
"""
    Custom JSON encoder for domain objects.

    Python's json.dumps() fails on:
      - UUID → we convert to string
      - Decimal → we convert to string (preserves precision!)
      - datetime → we convert to ISO format string
      - Enum → we convert to its .value
"""
class EventEncoder(json.JSONEncoder):
    def default (self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        #string preserves exact value
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, EventMetadata):
          return asdict(obj)
        if isinstance(obj, datetime):
              return obj.isoformat()
        if hasattr(obj, 'value'):
            return obj.value
       # Unknown type → let parent class handle (raises TypeError)
        return super().default(obj)
  #Concurrency error catch
"""Custom exception for optimistic concurrency control violations.
     This exception fires when two processes try to modify the same aggregate simultaneously:
      Custom exceptions allow:
        1. Specific error handling in calling code
        2. Clear error messages for debugging
        3. Different recovery strategies per error type
     """
class ConcurrencyError(Exception):
          """
      Raised when optimistic concurrency check fails.

      This happens when:
        - Process A reads aggregate at version 5
        - Process B reads aggregate at version 5
        - Process B saves, making version 6
        - Process A tries to save expecting version 5 → CONFLICT!

      Recovery strategy: Reload aggregate and retry the operation
      """
          def __init__(self, aggregate_id: UUID, expected: int, actual: int):
              self.aggregate_id = aggregate_id
              self.expected_version = expected
              self.actual_version = actual
              self.message = f"Concurrency error for aggregate {aggregate_id}: expected version {expected}, got version {actual}"
              super().__init__(self.message)
class AggregateNotFoundError(Exception):
  """
             - Invalid aggregate_id passed to load_events()
        - Aggregate was deleted (in event sourcing, we'd have a "Deleted" event)
        - Wrong tenant_id (multi-tenant isolation)

      Recovery strategy: Return 404 to client or create new aggregate
  """
  def __init__(self, aggregate_id: UUID, aggregate_type: str):
        self.aggregate_id = aggregate_id
        self.aggregate_type = aggregate_type
        self.message = f"{aggregate_type} with ID {aggregate_id} not found."
        super().__init__(self.message)

#Event Store Class
  """
      The EventStore is the SINGLE SOURCE OF TRUTH for all domain events.

      Key responsibilities:
        1. append_events() - Save new events with optimistic concurrency
        2. load_events() - Retrieve event history for an aggregate
        3. Outbox writes - Guarantee Kafka delivery (same transaction)

      Architecture pattern: Repository pattern for events
  """

class EventStore:
           """
      Async event store with outbox pattern for guaranteed delivery.

      Uses asyncpg connection pool for:
        - Efficient connection reuse (don't create new connection per query)
        - Automatic connection cleanup
        - Configurable pool size based on load
      """
           def __init__(self, db_pool: asyncpg.pool.Pool):
               self.db_pool = db_pool
           """
          Initialize with a connection pool.

          Why pool instead of single connection?
            - Single connection = bottleneck (one query at a time)
            - Pool = multiple concurrent queries
            - Pool handles connection lifecycle automatically

          Args:
              pool: asyncpg connection pool (created at app startup)
                """
           async def append_events(
               self,
               aggregate_id: UUID, #The entity these events belong to (e.g., account_id)
               aggregate_type: str, #Type of aggregate (e.g., "Account", "Order")
               expected_version: int, #Optimistic concurrency check
               new_events: list[DomainEvent], #Events to append
               tenant_id: UUID #For multi-tenant isolation
           ) -> int: #Returns new version after appending
               """
          Append events to the event store with optimistic concurrency.

          CRITICAL: This method does TWO things in ONE transaction:
            1. Insert events into 'events' table
            2. Insert into 'outbox' table for Kafka delivery

               If either fails, BOTH are rolled back (atomicity).
               """
               async with self.db_pool.acquire() as conn:
                   async with conn.transaction():
                       # Check current version
                       current_version = await conn.fetchval(
                           """
                           SELECT COALESCE(MAX(version), 0)
                           FROM events
                           WHERE aggregate_id = $1 AND tenant_id = $2
                           """,
                           aggregate_id,
                           tenant_id
                       )
                       # Step 2: Verify expected version matches current version
                       if current_version != expected_version:
                           raise ConcurrencyError(
                               aggregate_id,
                               expected_version,
                               current_version
                           ) 
                       new_version = expected_version
                       for event in new_events:
                           new_version += 1
                           event_data = json.dumps(asdict(event), cls=EventEncoder)
                           # Insert into events table
                           await conn.execute(
                               """
                               INSERT INTO events (
                                   event_id,
                                   aggregate_id,
                                   aggregate_type,
                                   event_type,
                                   event_data,
                                   version,
                                   tenant_id,
                                   created_at
                               ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                               """,
                               event.metadata.event_id,
                               aggregate_id,
                               aggregate_type,
                               event.__class__.__name__,
                               event_data,
                               new_version,
                               tenant_id,
                               event.metadata.timestamp
                           )

                           # OUTBOX INSERT (same transaction = guaranteed delivery)
                           # If event insert succeeds but outbox fails → BOTH rollback
                           # If both succeed → Kafka relay picks up from outbox later
                           await conn.execute(
                               """
                               INSERT INTO outbox (
                                   event_id,
                                   aggregate_type,
                                   aggregate_id,
                                   event_type,
                                   event_data,
                                   tenant_id
                               ) VALUES ($1, $2, $3, $4, $5, $6)
                               """,
                               event.metadata.event_id,
                               aggregate_type,
                               aggregate_id,
                               event.__class__.__name__,
                               event_data,
                               tenant_id
                           )
                       return new_version

           async def load_events(
               self,
               aggregate_id: UUID,
               aggregate_type: str,
               tenant_id: UUID) -> list[dict]:           
             """
          Load all events for an aggregate in chronological order.

          Used to reconstruct aggregate state by replaying events.

          Args:
              aggregate_id: The entity to load events for
              tenant_id: Organization ID (multi-tenant isolation)
              aggregate_type: Type name for error messages

          Returns:
              List of event records as dictionaries

          Raises:
              AggregateNotFoundError: If no events exist for this aggregate
           """
             async with self.db_pool.acquire() as conn:
                 rows = await conn.fetch(
                     """
                     SELECT event_id, event_type, event_data, version, created_at
                     FROM events
                     WHERE aggregate_id = $1 AND tenant_id = $2
                     ORDER BY version ASC
                    """,
                    # Why ORDER BY version ASC?
                    # - Events MUST be replayed in exact order they happened
                    # - Depositing then withdrawing ≠ withdrawing then depositing
                    # - Version number guarantees correct sequence even if timestamps are identical
                     aggregate_id,
                     tenant_id
                 )
                 if not rows:   # If no events found, aggregate doesn't exist
                     raise AggregateNotFoundError(aggregate_id, aggregate_type)
                 events = []
                 for row in rows:
                     event_dict = {
                         "event_id": row["event_id"],
                         "event_type": row["event_type"],
                         "event_data": json.loads(row["event_data"]),
                         "version": row["version"],
                         "created_at": row["created_at"]
                     }
                     events.append(event_dict)
                 return events

# OUTBOX RELAY CLASS
"""
The OutboxRelay bridges the gap between database and Kafka.

      Why separate from EventStore?
        - Single Responsibility: EventStore saves, OutboxRelay delivers
        - Can run independently (background worker)
        - Can retry without affecting main application

      This implements the "Polling Publisher" pattern:
        - Poll outbox table periodically
        - Send to Kafka
        - Delete after successful send

Guarantees "at-least-once" delivery to Kafka.
"""
class OutboxRelay:
    """
    Reads from outbox table and publishes to Kafka.
    Runs as background worker to forward messages from outbox table to Kafka.
    """

    def __init__(self, db_pool: asyncpg.pool.Pool, kafka_producer: AIOKafkaProducer):
        self.db_pool = db_pool  # Same database pool as EventStore (shared resource)
        self.kafka_producer = kafka_producer  # Configured Kafka producer instance

    async def process_outbox(self, batch_size: int = 100) -> int:
        """
        Reads pending messages and sends them to Kafka.

        Args:
            batch_size: Maximum messages to process per run (default 100)

        Returns:
            Number of successfully processed messages
        """
        async with self.db_pool.acquire() as conn:
            # Step 1: Fetch pending messages (oldest first - FIFO)
            rows = await conn.fetch(
                """
                SELECT id, event_id, aggregate_type, event_type, event_data, tenant_id
                FROM outbox
                ORDER BY created_at ASC
                LIMIT $1
                """,
                batch_size
            )

            if not rows:
                return 0  # No messages to process

            processed_count = 0

            for row in rows:
                try:
                    # Step 2: Build Kafka topic name from aggregate and event type
                    topic = f"finance.{row['aggregate_type'].lower()}.events"

                    # Step 3: Send to Kafka (wait for acknowledgment)
                    await self.kafka_producer.send_and_wait(
                        topic=topic,
                        key=str(row['event_id']).encode('utf-8'),
                        value=row['event_data'].encode('utf-8')
                    )

                    # Step 4: Delete only AFTER successful Kafka send
                    await conn.execute(
                        "DELETE FROM outbox WHERE id = $1",
                        row['id']
                    )

                    processed_count += 1

                except Exception as e:
                    # Failed message stays in outbox for retry
                    print(f"Error processing outbox message {row['id']}: {e}")
                    continue

            return processed_count
                  