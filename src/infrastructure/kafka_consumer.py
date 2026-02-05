"""
Idempotent Kafka Consumer Base Class

Problem: Kafka guarantees at-least-once delivery, meaning
         the same message might be delivered multiple times.

Solution: Track processed events in database. Before processing,
          check if already handled. This converts:

          At-least-once DELIVERY → Exactly-once PROCESSING

Pattern used by: Vipps, DNB, Stripe, Netflix
"""
import asyncpg
from abc import ABC, abstractmethod
import json
from uuid import UUID
from typing import Optional

class IdempotentConsumer(ABC):
    """
    Abstract base class for Kafka consumers with deduplication.

    How to use:
      1. Inherit from this class
      2. Implement process_event() method
      3. Call handle_event() when message arrives from Kafka
    """
    def __init__(self, db_pool: asyncpg.Pool, consumer_name: str):
        """
        Initialize the consumer.

        Args:
            pool: Database connection pool (shared with other components)
            consumer_name: Unique identifier like 'email_service', 'analytics'
                          Used in processed_events table to track what THIS
                          consumer has processed.
        """
        self.db_pool = db_pool
        self.consumer_name = consumer_name

    async def handle_event(self, event_id: UUID, event_type: str, event_data: dict) -> bool:
        """
        Handle an incoming event with deduplication.

        Steps:
          1. Check processed_events table for (event_id, consumer_name)
          2. If exists, skip processing (already handled)
          3. If not, call process_event() to handle it
          4. Record (event_id, consumer_name) in processed_events

        Args:
            event_id: Unique ID of the event/message
            event_type: Type/category of the event
            event_data: Actual event payload as dict
        """
        async with self.db_pool.acquire() as conn:
            already_processed = await conn.fetchval(
                # Check if already processed
                """
                SELECT 1 FROM processed_events
                WHERE event_id=$1 AND consumer_name=$2
                """,
                event_id,
                self.consumer_name
            )
            if already_processed:
                print(f"Event {event_id} already processed by {self.consumer_name}, skipping.")
                return False  # Indicate event was skipped
            try :
                await self.process_event(event_type, event_data)
                # Record as processed
            except Exception as e:
                 # If processing fails, DON'T mark as processed
                 # Message will be redelivered and retried
                 print(f" [{self.consumer_name}] event_id {event_id}: {e}")
                 raise # Re-raise so Kafka retries
            await conn.execute(
                """
                INSERT INTO processed_events (event_id, consumer_name, processed_at)
                VALUES ($1, $2, NOW())
                """,
                event_id,
                self.consumer_name
            )
            print(f"Event {event_id} processed and recorded by {self.consumer_name}.")
            return True  # Indicate event was processed
    @abstractmethod
    async def process_event(self, event_type: str, event_data: dict):
        """
        Abstract method to process the event.
        Must be implemented by subclasses.
        """
        pass

