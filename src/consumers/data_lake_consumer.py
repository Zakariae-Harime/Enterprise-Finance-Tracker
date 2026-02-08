"""
  Data Lake Consumer

  Listens to Kafka events and uploads them to Azure Data Lake Bronze layer.
  This is the entry point for raw event data into the Medallion Architecture.

  Flow:
    Kafka Topic → DataLakeConsumer → Bronze Layer (Parquet files)

  Why batch events?
    - Writing one Parquet file per event is inefficient
    - We collect events for a time window, then write as batch
    - This reduces I/O operations and storage costs
"""
import asyncio
import asyncpg
import json
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID,uuid4
from aiokafka import AIOKafkaConsumer
from src.infrastructure.kafka_consumer import IdempotentConsumer
from src.infrastructure.data_lake_client import DataLakeClient

class DataLakeConsumer(IdempotentConsumer):
    """
      Kafka consumer that uploads events to Data Lake Bronze layer.

      Inherits from IdempotentConsumer for exactly-once processing.
      Batches events before writing to reduce Parquet file count.
    """
    def __init__(self, db_pool :asyncpg.Pool, data_lake_client: DataLakeClient, batch_size: int = 100, flush_interval_seconds: int = 60):
        """
          Initialize Data Lake consumer.

          Args:
              db_pool: Database connection pool (for idempotency tracking)
              data_lake_client: Client for Azure Data Lake operations
              batch_size: Number of events to collect before writing (default: 100)
              flush_interval_seconds: Max seconds to wait before writing (default: 60)

          two thresholds :
              - batch_size: Write when we have enough events (throughput)
              - flush_interval: Write even if batch not full (latency)
              - Whichever comes first triggers the write
          """
        super().__init__(db_pool,consumer_name="data_lake_service")
        self.data_lake_client = data_lake_client
        self.batch_size = batch_size
        self._flush_interval = flush_interval_seconds
          # Event buffer: groups events by type for separate Parquet files
          # group by type so each event type has different schema
          # Example: {"TransactionCreated": [event1, event2], "AccountCreated": [event3]}          # Key: event_type, Value: list of event dicts
        self._event_buffer: dict[str, List[dict]] = {}
        self._last_flush_time = datetime.now(timezone.utc)

    async def process_event(self, event_type: str, event_data: dict) -> None:
        """
        Process incoming event by adding to buffer.

        This method is called by IdempotentConsumer.handle_event() AFTER
        checking that this event hasn't been processed before.

        Args:
            event_type: Type like 'TransactionCreated', 'AccountCreated'
            event_data: The full event payload as dictionary

        Flow:
            1. Add event to buffer (grouped by type)
            2. Check if we should flush (batch full OR time elapsed)
            3. If yes, write to Data Lake
        """
        # Initialize buffer for this event type if first time seeing it
        if event_type not in self._event_buffer:
            self._event_buffer[event_type] = []

        # Add event to the appropriate buffer
        self._event_buffer[event_type].append(event_data)

        # Calculate if we should flush
        buffer_size = len(self._event_buffer[event_type])
        time_elapsed = (datetime.now(timezone.utc) - self._last_flush_time).seconds

        # Flush if EITHER condition is met:
        # 1. Buffer is full (batch_size reached)
        # 2. Too much time has passed (flush_interval reached)
        should_flush = buffer_size >= self._batch_size or time_elapsed >= self._flush_interval

        if should_flush:
            await self._flush_buffer(event_type)
    async def _flush_buffer(self, event_type: str) -> None:
        """
        Write buffered events of a specific type to Data Lake as Parquet.

        Args:
            event_type: The type of events to flush (e.g., 'TransactionCreated')

        Flow:
            1. Get events from buffer for this type
            2. Write to Data Lake using DataLakeClient
            3. Clear buffer and update last flush time
            separate method :
              - Single Responsibility: process_event adds, _flush_buffer writes
              - Reusable: Called from process_event AND flush_all
              - Testable: Can test flush logic independently
        """
        events = self._event_buffer.get(event_type, [])
        if not events:
            return  # Nothing to flush

        # Generate unique batch ID using uuid4 (random UUID)
        batch_id = uuid4()
        try:
            # Upload to Bronze layer using our DataLakeClient
            # This creates a Parquet file at:
            # bronze/events/{event_type}/year=YYYY/month=MM/day=DD/{timestamp}.parquet
            path = await self._data_lake_client.upload_event_to_bronze(
                event_id=batch_id,
                event_type=event_type,
                events=events,
                partition_date=datetime.now(timezone.utc)
            )

            print(f"[data_lake_service] Flushed {len(events)} {event_type} events to {path}")
            # Clear buffer ONLY after successful write
            # If write fails, events stay in buffer for retry
            self._event_buffer[event_type] = []
            self._last_flush_time = datetime.now(timezone.utc)
        except Exception as e:
            print(f"[data_lake_service] Error flushing {event_type} events: {e}")
            # Do NOT clear buffer on error - keep events for retry
        raise # Re-raise to trigger Kafka retry mechanism