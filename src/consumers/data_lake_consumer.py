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
        self._data_lake_client = data_lake_client
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
    async def flush_all(self) -> None:
        """
          Flush ALL event type buffers.

          When to call this?
              - Graceful shutdown (SIGTERM received)
              - Before rebalancing (Kafka reassigning partitions)
              - Periodic cleanup

          Why is this important?
              Without this, events in buffer would be lost on shutdown!
        """
        for event_type in list(self._event_buffer.keys()):
            await self._flush_buffer(event_type)
            print(f"[data_lake_service] Flushed buffer for {event_type} during shutdown/rebalance.")
async def start_data_lake_consumer(
    db_pool: asyncpg.Pool,
    data_lake_client: DataLakeClient,
    kafka_bootstrap_servers: str = "localhost:9092",
    kafka_topic: str = "domain_events",
    dlq_topic: str = "domain_events_dlq", # Separate topic for failed messages (keeps main topic clean)
    max_retries: int = 3 # Don't retry forever (3 is industry standard)
) -> None:
    """
    Start the Data Lake consumer - connects to Kafka and processes messages.

    This is the main entry point that:
        1. Creates the DataLakeConsumer instance
        2. Connects to Kafka cluster
        3. Subscribes to the topic
        4. Processes messages in an infinite loop
        5. Handles graceful shutdown
        6. Sends failed messages to Dead Letter Queue after max retries

    Args:
        db_pool: Database connection pool (passed to consumer)
        data_lake_client: Configured Data Lake client
        kafka_bootstrap_servers: Kafka broker address (e.g., "kafka:29092")
        kafka_topic: Topic to consume from (e.g., "domain-events")
        dlq_topic: Dead Letter Queue topic for failed messages
        max_retries: Max retry attempts before sending to DLQ
    """
    from aiokafka import AIOKafkaProducer  # For DLQ publishing

    # Track retry counts per message
    # Key: event_id, Value: number of attempts
    retry_counts: dict[str, int] = {}
    """  - Kafka doesn't track retries for you
  - We store {event_id: attempt_count} in memory
  - Cleared on success or after sending to DLQ
    """

    # Create our consumer instance
    consumer = DataLakeConsumer(
        db_pool=db_pool,
        data_lake_client=data_lake_client,
        batch_size=100,
        flush_interval_seconds=60
    )

    # Create Kafka consumer
    kafka_consumer = AIOKafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="data_lake_consumer_group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    # Create Kafka producer for Dead Letter Queue
    """separate producer
  - Consumer reads from main topic
  - Producer writes to DLQ topic
  - Same Kafka cluster, different topics
  - Keeps main topic clean of failed messages"""
    dlq_producer = AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    await kafka_consumer.start()
    await dlq_producer.start()
    print(f"[data_lake_service] Consumer started. Main topic: {kafka_topic}, DLQ: {dlq_topic}")

    try:
        async for message in kafka_consumer:
            event_id = message.value.get("event_id")
            event_type = message.value.get("event_type")
            event_data = message.value

            # Get current retry count for this message
            retry_key = str(event_id)
            current_retries = retry_counts.get(retry_key, 0)

            try:
                processed = await consumer.handle_event(
                    event_id=UUID(event_id),
                    event_type=event_type,
                    event_data=event_data
                )
                if processed:
                    await kafka_consumer.commit()
                    # Clear retry count on success
                    retry_counts.pop(retry_key, None)

            except Exception as e:
                current_retries += 1
                retry_counts[retry_key] = current_retries

                if current_retries >= max_retries:
                    # Max retries exceeded - send to Dead Letter Queue
                    dlq_message = {
                        "original_event": event_data,
                        "error": str(e),
                        "retry_count": current_retries,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "original_topic": kafka_topic,
                        "consumer": "data_lake_service"
                    }
                    await dlq_producer.send(dlq_topic, value=dlq_message)
                    print(f"[data_lake_service] Event {event_id} sent to DLQ after {max_retries} failures")

                    # Commit to move past this message
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)
                else:
                    # Retry - don't commit, let Kafka redeliver
                    print(f"[data_lake_service] Event {event_id} failed (attempt {current_retries}/{max_retries}): {e}")

    except asyncio.CancelledError:
        print("[data_lake_service] Shutdown signal received...")

    finally:
        await consumer.flush_all()
        await kafka_consumer.stop()
        await dlq_producer.stop()
        print("[data_lake_service] Consumer stopped gracefully.")