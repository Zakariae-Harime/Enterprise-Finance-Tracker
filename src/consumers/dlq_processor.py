"""
Dead Letter Queue (DLQ) Processor

This service handles messages that failed processing in other consumers.
It provides:
    1. Visibility: Logs all failed messages to database
    2. Alerting: Can trigger alerts for operations team
    3. Replay: Can republish messages to original topic after fix
    4. Analysis: Categorizes errors (transient vs permanent)

Flow:
    DLQ Topic → DLQ Processor → Database (logging)
                             → Original Topic (replay)
                             → Alert System (notifications)

Why is this important?
    - Failed messages shouldn't disappear silently
    - Ops team needs visibility into what's failing
    - After fixing bugs, we need to replay failed messages
    - Transient errors (network timeout) can be auto-retried
"""
# === Standard Library Imports ===
import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID, uuid4
from enum import Enum

# === Third-Party Imports ===
import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer


class ErrorCategory(Enum):
    """
    Categories of errors for different handling strategies.

    Why categorize?
        - TRANSIENT: Temporary issues, safe to auto-retry
        - PERMANENT: Data issues, needs manual fix
        - UNKNOWN: New error types, investigate

    Real-world examples:
        - TRANSIENT: "Connection timeout", "Service unavailable"
        - PERMANENT: "Invalid JSON", "Missing required field"
        - UNKNOWN: Any error not in our known lists
    """
    TRANSIENT = "transient"   # Network issues, timeouts - auto-retry
    PERMANENT = "permanent"   # Bad data, schema errors - manual fix
    UNKNOWN = "unknown"       # New error types - investigate


# Known error patterns for categorization
# These strings are checked against the error message
TRANSIENT_ERRORS = [
    "timeout",
    "connection refused",
    "service unavailable",
    "too many requests",
    "temporary failure",
    "network unreachable",
    "connection reset",
]

PERMANENT_ERRORS = [
    "invalid json",
    "missing required field",
    "validation error",
    "schema mismatch",
    "null value",
    "type error",
    "key error",
]


def categorize_error(error_message: str) -> ErrorCategory:
    """
    Analyze error message and categorize it.

    Args:
        error_message: The error string from failed processing

    Returns:
        ErrorCategory enum indicating error type

    How it works:
        1. Convert error to lowercase (case-insensitive matching)
        2. Check against known transient patterns
        3. Check against known permanent patterns
        4. Default to UNKNOWN if no match

    Example:
        "Connection timeout after 30s" → TRANSIENT
        "ValidationError: missing field 'amount'" → PERMANENT
        "Some weird error we've never seen" → UNKNOWN
    """
    error_lower = error_message.lower()

    # Check for transient (temporary) errors
    for pattern in TRANSIENT_ERRORS:
        if pattern in error_lower:
            return ErrorCategory.TRANSIENT

    # Check for permanent (data) errors
    for pattern in PERMANENT_ERRORS:
        if pattern in error_lower:
            return ErrorCategory.PERMANENT

    # Unknown error type - needs investigation
    return ErrorCategory.UNKNOWN


class DLQProcessor:
    """
    Processes messages from Dead Letter Queue.

    Responsibilities:
        1. Consume messages from DLQ topic
        2. Log failures to database (for visibility)
        3. Categorize errors (transient vs permanent)
        4. Auto-retry transient errors
        5. Store permanent errors for manual review
        6. Provide replay functionality

    Why a separate class?
        - Single Responsibility: Only handles DLQ logic
        - Testable: Can mock dependencies
        - Reusable: Works with any consumer's DLQ
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        kafka_bootstrap_servers: str,
        dlq_topic: str = "domain_events_dlq",
        auto_retry_transient: bool = True,
        retry_delay_seconds: int = 60
    ):
        """
        Initialize DLQ Processor.

        Args:
            db_pool: Database connection pool
                - Used for logging failures to dlq_messages table
                - Shared with other components (connection pooling)

            kafka_bootstrap_servers: Kafka broker address
                - Example: "localhost:9092" or "kafka:29092"
                - Used for both consuming DLQ and republishing

            dlq_topic: Name of Dead Letter Queue topic
                - Convention: "{original_topic}_dlq"
                - Example: "domain_events_dlq"

            auto_retry_transient: Whether to auto-retry transient errors
                - True: Automatically republish after delay
                - False: Store for manual review

            retry_delay_seconds: Wait time before auto-retry
                - Gives external services time to recover
                - Default 60s is reasonable for most cases
        """
        self._db_pool = db_pool
        self._kafka_servers = kafka_bootstrap_servers
        self._dlq_topic = dlq_topic
        self._auto_retry = auto_retry_transient
        self._retry_delay = retry_delay_seconds

        # Will be initialized in start()
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        """
        Start the DLQ processor.

        Steps:
            1. Create Kafka consumer for DLQ topic
            2. Create Kafka producer for replaying messages
            3. Start both connections
            4. Begin processing loop

        Why async start?
            - Kafka connections are async
            - Allows proper error handling during startup
            - Can be called from async main()
        """
        # Create consumer for reading from DLQ
        self._consumer = AIOKafkaConsumer(
            self._dlq_topic,
            bootstrap_servers=self._kafka_servers,
            group_id="dlq_processor_group",
            auto_offset_reset="earliest",  # Don't miss any failed messages
            enable_auto_commit=False,      # Manual commit after processing
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        # Create producer for replaying messages to original topics
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        await self._consumer.start()
        await self._producer.start()
        print(f"[dlq_processor] Started. Monitoring: {self._dlq_topic}")

    async def stop(self) -> None:
        """
        Stop the DLQ processor gracefully.

        Why graceful shutdown?
            - Ensure in-flight messages are processed
            - Close connections properly
            - Avoid message loss
        """
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        print("[dlq_processor] Stopped gracefully")

    async def process_dlq_messages(self) -> None:
        """
        Main processing loop - reads and handles DLQ messages.

        For each message:
            1. Log to database (visibility)
            2. Categorize error type
            3. Handle based on category:
               - TRANSIENT: Auto-retry after delay
               - PERMANENT: Store for manual review
               - UNKNOWN: Store and alert
            4. Commit offset

        Why infinite loop?
            - DLQ processor runs continuously
            - New failures can arrive anytime
            - Should always be monitoring
        """
        if not self._consumer:
            raise RuntimeError("DLQ Processor not started. Call start() first.")

        async for message in self._consumer:
            # message.value structure (from our DataLakeConsumer):
            # {
            #     "original_event": {...},
            #     "error": "error message",
            #     "retry_count": 3,
            #     "failed_at": "2024-01-15T10:30:00Z",
            #     "original_topic": "domain_events",
            #     "consumer": "data_lake_service"
            # }

            dlq_message = message.value

            # Extract fields from DLQ message
            original_event = dlq_message.get("original_event", {})
            error_message = dlq_message.get("error", "Unknown error")
            retry_count = dlq_message.get("retry_count", 0)
            failed_at = dlq_message.get("failed_at")
            original_topic = dlq_message.get("original_topic", "unknown")
            consumer_name = dlq_message.get("consumer", "unknown")
            event_id = original_event.get("event_id", str(uuid4()))

            # Step 1: Categorize the error
            category = categorize_error(error_message)

            # Step 2: Log to database
            dlq_record_id = await self._log_to_database(
                event_id=event_id,
                original_event=original_event,
                error_message=error_message,
                error_category=category,
                retry_count=retry_count,
                failed_at=failed_at,
                original_topic=original_topic,
                consumer_name=consumer_name
            )

            print(f"[dlq_processor] Logged DLQ message: {event_id} | Category: {category.value}")

            # Step 3: Handle based on category
            if category == ErrorCategory.TRANSIENT and self._auto_retry:
                # Auto-retry transient errors
                await self._schedule_retry(
                    original_event=original_event,
                    original_topic=original_topic,
                    dlq_record_id=dlq_record_id
                )
            elif category == ErrorCategory.PERMANENT:
                # Mark as needs manual review
                await self._mark_for_review(dlq_record_id)
            else:  # UNKNOWN
                # Store and could trigger alert
                await self._mark_for_review(dlq_record_id)
                # TODO: Trigger alert to ops team

            # Step 4: Commit offset
            await self._consumer.commit()

    async def _log_to_database(
        self,
        event_id: str,
        original_event: dict,
        error_message: str,
        error_category: ErrorCategory,
        retry_count: int,
        failed_at: str,
        original_topic: str,
        consumer_name: str
    ) -> int:
        """
        Log DLQ message to database for tracking.

        Args:
            event_id: Original event's ID
            original_event: Full original message
            error_message: What went wrong
            error_category: TRANSIENT/PERMANENT/UNKNOWN
            retry_count: How many times we tried
            failed_at: When it failed
            original_topic: Where it came from
            consumer_name: Which consumer failed

        Returns:
            dlq_record_id: Database ID for this record

        Why log to database?
            - Persistence: Survives restarts
            - Queryable: "Show me all failures today"
            - Dashboard: Ops team can see failures
            - Audit: Track what failed and when
        """
        async with self._db_pool.acquire() as conn:
            record_id = await conn.fetchval(
                """
                INSERT INTO dlq_messages (
                    event_id,
                    original_event,
                    error_message,
                    error_category,
                    retry_count,
                    failed_at,
                    original_topic,
                    consumer_name,
                    status,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                RETURNING id
                """,
                event_id,
                json.dumps(original_event),  # Store as JSON string
                error_message,
                error_category.value,
                retry_count,
                failed_at,
                original_topic,
                consumer_name,
                "pending"  # Initial status
            )
            return record_id

    async def _schedule_retry(
        self,
        original_event: dict,
        original_topic: str,
        dlq_record_id: int
    ) -> None:
        """
        Schedule auto-retry for transient errors.

        Args:
            original_event: The message to replay
            original_topic: Where to republish it
            dlq_record_id: Database record to update

        Flow:
            1. Wait for retry_delay (let service recover)
            2. Republish to original topic
            3. Update database status to "retried"

        Why delay?
            - Transient errors often resolve themselves
            - Network timeouts: Service might be restarting
            - Rate limiting: Wait for quota reset
        """
        print(f"[dlq_processor] Scheduling retry in {self._retry_delay}s for DLQ #{dlq_record_id}")

        # Wait before retry
        await asyncio.sleep(self._retry_delay)

        # Republish to original topic
        await self._producer.send(original_topic, value=original_event)

        # Update database status
        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE dlq_messages
                SET status = 'retried',
                    retried_at = NOW()
                WHERE id = $1
                """,
                dlq_record_id
            )

        print(f"[dlq_processor] Retried DLQ #{dlq_record_id} → {original_topic}")

    async def _mark_for_review(self, dlq_record_id: int) -> None:
        """
        Mark message as needing manual review.

        Args:
            dlq_record_id: Database record to update

        When is this called?
            - PERMANENT errors (bad data)
            - UNKNOWN errors (new error types)

        What happens next?
            - Ops team sees this in dashboard
            - They investigate and fix the issue
            - Then manually trigger replay
        """
        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE dlq_messages
                SET status = 'needs_review'
                WHERE id = $1
                """,
                dlq_record_id
            )
        print(f"[dlq_processor] DLQ #{dlq_record_id} marked for manual review")

    async def replay_message(self, dlq_record_id: int) -> bool:
        """
        Manually replay a specific DLQ message.

        Args:
            dlq_record_id: Which message to replay

        Returns:
            True if replayed successfully, False otherwise

        Use case:
            1. Developer fixes a bug
            2. Ops team calls this method via API
            3. Message is republished to original topic
            4. Consumer (now fixed) processes it successfully

        Example API endpoint:
            POST /api/dlq/{id}/replay
        """
        async with self._db_pool.acquire() as conn:
            # Fetch the DLQ record
            record = await conn.fetchrow(
                """
                SELECT original_event, original_topic, status
                FROM dlq_messages
                WHERE id = $1
                """,
                dlq_record_id
            )

            if not record:
                print(f"[dlq_processor] DLQ #{dlq_record_id} not found")
                return False

            if record["status"] == "replayed":
                print(f"[dlq_processor] DLQ #{dlq_record_id} already replayed")
                return False

            # Parse the original event
            original_event = json.loads(record["original_event"])
            original_topic = record["original_topic"]

            # Republish to original topic
            await self._producer.send(original_topic, value=original_event)

            # Update status
            await conn.execute(
                """
                UPDATE dlq_messages
                SET status = 'replayed',
                    replayed_at = NOW()
                WHERE id = $1
                """,
                dlq_record_id
            )

            print(f"[dlq_processor] Replayed DLQ #{dlq_record_id} → {original_topic}")
            return True

    async def replay_all_pending(self) -> int:
        """
        Replay ALL messages that are pending review.

        Returns:
            Number of messages replayed

        Use case:
            - Bug fix deployed
            - Replay all failed messages at once

        Warning:
            Use with caution! Only after confirming fix.
        """
        async with self._db_pool.acquire() as conn:
            # Get all pending messages
            records = await conn.fetch(
                """
                SELECT id FROM dlq_messages
                WHERE status IN ('pending', 'needs_review')
                ORDER BY created_at
                """
            )

            replayed_count = 0
            for record in records:
                success = await self.replay_message(record["id"])
                if success:
                    replayed_count += 1

            print(f"[dlq_processor] Replayed {replayed_count} messages")
            return replayed_count

    async def get_dlq_stats(self) -> dict:
        """
        Get statistics about DLQ messages.

        Returns:
            Dict with counts by status and category

        Use case:
            - Dashboard display
            - Monitoring/alerting

        Example response:
            {
                "total": 150,
                "by_status": {
                    "pending": 10,
                    "needs_review": 25,
                    "retried": 100,
                    "replayed": 15
                },
                "by_category": {
                    "transient": 100,
                    "permanent": 35,
                    "unknown": 15
                },
                "by_consumer": {
                    "data_lake_service": 120,
                    "email_service": 30
                }
            }
        """
        async with self._db_pool.acquire() as conn:
            # Total count
            total = await conn.fetchval("SELECT COUNT(*) FROM dlq_messages")

            # Count by status
            status_rows = await conn.fetch(
                """
                SELECT status, COUNT(*) as count
                FROM dlq_messages
                GROUP BY status
                """
            )
            by_status = {row["status"]: row["count"] for row in status_rows}

            # Count by error category
            category_rows = await conn.fetch(
                """
                SELECT error_category, COUNT(*) as count
                FROM dlq_messages
                GROUP BY error_category
                """
            )
            by_category = {row["error_category"]: row["count"] for row in category_rows}

            # Count by consumer
            consumer_rows = await conn.fetch(
                """
                SELECT consumer_name, COUNT(*) as count
                FROM dlq_messages
                GROUP BY consumer_name
                """
            )
            by_consumer = {row["consumer_name"]: row["count"] for row in consumer_rows}

            return {
                "total": total,
                "by_status": by_status,
                "by_category": by_category,
                "by_consumer": by_consumer
            }


async def start_dlq_processor(
    db_pool: asyncpg.Pool,
    kafka_bootstrap_servers: str = "localhost:9092",
    dlq_topic: str = "domain_events_dlq"
) -> None:
    """
    Start the DLQ processor service.

    This is the main entry point for running the DLQ processor.

    Args:
        db_pool: Database connection pool
        kafka_bootstrap_servers: Kafka broker address
        dlq_topic: DLQ topic to monitor

    Usage:
        asyncio.run(start_dlq_processor(db_pool))
    """
    processor = DLQProcessor(
        db_pool=db_pool,
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        dlq_topic=dlq_topic,
        auto_retry_transient=True,
        retry_delay_seconds=60
    )

    try:
        await processor.start()
        await processor.process_dlq_messages()
    except asyncio.CancelledError:
        print("[dlq_processor] Shutdown signal received")
    finally:
        await processor.stop()
