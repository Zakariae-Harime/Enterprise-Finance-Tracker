"""
Approval Projection Consumer

Keeps the `expenses` projection table in sync with expense domain events.

Without this consumer, ApprovalService.approve() fails in production:
  submit() writes an ExpenseSubmitted event (event store only)
  approve() reads from the expenses TABLE — which is empty until this consumer runs

Handles:
  ExpenseSubmitted  → INSERT into expenses (status='pending')
  ExpenseApproved   → UPDATE expenses SET status='approved', approved_by=..., approved_at=NOW()
  ExpenseRejected   → UPDATE expenses SET status='rejected', rejection_reason=...

Uses IdempotentConsumer for exactly-once processing.
Topic: finance.expense.events
"""
import asyncio
import asyncpg
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from src.infrastructure.kafka_consumer import IdempotentConsumer

logger = logging.getLogger(__name__)


class ApprovalConsumer(IdempotentConsumer):
    """
    Maintains the expenses projection table by listening to expense events.

    Inherits from IdempotentConsumer which provides:
      - Exactly-once processing semantics via (event_id, consumer_name) dedup
      - Automatic offset management
      - Error handling and retries
    """

    def __init__(self, db_pool: asyncpg.Pool):
        super().__init__(db_pool, consumer_name="approval_projection_service")

    async def process_event(self, event_type: str, event_data: dict) -> None:
        """Dispatch to the appropriate handler. Unknown events are silently skipped."""
        if event_type == "ExpenseSubmitted":
            await self._handle_submitted(event_data)
        elif event_type == "ExpenseApproved":
            await self._handle_approved(event_data)
        elif event_type == "ExpenseRejected":
            await self._handle_rejected(event_data)
        else:
            logger.info("[approval_projection_service] Unhandled event: %s", event_type)

    async def _handle_submitted(self, event_data: dict) -> None:
        """
        INSERT a new row into expenses when an expense is submitted.

        ON CONFLICT DO NOTHING makes this idempotent — redelivered events are ignored.
        """
        expense_id = event_data.get("aggregate_id")
        organization_id = event_data.get("organization_id")

        if not expense_id:
            logger.error("[approval_projection_service] Missing aggregate_id in ExpenseSubmitted")
            return
        if not organization_id:
            logger.error("[approval_projection_service] Missing organization_id in ExpenseSubmitted")
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO expenses (
                    id, organization_id, submitted_by,
                    amount, currency, description, merchant_name,
                    expense_date, category, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending')
                ON CONFLICT (id) DO NOTHING
                """,
                UUID(expense_id),
                UUID(organization_id),
                UUID(event_data["submitted_by"]),
                Decimal(str(event_data["amount"])),
                event_data["currency"],
                event_data["description"],
                event_data["merchant_name"],
                event_data["expense_date"],
                event_data.get("category"),
            )
        logger.info(
            "[approval_projection_service] Expense %s inserted (pending)", expense_id
        )

    async def _handle_approved(self, event_data: dict) -> None:
        """UPDATE expenses status to 'approved' when an approval event arrives."""
        expense_id = event_data.get("aggregate_id")
        approved_by = event_data.get("approved_by")

        if not expense_id or not approved_by:
            logger.error("[approval_projection_service] Missing fields in ExpenseApproved: %s", event_data)
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE expenses
                SET status = 'approved',
                    approved_by = $2,
                    approved_at = NOW()
                WHERE id = $1
                """,
                UUID(expense_id),
                UUID(approved_by),
            )
        logger.info(
            "[approval_projection_service] Expense %s approved by %s", expense_id, approved_by
        )

    async def _handle_rejected(self, event_data: dict) -> None:
        """UPDATE expenses status to 'rejected' with the rejection reason."""
        expense_id = event_data.get("aggregate_id")
        reason = event_data.get("reason")

        if not expense_id:
            logger.error("[approval_projection_service] Missing aggregate_id in ExpenseRejected")
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE expenses
                SET status = 'rejected',
                    rejection_reason = $2
                WHERE id = $1
                """,
                UUID(expense_id),
                reason,
            )
        logger.info(
            "[approval_projection_service] Expense %s rejected: %s", expense_id, reason
        )


async def start_approval_consumer(
    db_pool: asyncpg.Pool,
    kafka_bootstrap_servers: str = "localhost:9092",
    dlq_topic: str = "domain_events_dlq",
    max_retries: int = 3,
) -> None:
    """
    Connect to Kafka and process expense events forever.

    Subscribes to: finance.expense.events
      ExpenseSubmitted  → INSERT projection row
      ExpenseApproved   → UPDATE to approved
      ExpenseRejected   → UPDATE to rejected
    """
    consumer = ApprovalConsumer(db_pool=db_pool)

    kafka_consumer = AIOKafkaConsumer(
        "finance.expense.events",
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="approval_projection_consumer_group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    dlq_producer = AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    await kafka_consumer.start()
    await dlq_producer.start()
    logger.info(
        "[approval_projection_service] Consumer started. Topic: finance.expense.events"
    )

    retry_counts: dict[str, int] = {}

    try:
        async for message in kafka_consumer:
            event_id = message.value.get("event_id")
            event_type = message.value.get("event_type")
            event_data = message.value

            retry_key = str(event_id)
            current_retries = retry_counts.get(retry_key, 0)

            try:
                processed = await consumer.handle_event(
                    event_id=UUID(event_id),
                    event_type=event_type,
                    event_data=event_data,
                )
                if processed:
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)

            except Exception as e:
                current_retries += 1
                retry_counts[retry_key] = current_retries

                if current_retries >= max_retries:
                    dlq_message = {
                        "original_event": event_data,
                        "error": str(e),
                        "retry_count": current_retries,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "original_topic": message.topic,
                        "consumer": "approval_projection_service",
                    }
                    await dlq_producer.send(dlq_topic, value=dlq_message)
                    logger.error(
                        "[approval_projection_service] Event %s sent to DLQ after %d failures",
                        event_id, max_retries,
                    )
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)
                else:
                    logger.warning(
                        "[approval_projection_service] Event %s failed (attempt %d/%d): %s",
                        event_id, current_retries, max_retries, e,
                    )

    except asyncio.CancelledError:
        logger.info("[approval_projection_service] Shutdown signal received...")

    finally:
        await kafka_consumer.stop()
        await dlq_producer.stop()
        logger.info("[approval_projection_service] Consumer stopped gracefully.")
