"""
  Budget Projection Consumer

  Maintains the budget_status read model by listening to Kafka events.
  This is the CQRS bridge for budgets — writes go to the event store,
  this consumer asynchronously builds an optimized read table and fires
  alert events when spending limits are approached or exceeded.

  Handles:
    - BudgetCreated     -> INSERT new budget_status row
    - TransactionCreated -> UPDATE spent_amount (debit/transfer only)
                           -> Emit BudgetThresholdExceeded or BudgetExceeded if crossed

  Uses IdempotentConsumer for exactly-once processing.
  Topics: finance.budget.events, finance.transaction.events
"""
import asyncio
import asyncpg
import json
import logging
from datetime import datetime, timezone, date
from decimal import Decimal
from uuid import UUID
from uuid import uuid4
from uuid6 import uuid6

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from src.infrastructure.kafka_consumer import IdempotentConsumer

logger = logging.getLogger(__name__)


class BudgetProjectionConsumer(IdempotentConsumer):
    """
    Consumer that maintains the budget_status table and fires spending alerts.

    Why track budgets via Kafka instead of synchronous DB writes?
      - Transactions and budgets are separate aggregates
      - The consumer decouples them: a slow budget check never blocks a transaction
      - If the consumer is down, events queue in Kafka — no data is lost
      - Same pattern Vipps uses for spending insights in their app
    """
    def __init__(self, db_pool: asyncpg.Pool):
        super().__init__(db_pool, consumer_name="budget_projection_service")

    async def process_event(self, event_type: str, event_data: dict) -> None:
        """Dispatch incoming events to the appropriate handler."""
        if event_type == "BudgetCreated":
            await self.handle_budget_created(event_data)
        elif event_type == "TransactionCreated":
            await self.handle_transaction_created(event_data)
        else:
            logger.info("[budget_projection_service] Unhandled event: %s", event_type)

    async def handle_budget_created(self, event_data: dict) -> None:
        """
        INSERT a new row into budget_status when a budget is created.

        ON CONFLICT DO NOTHING makes this idempotent — reprocessing the same
        BudgetCreated event (Kafka at-least-once) won't duplicate the row.
        """
        budget_id = event_data.get("aggregate_id")
        if not budget_id:
            logger.error("[budget_projection_service] Missing aggregate_id in BudgetCreated")
            return

        budget_amount = Decimal(str(event_data.get("amount", "0.00")))
        category = event_data.get("category") or "uncategorized"
        alert_threshold = event_data.get("alert_threshold", 0.8)

        # Parse start_date to get the month (budget_status.month is a DATE column)
        start_date_str = event_data.get("start_date", "")
        try:
            month = date.fromisoformat(start_date_str[:10])  # "2026-02-01T00:00:00+00:00" -> date(2026,2,1)
        except (ValueError, TypeError):
            month = date.today().replace(day=1)

        user_id = UUID("00000000-0000-0000-0000-000000000001")

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO budget_status
                    (id, user_id, category, month, budget_amount, spent_amount, alert_threshold_reached)
                VALUES ($1, $2, $3, $4, $5, 0.00, FALSE)
                ON CONFLICT (user_id, category, month) DO NOTHING
                """,
                UUID(budget_id), user_id, category, month, budget_amount,
            )

        logger.info(
            "[budget_projection_service] Budget %s projected: %s %s for %s",
            budget_id, budget_amount, event_data.get("currency", "NOK"), category,
        )

    async def handle_transaction_created(self, event_data: dict) -> None:
        """
        UPDATE spent_amount when a relevant transaction occurs.

        Only DEBIT and TRANSFER reduce a budget — CREDIT is money coming IN
        (salary, refunds) and should not count against spending limits.

        After updating, checks if any alert thresholds are crossed.
        """
        transaction_type = event_data.get("transaction_type", "debit")

        # Credits are income — they do not count as spending
        if transaction_type == "credit":
            return

        category = event_data.get("category")
        if not category:
            # Uncategorized transactions can't be matched to a category budget
            logger.debug("[budget_projection_service] Transaction has no category, skipping budget check")
            return

        amount = Decimal(str(event_data.get("amount", "0")))
        tx_date_str = event_data.get("transaction_date", "")
        try:
            tx_date = date.fromisoformat(tx_date_str[:10])
            month = tx_date.replace(day=1)   # normalize to first of month
        except (ValueError, TypeError):
            month = date.today().replace(day=1)

        user_id = UUID("00000000-0000-0000-0000-000000000001")

        async with self.db_pool.acquire() as conn:
            # Atomic update — PostgreSQL row lock prevents race conditions
            # between two concurrent transactions on the same budget
            updated = await conn.fetchrow(
                """
                UPDATE budget_status
                SET spent_amount = spent_amount + $1,
                    updated_at = NOW()
                WHERE user_id = $2
                  AND category = $3
                  AND month = $4
                RETURNING id, spent_amount, budget_amount, alert_threshold_reached
                """,
                amount, user_id, category, month,
            )

        if not updated:
            # No budget exists for this category/month — nothing to track
            logger.debug(
                "[budget_projection_service] No budget found for category=%s month=%s",
                category, month,
            )
            return

        budget_id = updated["id"]
        spent = updated["spent_amount"]
        limit = updated["budget_amount"]
        already_alerted = updated["alert_threshold_reached"]

        logger.info(
            "[budget_projection_service] Budget %s: spent %s / %s (%s%%)",
            budget_id, spent, limit,
            round(float(spent) / float(limit) * 100, 1) if limit else 0,
        )

        await self._check_and_emit_alerts(budget_id, spent, limit, already_alerted)

    @staticmethod
    def _should_alert_threshold(spent: Decimal, limit: Decimal, threshold: float) -> bool:
        """
        Pure function: returns True when spending has crossed the alert threshold.

        Args:
            spent: current accumulated spending (always >= 0)
            limit: the budget ceiling (always > 0)
            threshold: fraction that triggers the warning (e.g. 0.8 = 80%)

        Returns True only if spending >= threshold fraction of limit.
        """
        if limit <= 0:
            return False
        return spent >= limit * Decimal(str(threshold))

    @staticmethod
    def _should_alert_exceeded(spent: Decimal, limit: Decimal) -> bool:
        """
        Pure function: returns True when spending has crossed the budget limit.

        Args:
            spent: current accumulated spending
            limit: the budget ceiling

        Returns True only if spending >= limit (100% of budget used).
        """
        return spent >= limit

    async def _check_and_emit_alerts(
        self,
        budget_id: UUID,
        spent: Decimal,
        limit: Decimal,
        already_alerted: bool,
    ) -> None:
        """

        Called after every transaction update to determine if alerts should fire.

        Args:
            budget_id: the budget aggregate ID
            spent: current spent_amount after this transaction
            limit: the budget_amount ceiling
            already_alerted: whether alert_threshold_reached is already TRUE in DB
                             (prevents re-firing the threshold alert on every transaction)

        Options to implement:
          A: Fire threshold alert only once (check already_alerted flag),
             fire exceeded alert every time spending crosses limit.
          B: Fire both alerts every time the condition is met (simpler, but noisy).
          C: Fire threshold alert once, never fire exceeded (let the UI derive it from percentage_used).

        The already_alerted flag in budget_status is designed for Option A.
        Use self._should_alert_threshold() and self._should_alert_exceeded()
        to check conditions, then UPDATE alert_threshold_reached = TRUE and log the alert.
        """
        if self._should_alert_exceeded(spent, limit):
            logger.warning(
                "[budget_projection_service] Budget %s EXCEEDED: spent %s / %s",
                budget_id, spent, limit,
            )
        elif self._should_alert_threshold(spent, limit, 0.8) and not already_alerted:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE budget_status SET alert_threshold_reached = TRUE WHERE id = $1",
                    budget_id,
                )
            logger.warning(
                "[budget_projection_service] Budget %s threshold reached: spent %s / %s",
                budget_id, spent, limit,
            )


async def start_budget_projection_consumer(
    db_pool: asyncpg.Pool,
    kafka_bootstrap_servers: str = "localhost:9092",
    dlq_topic: str = "domain_events_dlq",
    max_retries: int = 3,
) -> None:
    """
    Connect to Kafka and process budget + transaction events forever.

    Subscribes to TWO topics:
      - finance.budget.events      -> BudgetCreated (INSERT rows)
      - finance.transaction.events -> TransactionCreated (UPDATE spent_amount)

    Same retry + DLQ pattern as account_projection_consumer.py.
    """
    consumer = BudgetProjectionConsumer(db_pool=db_pool)

    kafka_consumer = AIOKafkaConsumer(
        "finance.budget.events",
        "finance.transaction.events",
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="budget_projection_consumer_group",
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
        "[budget_projection_service] Consumer started. "
        "Topics: finance.budget.events, finance.transaction.events"
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
                        "consumer": "budget_projection_service",
                    }
                    await dlq_producer.send(dlq_topic, value=dlq_message)
                    logger.error(
                        "[budget_projection_service] Event %s sent to DLQ after %d failures",
                        event_id, max_retries,
                    )
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)
                else:
                    logger.warning(
                        "[budget_projection_service] Event %s failed (attempt %d/%d): %s",
                        event_id, current_retries, max_retries, e,
                    )

    except asyncio.CancelledError:
        logger.info("[budget_projection_service] Shutdown signal received...")

    finally:
        await kafka_consumer.stop()
        await dlq_producer.stop()
        logger.info("[budget_projection_service] Consumer stopped gracefully.")
