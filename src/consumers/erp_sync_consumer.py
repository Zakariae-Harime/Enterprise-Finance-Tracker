"""
ERP Sync Consumer

Listens for ExpenseApproved events on finance.expense.events.
For each approved expense, pushes it to every active ERP integration
belonging to that organisation via the appropriate adapter.

Flow:
  ExpenseApproved (Kafka)
    → load expense row from DB
    → load all active integrations for org
    → for each integration: decrypt creds → adapter.push_expense()
    → log result (success / failure)

Uses IdempotentConsumer for exactly-once processing.
Topic: finance.expense.events
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.infrastructure.kafka_consumer import IdempotentConsumer
from src.integrations.models import ERPExpense
from src.erp import get_adapter
from src.services.credentials import decrypt_credentials

logger = logging.getLogger(__name__)


class ERPSyncConsumer(IdempotentConsumer):

    def __init__(self, db_pool: asyncpg.Pool):
        super().__init__(db_pool, consumer_name="erp_sync_service")

    async def process_event(self, event_type: str, event_data: dict) -> None:
        if event_type == "ExpenseApproved":
            await self._handle_expense_approved(event_data)
        else:
            logger.debug("[erp_sync_service] Ignored event: %s", event_type)

    async def _handle_expense_approved(self, event_data: dict) -> None:
        expense_id = UUID(event_data["aggregate_id"])
        org_id     = UUID(event_data["org_id"])

        expense = await self._load_expense(expense_id)
        if not expense:
            logger.error("[erp_sync_service] Expense %s not found in DB", expense_id)
            return

        integrations = await self._load_integrations(org_id)
        if not integrations:
            logger.info("[erp_sync_service] No active ERP integrations for org %s", org_id)
            return

        erp_expense = ERPExpense(
            expense_id=expense_id,
            amount=Decimal(str(expense["amount"])),
            currency=expense.get("currency", "NOK"),
            description=expense.get("description", ""),
            merchant_name=expense.get("merchant_name", ""),
            expense_date=expense.get("expense_date", ""),
            category=expense.get("category"),
        )

        for integration in integrations:
            await self._push_to_erp(erp_expense, integration)

    async def _load_expense(self, expense_id: UUID) -> dict | None:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, amount, currency, description, merchant_name, expense_date, category "
                "FROM expenses WHERE id = $1",
                expense_id,
            )
            return dict(row) if row else None

    async def _load_integrations(self, org_id: UUID) -> list[dict]:
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, provider, encrypted_credentials "
                "FROM integrations WHERE organization_id = $1 AND status = 'active'",
                org_id,
            )
            return [dict(r) for r in rows]

    async def _push_to_erp(self, expense: ERPExpense, integration: dict) -> None:
        provider       = integration["provider"]
        integration_id = integration["id"]

        try:
            credentials   = decrypt_credentials(integration["encrypted_credentials"])
            adapter_class = get_adapter(provider)
            adapter       = adapter_class(**credentials)
            result        = await adapter.push_expense(expense)

            if result.success:
                logger.info(
                    "[erp_sync_service] Expense %s pushed to %s → external_id=%s",
                    expense.expense_id, provider, result.external_id,
                )
            else:
                logger.error(
                    "[erp_sync_service] Failed to push expense %s to %s: %s",
                    expense.expense_id, provider, result.error,
                )

        except Exception as exc:
            logger.error(
                "[erp_sync_service] Exception pushing expense %s to integration %s (%s): %s",
                expense.expense_id, integration_id, provider, exc,
            )


async def start_erp_sync_consumer(
    db_pool: asyncpg.Pool,
    kafka_bootstrap_servers: str = "localhost:9092",
    dlq_topic: str = "domain_events_dlq",
    max_retries: int = 3,
) -> None:
    consumer = ERPSyncConsumer(db_pool=db_pool)

    kafka_consumer = AIOKafkaConsumer(
        "finance.expense.events",
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="erp_sync_consumer_group",
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
    logger.info("[erp_sync_service] Consumer started. Topic: finance.expense.events")

    retry_counts: dict[str, int] = {}

    try:
        async for message in kafka_consumer:
            event_id   = message.value.get("event_id")
            event_type = message.value.get("event_type")
            event_data = message.value
            retry_key  = str(event_id)

            try:
                processed = await consumer.handle_event(
                    event_id=UUID(event_id),
                    event_type=event_type,
                    event_data=event_data,
                )
                if processed:
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)

            except Exception as exc:
                retries = retry_counts.get(retry_key, 0) + 1
                retry_counts[retry_key] = retries

                if retries >= max_retries:
                    await dlq_producer.send(dlq_topic, value={
                        "original_event": event_data,
                        "error": str(exc),
                        "retry_count": retries,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "original_topic": message.topic,
                        "consumer": "erp_sync_service",
                    })
                    logger.error(
                        "[erp_sync_service] Event %s sent to DLQ after %d failures",
                        event_id, max_retries,
                    )
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)
                else:
                    logger.warning(
                        "[erp_sync_service] Event %s failed (attempt %d/%d): %s",
                        event_id, retries, max_retries, exc,
                    )

    except asyncio.CancelledError:
        logger.info("[erp_sync_service] Shutdown signal received...")

    finally:
        await kafka_consumer.stop()
        await dlq_producer.stop()
        logger.info("[erp_sync_service] Consumer stopped gracefully.")
