"""
  Account Projection Consumer

  Maintains the account_projections read model by listening to Kafka events.
  This is the CQRS bridge — writes go to the event store, this consumer
  asynchronously builds an optimized read table for fast balance queries.

  Handles:
    - AccountCreated → INSERT new projection row with initial_balance
    - TransactionCreated → UPDATE current_balance (CREDIT adds, DEBIT subtracts)

  Uses IdempotentConsumer for exactly-once processing.
  Topics: finance.account.events, finance.transaction.events
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
from src.infrastructure.cache import CacheClient
logger=logging.getLogger(__name__) #logger instead of print() for better logging practices  filter by level (DEBUG/INFO/WARNING/ERROR) and route output to files/monitoring without code changes.
class AccountProjectionConsumer(IdempotentConsumer):
    """
    Consumer that listens to account and transaction events to maintain the account_projections table.

    Inherits from IdempotentConsumer which provides:
      - Exactly-once processing semantics
      - Automatic offset management
      - Error handling and retries
      Kafka consumer that maintains the account_projections table.

      Why a separate read model instead of replaying events on every GET?
        - Event replay is O(n) where n = number of events per aggregate
        - Projection lookup is O(1) — single row by primary key
        - As transactions grow (thousands per account), replay gets slow
        - This is what DNB does: your balance is a projection, not a live replay
      """
    def __init__(self, db_pool: asyncpg.Pool, cache: CacheClient | None = None):
          # "account_projection_service" is unique in the processed_events table.
          # Same event can be processed by BOTH this and email_consumer —
          # tracked independently via (event_id, consumer_name) composite PK.
        super().__init__(db_pool, consumer_name="account_projection_service")
        self.cache = cache
    
    # process_event is the abstract method IdempotentConsumer calls after dedup check
    async def process_event(self, event_type: str, event_data: dict) -> None:
        """
        Dispatch incoming events to the appropriate handler.

        Called by IdempotentConsumer.handle_event() ONLY after the dedup check.
        If this method raises, the event is NOT marked as processed —
        Kafka will redeliver it. This is the "retry on failure" guarantee.
        """
        if event_type == "AccountCreated":
            await self.handle_account_created(event_data)
        elif event_type == "TransactionCreated":
            await self.handle_transaction_created(event_data)
        else:
            logger.info("[account_projection_service] Unhandled event: %s", event_type)
    #AccountCreatedHandler that inserts a new row into account_projections with initial balance
    async def handle_account_created(self, event_data: dict) -> None:
        """
          INSERT a new row into account_projections when an account is created.

          ON CONFLICT DO NOTHING makes this idempotent — if the same event
          is somehow processed twice, the second INSERT is silently ignored.
        """
        # AccountCreated stores the account ID as aggregate_id (inherited from DomainEvent)
        account_id = event_data.get("aggregate_id")
        if not account_id:
            logger.error("[account_projection_service] Missing aggregate_id in AccountCreated event")
            return
        # EventEncoder serializes Decimals as strings — convert back for DB insert
        initial_balance = Decimal(event_data.get("initial_balance", "0.00"))
        currency = event_data.get("currency", "NOK")
        account_type = event_data.get("account_type", "checking")
        account_name = event_data.get("account_name")
        # Hardcoded tenant — same pattern as API routes
        user_id = UUID("00000000-0000-0000-0000-000000000001")
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO account_projections
                    (account_id, user_id, bank_name, account_type,
                     currency, current_balance, last_event_version)
                VALUES ($1, $2, $3, $4, $5, $6, 1)
                ON CONFLICT (account_id) DO NOTHING
                """,
                UUID(account_id), user_id, account_name, account_type, currency, initial_balance,
            )
        logger.info(
            "[account_projection_service] Account %s projected with balance %s %s",
            account_id, initial_balance, currency,
        )

    async def handle_transaction_created(self, event_data: dict) -> None:
        """
        UPDATE current_balance when a transaction occurs.

        Atomic SQL: current_balance = current_balance + $1
        PostgreSQL locks the row during UPDATE, applies delta, releases.
        Two concurrent transactions on the same account both apply correctly.
        """
        account_id = event_data.get("account_id")
        if not account_id:
            # Historical events before we added account_id — skip gracefully
            logger.warning(
                "[account_projection_service] TransactionCreated missing account_id, skipping"
            )
            return

        amount = Decimal(event_data.get("amount", "0"))
        transaction_type = event_data.get("transaction_type", "debit")

        balance_delta = self._calculate_balance_delta(amount, transaction_type)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE account_projections
                SET current_balance = current_balance + $1,
                    last_synced_at = NOW()
                WHERE account_id = $2
                """,
                balance_delta,
                UUID(account_id),
            )
            # updated_at is auto-set by DB trigger (init-db.sql line 228)

        # Invalidate cached account so the next GET reflects the new balance.
        # We don't know the org_id here, so use a wildcard pattern.
        if self.cache:
            await self.cache.delete_pattern(f"account:*:{account_id}")

        logger.info(
            "[account_projection_service] Account %s balance changed by %s (%s)",
            account_id, balance_delta, transaction_type,
        )
    @staticmethod
    def _calculate_balance_delta(amount: Decimal, transaction_type: str) -> Decimal:
        """
        Pure function: determines how a transaction affects balance.

        Args:
            amount: Always positive (Pydantic enforces gt=0)
            transaction_type: "credit", "debit", or "transfer"

        Returns:
            Positive for credits (money IN), negative for debits (money OUT).

        Transfer strategy: debit source account only.
        Destination account credit requires a future paired TransferCredited event.
        """
        if transaction_type == "credit":
            return amount
        elif transaction_type == "debit":
            return -amount
        elif transaction_type == "transfer":
            # Strategy A: treat as debit on source account.
            # Full dual-account support (TransferDebited/TransferCredited paired events)
            # is not yet implemented — log so monitoring can surface it.
            logger.warning(
                "[account_projection_service] Transfer processed as source debit only. "
                "Destination account balance not updated. "
                "Implement paired TransferCredited event for full support."
            )
            return -amount
        else:
            return Decimal("0")
async def start_account_projection_consumer(
    db_pool: asyncpg.Pool,
    kafka_bootstrap_servers: str = "localhost:9092",
    dlq_topic: str = "domain_events_dlq",
    max_retries: int = 3,
    cache: CacheClient | None = None,
) -> None:
    """
    Connect to Kafka and process messages forever.

    Subscribes to TWO topics:
      - finance.account.events → AccountCreated (INSERT rows)
      - finance.transaction.events → TransactionCreated (UPDATE balances)

    Same pattern as start_data_lake_consumer() in data_lake_consumer.py.
    """
    consumer = AccountProjectionConsumer(db_pool=db_pool, cache=cache)

    # AIOKafkaConsumer accepts multiple topic names as positional args
    kafka_consumer = AIOKafkaConsumer(
        "finance.account.events",
        "finance.transaction.events",
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="account_projection_consumer_group",
        auto_offset_reset="earliest",      # Don't miss any events on first start
        enable_auto_commit=False,           # Manual commit = no message loss
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    dlq_producer = AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    await kafka_consumer.start()
    await dlq_producer.start()
    logger.info(
        "[account_projection_service] Consumer started. "
        "Topics: finance.account.events, finance.transaction.events"
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
                        "consumer": "account_projection_service",
                    }
                    await dlq_producer.send(dlq_topic, value=dlq_message)
                    logger.error(
                        "[account_projection_service] Event %s sent to DLQ after %d failures",
                        event_id, max_retries,
                    )
                    await kafka_consumer.commit()
                    retry_counts.pop(retry_key, None)
                else:
                    logger.warning(
                        "[account_projection_service] Event %s failed (attempt %d/%d): %s",
                        event_id, current_retries, max_retries, e,
                    )

    except asyncio.CancelledError:
        logger.info("[account_projection_service] Shutdown signal received...")

    finally:
        await kafka_consumer.stop()
        await dlq_producer.stop()
        logger.info("[account_projection_service] Consumer stopped gracefully.")
#This is a near-copy of start_data_lake_consumer() with two differences: (1) subscribes to two topics instead of one, and (2) no batching/flushing since balance updates are instant single-row writes.
#The retry flow: fail → increment counter → don't commit (Kafka redelivers) → fail again → after 3 attempts → send to DLQ → commit (move past it). The DLQ processor you already built can auto-retry transient errors or flag permanent ones for review.