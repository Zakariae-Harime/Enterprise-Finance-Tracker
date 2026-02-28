"""
Integration tests for AccountProjectionConsumer.
REQUIRES: docker-compose up -d db

Calls consumer handler methods directly (no Kafka needed).
Tests that real DB writes happen correctly when events arrive.
"""
import pytest
from uuid import uuid4
from decimal import Decimal
from datetime import datetime, timezone

from src.consumers.account_projection_consumer import AccountProjectionConsumer

pytestmark = pytest.mark.asyncio

TENANT_ID = "00000000-0000-0000-0000-000000000001"


class TestAccountProjectionInsert:
    """
    Tests handle_account_created() — the INSERT path.
    Each test uses a unique account_id UUID to avoid conflicts.
    """

    async def test_handle_account_created_inserts_row(self, db_pool):
        """
        When AccountCreated event arrives, a row must appear
        in account_projections with the correct initial balance.
        """
        # Arrange
        consumer = AccountProjectionConsumer(db_pool)
        account_id = str(uuid4())   # str because event_data comes from JSON

        event_data = {
            "aggregate_id": account_id,
            "account_name": "DNB Business",
            "currency": "NOK",
            "account_type": "checking",
            "initial_balance": "25000.00",
        }

        # Act — same method the Kafka consumer calls after deserializing the message
        await consumer.handle_account_created(event_data)

        # Assert
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM account_projections WHERE account_id = $1",
                uuid4().__class__(account_id),  # convert str back to UUID for query
            )

        assert row is not None
        assert row["current_balance"] == Decimal("25000.00")
        assert row["currency"] == "NOK"
        assert row["bank_name"] == "DNB Business"

    async def test_handle_account_created_idempotent(self, db_pool):
        """
        ON CONFLICT DO NOTHING: processing the same event twice
        must not duplicate the row or raise an error.
        This is the exactly-once guarantee — critical for Kafka at-least-once delivery.
        """
        consumer = AccountProjectionConsumer(db_pool)
        account_id = str(uuid4())

        event_data = {
            "aggregate_id": account_id,
            "account_name": "Vipps Savings",
            "currency": "NOK",
            "account_type": "savings",
            "initial_balance": "5000.00",
        }

        # Act — call twice with identical data
        await consumer.handle_account_created(event_data)
        await consumer.handle_account_created(event_data)   # should not raise

        # Assert — still only one row
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM account_projections WHERE account_id = $1",
                uuid4().__class__(account_id),
            )

        assert count == 1


class TestTransactionBalanceUpdate:
    """
    Tests handle_transaction_created() — the UPDATE path.
    Seeds an account row first, then applies transactions.
    """

    async def _seed_account(self, conn, account_id, balance="10000.00"):
        """Helper: insert a bare account_projections row to test against."""
        await conn.execute(
            """
            INSERT INTO account_projections
                (account_id, user_id, bank_name, account_type, currency, current_balance, last_event_version)
            VALUES ($1, $2, 'Test Bank', 'checking', 'NOK', $3, 1)
            """,
            account_id,
            uuid4().__class__(TENANT_ID),
            Decimal(balance),
        )

    async def test_debit_decreases_balance(self, db_pool):
        """
        DEBIT = money out. Balance must decrease by the transaction amount.
        """
        consumer = AccountProjectionConsumer(db_pool)
        account_id = uuid4()

        async with db_pool.acquire() as conn:
            await self._seed_account(conn, account_id, "10000.00")

        event_data = {
            "account_id": str(account_id),
            "amount": "2500.00",
            "transaction_type": "debit",
            "transaction_date": datetime.now(timezone.utc).isoformat(),
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            balance = await conn.fetchval(
                "SELECT current_balance FROM account_projections WHERE account_id = $1",
                account_id,
            )

        assert balance == Decimal("7500.00")

    async def test_credit_increases_balance(self, db_pool):
        """
        CREDIT = money in (salary, Vipps refund). Balance must increase.
        """
        consumer = AccountProjectionConsumer(db_pool)
        account_id = uuid4()

        async with db_pool.acquire() as conn:
            await self._seed_account(conn, account_id, "1000.00")

        event_data = {
            "account_id": str(account_id),
            "amount": "50000.00",
            "transaction_type": "credit",
            "transaction_date": datetime.now(timezone.utc).isoformat(),
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            balance = await conn.fetchval(
                "SELECT current_balance FROM account_projections WHERE account_id = $1",
                account_id,
            )

        assert balance == Decimal("51000.00")
