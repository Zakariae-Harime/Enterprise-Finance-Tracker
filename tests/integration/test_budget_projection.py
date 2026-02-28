"""
Integration tests for BudgetProjectionConsumer.
REQUIRES: docker-compose up -d db

Calls consumer handler methods directly (no Kafka needed).
Tests that real DB writes happen correctly when budget events arrive.
"""
import pytest
from uuid import uuid4, UUID
from decimal import Decimal
from datetime import date

from src.consumers.budget_projection_consumer import BudgetProjectionConsumer

pytestmark = pytest.mark.asyncio

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
THIS_MONTH = date.today().replace(day=1).isoformat()   # "2026-02-01"


class TestBudgetProjectionInsert:
    """
    Tests handle_budget_created() — the INSERT path.
    Each test uses a unique budget UUID to avoid UNIQUE(user_id, category, month) conflicts.
    """

    async def _seed_budget(self, conn, budget_id, category, month, budget_amount="3000.00"):
        """Helper: insert a bare budget_status row for update tests."""
        await conn.execute(
            """
            INSERT INTO budget_status
                (id, user_id, category, month, budget_amount, spent_amount, alert_threshold_reached)
            VALUES ($1, $2, $3, $4, $5, 0.00, FALSE)
            """,
            budget_id,
            TENANT_ID,
            category,
            date.fromisoformat(month),
            Decimal(budget_amount),
        )

    async def test_handle_budget_created_inserts_row(self, db_pool):
        """
        When BudgetCreated event arrives, a row must appear in budget_status
        with the correct budget_amount and category.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = str(uuid4())
        # Use unique category to avoid UNIQUE(user_id, category, month) conflict
        category = f"food_{uuid4().hex[:6]}"

        event_data = {
            "aggregate_id": budget_id,
            "category": category,
            "amount": "3000.00",
            "currency": "NOK",
            "start_date": THIS_MONTH,
            "alert_threshold": 0.8,
        }

        await consumer.handle_budget_created(event_data)

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM budget_status WHERE id = $1",
                UUID(budget_id),
            )

        assert row is not None
        assert row["budget_amount"] == Decimal("3000.00")
        assert row["spent_amount"] == Decimal("0.00")
        assert row["alert_threshold_reached"] is False

    async def test_handle_budget_created_idempotent(self, db_pool):
        """
        ON CONFLICT DO NOTHING: same BudgetCreated event processed twice
        must not raise or duplicate the row.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = str(uuid4())
        category = f"transport_{uuid4().hex[:6]}"

        event_data = {
            "aggregate_id": budget_id,
            "category": category,
            "amount": "1500.00",
            "currency": "NOK",
            "start_date": THIS_MONTH,
        }

        await consumer.handle_budget_created(event_data)
        await consumer.handle_budget_created(event_data)   # second call — must not raise

        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM budget_status WHERE id = $1",
                UUID(budget_id),
            )

        assert count == 1


class TestBudgetSpendingUpdate:
    """
    Tests handle_transaction_created() — the UPDATE path.
    Seeds a budget row first, then applies transactions.
    """

    async def _seed_budget(self, conn, budget_id, category, month, budget_amount="3000.00"):
        await conn.execute(
            """
            INSERT INTO budget_status
                (id, user_id, category, month, budget_amount, spent_amount, alert_threshold_reached)
            VALUES ($1, $2, $3, $4, $5, 0.00, FALSE)
            """,
            budget_id,
            TENANT_ID,
            category,
            date.fromisoformat(month),
            Decimal(budget_amount),
        )

    async def test_debit_increases_spent_amount(self, db_pool):
        """
        DEBIT on a categorized transaction → spent_amount increases.
        The budget is tracking outflows, so debits count against the limit.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = uuid4()
        category = f"groceries_{uuid4().hex[:6]}"

        async with db_pool.acquire() as conn:
            await self._seed_budget(conn, budget_id, category, THIS_MONTH, "3000.00")

        event_data = {
            "account_id": str(uuid4()),
            "amount": "800.00",
            "transaction_type": "debit",
            "category": category,
            "transaction_date": THIS_MONTH + "T10:00:00+00:00",
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            spent = await conn.fetchval(
                "SELECT spent_amount FROM budget_status WHERE id = $1",
                budget_id,
            )

        assert spent == Decimal("800.00")

    async def test_credit_does_not_change_spent_amount(self, db_pool):
        """
        CREDIT = money IN (salary, refund). Must NOT count against budget.
        Spending limits track outflows only.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = uuid4()
        category = f"income_{uuid4().hex[:6]}"

        async with db_pool.acquire() as conn:
            await self._seed_budget(conn, budget_id, category, THIS_MONTH, "3000.00")

        event_data = {
            "account_id": str(uuid4()),
            "amount": "50000.00",
            "transaction_type": "credit",    # salary — should be ignored
            "category": category,
            "transaction_date": THIS_MONTH + "T08:00:00+00:00",
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            spent = await conn.fetchval(
                "SELECT spent_amount FROM budget_status WHERE id = $1",
                budget_id,
            )

        assert spent == Decimal("0.00")   # unchanged

    async def test_uncategorized_transaction_skipped(self, db_pool):
        """
        Transaction without a category cannot be matched to any budget.
        spent_amount must remain unchanged for all budgets.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = uuid4()
        category = f"dining_{uuid4().hex[:6]}"

        async with db_pool.acquire() as conn:
            await self._seed_budget(conn, budget_id, category, THIS_MONTH, "2000.00")

        event_data = {
            "account_id": str(uuid4()),
            "amount": "400.00",
            "transaction_type": "debit",
            "category": None,              # no category → skip
            "transaction_date": THIS_MONTH + "T12:00:00+00:00",
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            spent = await conn.fetchval(
                "SELECT spent_amount FROM budget_status WHERE id = $1",
                budget_id,
            )

        assert spent == Decimal("0.00")


class TestAlertThresholdBehavior:
    """
    Tests that the alert_threshold_reached flag is set correctly in the DB
    when spending crosses 80% of the budget limit.
    """

    async def _seed_budget(self, conn, budget_id, category, month, budget_amount, spent_amount="0.00", already_alerted=False):
        await conn.execute(
            """
            INSERT INTO budget_status
                (id, user_id, category, month, budget_amount, spent_amount, alert_threshold_reached)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            budget_id,
            TENANT_ID,
            category,
            date.fromisoformat(month),
            Decimal(budget_amount),
            Decimal(spent_amount),
            already_alerted,
        )

    async def test_below_threshold_flag_stays_false(self, db_pool):
        """
        Spending at 66% (2000/3000) must NOT set alert_threshold_reached.
        """
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = uuid4()
        category = f"rent_{uuid4().hex[:6]}"

        async with db_pool.acquire() as conn:
            await self._seed_budget(conn, budget_id, category, THIS_MONTH, "3000.00")

        event_data = {
            "amount": "2000.00",
            "transaction_type": "debit",
            "category": category,
            "transaction_date": THIS_MONTH + "T09:00:00+00:00",
        }

        await consumer.handle_transaction_created(event_data)

        async with db_pool.acquire() as conn:
            flag = await conn.fetchval(
                "SELECT alert_threshold_reached FROM budget_status WHERE id = $1",
                budget_id,
            )

        assert flag is False

    async def test_threshold_alert_sets_flag(self, db_pool):
        """
        When a transaction pushes spending to exactly 80% of the budget limit,
        the consumer must flip alert_threshold_reached = TRUE in the DB.

        Scenario:
          - Budget: 3000 NOK for groceries
          - Already spent: 2200 NOK (73% — below threshold, flag is FALSE)
          - New debit: 200 NOK at REMA
          - New total: 2400 NOK (exactly 80% — threshold crossed → flag = TRUE)
        """
        # Arrange — unique category prevents UNIQUE(user_id, category, month) conflict
        consumer = BudgetProjectionConsumer(db_pool)
        budget_id = uuid4()
        category = f"groceries_{uuid4().hex[:6]}"

        async with db_pool.acquire() as conn:
            await self._seed_budget(
                conn, budget_id, category, THIS_MONTH,
                budget_amount="3000.00",
                spent_amount="2200.00",   # already 73% — just below threshold
                already_alerted=False,
            )

        event_data = {
            "amount": "200.00",                                  # pushes to 2400 = 80%
            "transaction_type": "debit",
            "category": category,
            "transaction_date": THIS_MONTH + "T14:00:00+00:00",
        }

        # Act — consumer updates spent_amount then calls _check_and_emit_alerts()
        await consumer.handle_transaction_created(event_data)

        # Assert — flag must now be TRUE in the DB
        async with db_pool.acquire() as conn:
            flag = await conn.fetchval(
                "SELECT alert_threshold_reached FROM budget_status WHERE id = $1",
                budget_id,
            )

        assert flag is True
