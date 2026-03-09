"""
ExpenseProjectionConsumer

Builds the `expenses` read table from domain events.

Event handlers:
  ExpenseSubmitted  → INSERT into expenses with status='pending'
  ExpenseApproved   → UPDATE expenses SET status='approved'
  ExpenseRejected   → UPDATE expenses SET status='rejected'
"""
import logging
from uuid import UUID
from decimal import Decimal

logger = logging.getLogger(__name__)


class ExpenseProjectionConsumer:

    def __init__(self, pool):
        self._pool = pool

    async def handle(self, event_type: str, event_data: dict, aggregate_id: UUID, tenant_id: UUID) -> None:
        handlers = {
            "ExpenseSubmitted": self._on_submitted,
            "ExpenseApproved":  self._on_approved,
            "ExpenseRejected":  self._on_rejected,
        }
        handler = handlers.get(event_type)
        if handler:
            await handler(event_data, aggregate_id, tenant_id)
        else:
            logger.debug("ExpenseProjectionConsumer: ignoring event type '%s'", event_type)

    async def _on_submitted(self, data: dict, expense_id: UUID, org_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO expenses (
                    id, organization_id, submitted_by,
                    amount, currency, description, merchant_name,
                    expense_date, category, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending')
                ON CONFLICT (id) DO NOTHING
                """,
                expense_id,
                org_id,
                UUID(data["submitted_by"]),
                Decimal(str(data["amount"])),
                data["currency"],
                data["description"],
                data["merchant_name"],
                data["expense_date"],
                data.get("category"),
            )

    async def _on_approved(self, data: dict, expense_id: UUID, _: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE expenses
                SET status = 'approved',
                    approved_by = $2,
                    approved_at = NOW()
                WHERE id = $1
                """,
                expense_id,
                UUID(data["approved_by"]),
            )

    async def _on_rejected(self, data: dict, expense_id: UUID, _: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE expenses
                SET status = 'rejected',
                    rejection_reason = $2
                WHERE id = $1
                """,
                expense_id,
                data["reason"],
            )
