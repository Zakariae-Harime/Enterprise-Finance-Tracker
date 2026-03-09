"""
ApprovalService — orchestrates expense submission and approval workflow.

Responsibilities:
  1. Load current expense state from the expenses projection table
  2. Delegate authorization to ApprovalRuleEngine
  3. Enforce valid state transitions (state machine)
  4. Append domain events to EventStore

This class has NO knowledge of HTTP — no Request, no Response, no HTTPException.
Routes translate ApprovalService exceptions into HTTP status codes.

State machine:
  PENDING --[approve()]--> APPROVED
  PENDING --[reject()]---> REJECTED
  APPROVED / REJECTED: terminal — no further transitions
"""
from decimal import Decimal
from uuid import UUID, uuid4
from typing import Optional

from src.domain.events_store import EventStore
from src.domain.expense_events import ExpenseSubmitted, ExpenseApproved, ExpenseRejected
from src.domain import EventMetadata
from src.services.approval_rules import ApprovalRuleEngine


class ExpenseNotFoundError(Exception):
    pass


class InvalidStateTransitionError(Exception):
    pass


class ApprovalService:

    def __init__(self, pool, event_store: EventStore):
        self._pool = pool
        self._event_store = event_store
        self._rules = ApprovalRuleEngine()

    # ── SUBMIT ────────────────────────────────────────────────────────────────

    async def submit(
        self,
        submitted_by: UUID,
        organization_id: UUID,
        amount: Decimal,
        currency: str,
        description: str,
        merchant_name: str,
        expense_date: str,
        category: Optional[str] = None,
    ) -> UUID:
        """Submit a new expense. Returns the new expense_id."""
        expense_id = uuid4()
        event = ExpenseSubmitted(
            aggregate_id=expense_id,
            metadata=EventMetadata(user_id=submitted_by),
            submitted_by=submitted_by,
            amount=amount,
            currency=currency,
            description=description,
            merchant_name=merchant_name,
            expense_date=expense_date,
            category=category,
        )
        await self._event_store.append_events(
            aggregate_id=expense_id,
            aggregate_type="Expense",
            new_events=[event],
            expected_version=0,
            tenant_id=organization_id,
        )
        return expense_id

    # ── APPROVE ───────────────────────────────────────────────────────────────

    async def approve(
        self,
        expense_id: UUID,
        approver_id: UUID,
        organization_id: UUID,
    ) -> None:
        """Approve a PENDING expense. Raises if not found, wrong state, or insufficient authority."""
        expense, approver = await self._load_expense_and_approver(
            expense_id, approver_id, organization_id
        )
        self._assert_state(expense, "pending")
        self._rules.check_can_approve(
            approver_role=approver["role"],
            can_approve_up_to=approver["can_approve_up_to"],
            expense_amount=expense["amount"],
        )
        event = ExpenseApproved(
            aggregate_id=expense_id,
            metadata=EventMetadata(user_id=approver_id),
            approved_by=approver_id,
        )
        await self._event_store.append_events(
            aggregate_id=expense_id,
            aggregate_type="Expense",
            new_events=[event],
            expected_version=1,
            tenant_id=organization_id,
        )

    # ── REJECT ────────────────────────────────────────────────────────────────

    async def reject(
        self,
        expense_id: UUID,
        rejector_id: UUID,
        organization_id: UUID,
        reason: str,
    ) -> None:
        """Reject a PENDING expense with a mandatory reason."""
        expense, approver = await self._load_expense_and_approver(
            expense_id, rejector_id, organization_id
        )
        self._assert_state(expense, "pending")
        self._rules.check_can_approve(
            approver_role=approver["role"],
            can_approve_up_to=approver["can_approve_up_to"],
            expense_amount=expense["amount"],
        )
        event = ExpenseRejected(
            aggregate_id=expense_id,
            metadata=EventMetadata(user_id=rejector_id),
            rejected_by=rejector_id,
            reason=reason,
        )
        await self._event_store.append_events(
            aggregate_id=expense_id,
            aggregate_type="Expense",
            new_events=[event],
            expected_version=1,
            tenant_id=organization_id,
        )

    # ── HELPERS ───────────────────────────────────────────────────────────────

    async def _load_expense_and_approver(self, expense_id, approver_id, organization_id):
        async with self._pool.acquire() as conn:
            expense = await conn.fetchrow(
                """
                SELECT id, status, amount, organization_id
                FROM expenses
                WHERE id = $1 AND organization_id = $2
                """,
                expense_id,
                organization_id,
            )
            if not expense:
                raise ExpenseNotFoundError(f"Expense {expense_id} not found.")

            approver = await conn.fetchrow(
                """
                SELECT role, can_approve_up_to
                FROM organization_members
                WHERE user_id = $1 AND organization_id = $2
                """,
                approver_id,
                organization_id,
            )

        return expense, approver

    def _assert_state(self, expense, expected_status: str) -> None:
        if expense["status"] != expected_status:
            raise InvalidStateTransitionError(
                f"Expense is '{expense['status']}' — cannot transition from this state. "
                f"Expected '{expected_status}'."
            )
