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
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from typing import Optional

from src.domain.events_store import EventStore
from src.domain.expense_events import (
    ExpenseSubmitted,
    ExpenseApprovalRequested,
    ExpenseApproved,
    ExpenseRejected,
)
from src.domain import EventMetadata
from src.services.approval_rules import ApprovalRuleEngine

# System actor UUID used when auto-approving via rules (no human approver)
_SYSTEM_UUID = UUID("00000000-0000-0000-0000-000000000000")


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
        """
        Submit a new expense. Returns the new expense_id.

        After appending ExpenseSubmitted, queries approval_rules:
          - auto_approve=True  → immediately appends ExpenseApproved (system actor)
          - rule found         → appends ExpenseApprovalRequested (role-based routing)
          - no rule            → stays pending, awaits manual approval
        """
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
            organization_id=organization_id,
        )
        await self._event_store.append_events(
            aggregate_id=expense_id,
            aggregate_type="Expense",
            new_events=[event],
            expected_version=0,
            tenant_id=organization_id,
        )

        # Check approval rules — auto-approve or route to approver role
        rule = await self._find_matching_rule(organization_id, amount, category)
        if rule is not None and rule["auto_approve"]:
            auto_event = ExpenseApproved(
                aggregate_id=expense_id,
                metadata=EventMetadata(user_id=_SYSTEM_UUID),
                approved_by=_SYSTEM_UUID,
            )
            await self._event_store.append_events(
                aggregate_id=expense_id,
                aggregate_type="Expense",
                new_events=[auto_event],
                expected_version=1,
                tenant_id=organization_id,
            )
        elif rule is not None:
            due = datetime.now(timezone.utc) + timedelta(days=7)
            req_event = ExpenseApprovalRequested(
                aggregate_id=expense_id,
                metadata=EventMetadata(user_id=None),
                approver_id=None,
                due_date=due.isoformat(),
                approval_level=1,
            )
            await self._event_store.append_events(
                aggregate_id=expense_id,
                aggregate_type="Expense",
                new_events=[req_event],
                expected_version=1,
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
        # Load current version to avoid hardcoding expected_version=1
        events = await self._event_store.load_events(expense_id, "Expense", organization_id)
        current_version = len(events)
        event = ExpenseApproved(
            aggregate_id=expense_id,
            metadata=EventMetadata(user_id=approver_id),
            approved_by=approver_id,
        )
        await self._event_store.append_events(
            aggregate_id=expense_id,
            aggregate_type="Expense",
            new_events=[event],
            expected_version=current_version,
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
        # Load current version to avoid hardcoding expected_version=1
        events = await self._event_store.load_events(expense_id, "Expense", organization_id)
        current_version = len(events)
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
            expected_version=current_version,
            tenant_id=organization_id,
        )

    # ── HELPERS ───────────────────────────────────────────────────────────────

    async def _find_matching_rule(
        self,
        org_id: UUID,
        amount: Decimal,
        category: Optional[str],
    ) -> Optional[dict]:
        """
        Query approval_rules ordered by priority (lowest first = highest priority).
        Returns the first matching rule, or None if no rule applies.

        Matching logic:
          'amount_above' → amount > condition_value['threshold']
          'category'     → category == condition_value['category']
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT condition_type, condition_value, approver_role, auto_approve
                FROM approval_rules
                WHERE organization_id = $1
                ORDER BY priority ASC
                """,
                org_id,
            )
        for row in rows:
            ct = row["condition_type"]
            cv = row["condition_value"]
            if isinstance(cv, str):
                import json as _json
                cv = _json.loads(cv)
            if ct == "amount_above" and amount > Decimal(str(cv["threshold"])):
                return dict(row)
            if ct == "category" and category == cv.get("category"):
                return dict(row)
        return None

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
