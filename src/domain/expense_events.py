"""
Expense domain events.

Three events cover the full lifecycle:
  ExpenseSubmitted  → employee submits for approval
  ExpenseApproved   → approver accepts
  ExpenseRejected   → approver declines with reason
"""
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID
from typing import Optional

from src.domain import DomainEvent


@dataclass(frozen=True)
class ExpenseSubmitted(DomainEvent):
    submitted_by: UUID          # user_id of the submitter
    amount: Decimal
    currency: str               # 'NOK', 'EUR', etc.
    description: str
    merchant_name: str
    expense_date: str           # ISO date string 'YYYY-MM-DD'
    category: Optional[str] = None


@dataclass(frozen=True)
class ExpenseApproved(DomainEvent):
    approved_by: UUID           # user_id of the approver


@dataclass(frozen=True)
class ExpenseRejected(DomainEvent):
    rejected_by: UUID           # user_id of the rejector
    reason: str
