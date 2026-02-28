"""
  Budget Schemas (DTOs)

  Pydantic models for budget request validation and response serialization.

  Endpoints:
    - POST /budgets        → CreateBudgetRequest → BudgetCreatedResponse
    - GET  /budgets/{id}   → BudgetResponse
    - GET  /budgets        → list[BudgetResponse]
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import date, datetime
from uuid import UUID
from decimal import Decimal
from typing import Optional
from enum import Enum

from src.domain import ExpenseCategory, Currency


class BudgetPeriod(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class BudgetStatus(str, Enum):
    ACTIVE = "active"
    THRESHOLD_REACHED = "threshold_reached"
    EXCEEDED = "exceeded"


class CreateBudgetRequest(BaseModel):
    """
    Request body for creating a new spending budget.

    Example:
    {
        "budget_name": "Food & Groceries",
        "category": "food",
        "amount": "3000.00",
        "currency": "NOK",
        "period": "monthly",
        "start_date": "2026-02-01",
        "alert_threshold": 0.8
    }
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "budget_name": "Food & Groceries",
                "category": "food",
                "amount": "3000.00",
                "currency": "NOK",
                "period": "monthly",
                "start_date": "2026-02-01",
                "alert_threshold": 0.8,
            }
        }
    )

    budget_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Human-readable budget name",
    )
    category: ExpenseCategory = Field(
        ...,
        description="Expense category this budget tracks (e.g. food, transport)",
    )
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Spending limit in the given currency",
    )
    currency: Currency = Field(
        default=Currency.NOK,
        description="Currency of the budget (default: NOK)",
    )
    period: BudgetPeriod = Field(
        default=BudgetPeriod.MONTHLY,
        description="Budget reset period",
    )
    start_date: date = Field(
        ...,
        description="First day this budget is active (YYYY-MM-DD)",
    )
    alert_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of limit that triggers a warning (0.0–1.0, default 0.8 = 80%)",
    )


class BudgetCreatedResponse(BaseModel):
    """Minimal confirmation returned after POST /budgets."""
    budget_id: UUID
    status: str = "created"
    message: str


class BudgetResponse(BaseModel):
    """
    Full budget state returned by GET /budgets/{id} and GET /budgets.

    remaining_amount and percentage_used are GENERATED columns in PostgreSQL
    (budget_status table) — the DB computes them, we just surface them here.
    """
    model_config = ConfigDict(from_attributes=True)

    budget_id: UUID
    budget_name: str
    category: str
    amount: Decimal                   # the spending limit
    currency: str
    period: str
    spent_amount: Decimal             # running total from transactions
    remaining_amount: Decimal         # amount - spent_amount (DB computed)
    percentage_used: Decimal          # (spent / amount * 100) (DB computed)
    alert_threshold: float
    start_date: date
    status: BudgetStatus              # active / threshold_reached / exceeded
    created_at: Optional[datetime] = None
