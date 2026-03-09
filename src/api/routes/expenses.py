"""
Expense API Routes — thin HTTP layer only.

All business logic lives in ApprovalService.
Routes translate service exceptions into HTTP status codes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from uuid import UUID
from decimal import Decimal
from datetime import date
from typing import Optional

from src.auth.dependencies import get_current_user, UserContext
from src.api.dependencies import get_db_pool, get_event_store
from src.services.approval_service import (
    ApprovalService,
    ExpenseNotFoundError,
    InvalidStateTransitionError,
)
from src.services.approval_rules import InsufficientApprovalAuthorityError

router = APIRouter(prefix="/expenses", tags=["expenses"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SubmitExpenseRequest(BaseModel):
    amount: Decimal
    currency: str = "NOK"
    description: str
    merchant_name: str
    expense_date: date
    category: Optional[str] = None


class ExpenseResponse(BaseModel):
    expense_id: UUID
    status: str
    message: str


class RejectRequest(BaseModel):
    reason: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _service(db_pool, event_store) -> ApprovalService:
    return ApprovalService(pool=db_pool, event_store=event_store)


# ── POST /expenses/ ───────────────────────────────────────────────────────────

@router.post("/", response_model=ExpenseResponse, status_code=status.HTTP_201_CREATED)
async def submit_expense(
    request: SubmitExpenseRequest,
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
    event_store=Depends(get_event_store),
) -> ExpenseResponse:
    service = _service(db_pool, event_store)
    expense_id = await service.submit(
        submitted_by=current_user.user_id,
        organization_id=current_user.organization_id,
        amount=request.amount,
        currency=request.currency,
        description=request.description,
        merchant_name=request.merchant_name,
        expense_date=str(request.expense_date),
        category=request.category,
    )
    return ExpenseResponse(expense_id=expense_id, status="pending", message="Expense submitted for approval")


# ── POST /expenses/{id}/approve ───────────────────────────────────────────────

@router.post("/{expense_id}/approve", response_model=ExpenseResponse)
async def approve_expense(
    expense_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
    event_store=Depends(get_event_store),
) -> ExpenseResponse:
    service = _service(db_pool, event_store)
    try:
        await service.approve(
            expense_id=expense_id,
            approver_id=current_user.user_id,
            organization_id=current_user.organization_id,
        )
    except ExpenseNotFoundError:
        raise HTTPException(status_code=404, detail="Expense not found")
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except InsufficientApprovalAuthorityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return ExpenseResponse(expense_id=expense_id, status="approved", message="Expense approved")


# ── POST /expenses/{id}/reject ────────────────────────────────────────────────

@router.post("/{expense_id}/reject", response_model=ExpenseResponse)
async def reject_expense(
    expense_id: UUID,
    request: RejectRequest,
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
    event_store=Depends(get_event_store),
) -> ExpenseResponse:
    service = _service(db_pool, event_store)
    try:
        await service.reject(
            expense_id=expense_id,
            rejector_id=current_user.user_id,
            organization_id=current_user.organization_id,
            reason=request.reason,
        )
    except ExpenseNotFoundError:
        raise HTTPException(status_code=404, detail="Expense not found")
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except InsufficientApprovalAuthorityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return ExpenseResponse(expense_id=expense_id, status="rejected", message="Expense rejected")


# ── GET /expenses/ ────────────────────────────────────────────────────────────

@router.get("/", response_model=list[dict])
async def list_expenses(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
) -> list[dict]:
    """Employees see only their own. Finance/Admin/Owner see all org expenses."""
    async with db_pool.acquire() as conn:
        if current_user.role in ("finance", "admin", "owner"):
            rows = await conn.fetch(
                "SELECT * FROM expenses WHERE organization_id = $1 ORDER BY expense_date DESC",
                current_user.organization_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM expenses WHERE organization_id = $1 AND submitted_by = $2 ORDER BY expense_date DESC",
                current_user.organization_id,
                current_user.user_id,
            )
    return [dict(r) for r in rows]
