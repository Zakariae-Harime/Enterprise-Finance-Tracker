"""
  Budget API Routes

  Endpoints for managing spending budgets:
    - POST /budgets           - Create a new budget
    - GET  /budgets/{id}      - Get budget by ID
    - GET  /budgets           - List all budgets for the tenant

  Pattern: CQRS
    - POST = Command  -> writes BudgetCreated event to EventStore
    - GET  = Query    -> reads from budget_status projection table (fast path)
                         falls back to event replay (slow path / eventual consistency)
"""
import calendar
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime, timezone, date

from src.api.schemas.budget import (
    CreateBudgetRequest,
    BudgetCreatedResponse,
    BudgetResponse,
    BudgetStatus,
)
from src.api.dependencies import get_db_pool, get_event_store
from src.domain.events_store import EventStore, AggregateNotFoundError
from src.domain import EventMetadata, BudgetCreated, Currency, ExpenseCategory

router = APIRouter(
    prefix="/budgets",
    tags=["budgets"],
)

# Hardcoded tenant — same pattern as accounts and transactions routes
TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


# -- CREATE --------------------------------------------------------------------

@router.post(
    "/",
    response_model=BudgetCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new spending budget",
)
async def create_budget(
    request: CreateBudgetRequest,
    event_store: EventStore = Depends(get_event_store),
) -> BudgetCreatedResponse:
    """
    Command: stores a BudgetCreated event in the EventStore.

    The BudgetProjectionConsumer picks this up from Kafka and INSERTs
    a row into budget_status, giving GET /budgets/{id} its fast read path.
    """
    # Convert date -> datetime so BudgetCreated (which stores datetime) is satisfied
    start_dt = datetime(
        request.start_date.year,
        request.start_date.month,
        request.start_date.day,
        tzinfo=timezone.utc,
    )
    last_day = calendar.monthrange(request.start_date.year, request.start_date.month)[1]
    end_dt = datetime(
        request.start_date.year,
        request.start_date.month,
        last_day,
        tzinfo=timezone.utc,
    )

    event = BudgetCreated(
        aggregate_id=uuid4(),
        metadata=EventMetadata(),
        budget_name=request.budget_name,
        amount=request.amount,
        currency=Currency(request.currency.value),
        period=request.period.value,
        start_date=start_dt,
        end_date=end_dt,
        alert_threshold=request.alert_threshold,
        category=ExpenseCategory(request.category.value) if request.category else None,
    )

    await event_store.append_events(
        aggregate_id=event.aggregate_id,
        aggregate_type="Budget",
        new_events=[event],
        expected_version=0,
        tenant_id=TENANT_ID,
    )

    return BudgetCreatedResponse(
        budget_id=event.aggregate_id,
        status="created",
        message=f"Budget '{request.budget_name}' created successfully",
    )


# -- GET BY ID -----------------------------------------------------------------

@router.get(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Get budget details by ID",
)
async def get_budget(
    budget_id: UUID,
    event_store: EventStore = Depends(get_event_store),
    db_pool=Depends(get_db_pool),
) -> BudgetResponse:
    """
    CQRS Query: fast path (budget_status table) with event replay fallback.

    Fast path  -> budget_status table (O(1) single row lookup)
    Slow path  -> event replay from events table (O(n))
    """
    # -- FAST PATH -------------------------------------------------------------
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, category, month, budget_amount,
                   spent_amount, remaining_amount, percentage_used,
                   alert_threshold_reached, updated_at
            FROM budget_status
            WHERE id = $1
            """,
            budget_id,
        )

    if row:
        spent = row["spent_amount"] or Decimal("0.00")
        limit = row["budget_amount"] or Decimal("0.00")

        if spent >= limit:
            status_val = BudgetStatus.EXCEEDED
        elif row["alert_threshold_reached"]:
            status_val = BudgetStatus.THRESHOLD_REACHED
        else:
            status_val = BudgetStatus.ACTIVE

        return BudgetResponse(
            budget_id=budget_id,
            budget_name=row["category"],
            category=row["category"],
            amount=limit,
            currency="NOK",
            period="monthly",
            spent_amount=spent,
            remaining_amount=row["remaining_amount"] or Decimal("0.00"),
            percentage_used=row["percentage_used"] or Decimal("0.00"),
            alert_threshold=0.8,
            start_date=row["month"],
            status=status_val,
            created_at=row["updated_at"],
        )

    # -- SLOW PATH: event replay -----------------------------------------------
    try:
        events = await event_store.load_events(
            aggregate_id=budget_id,
            aggregate_type="Budget",
            tenant_id=TENANT_ID,
        )
    except AggregateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Budget {budget_id} not found")

    first_event = events[0]
    event_data = first_event["event_data"]

    return BudgetResponse(
        budget_id=budget_id,
        budget_name=event_data.get("budget_name", "Unknown"),
        category=event_data.get("category") or "unknown",
        amount=Decimal(str(event_data.get("amount", "0.00"))),
        currency=event_data.get("currency", "NOK"),
        period=event_data.get("period", "monthly"),
        spent_amount=Decimal("0.00"),        # no transactions processed yet
        remaining_amount=Decimal(str(event_data.get("amount", "0.00"))),
        percentage_used=Decimal("0.00"),
        alert_threshold=event_data.get("alert_threshold", 0.8),
        start_date=date.fromisoformat(event_data["start_date"][:10]),
        status=BudgetStatus.ACTIVE,
        created_at=first_event["created_at"],
    )


# -- LIST ALL ------------------------------------------------------------------

@router.get(
    "/",
    response_model=list[BudgetResponse],
    summary="List all budgets",
)
async def list_budgets(
    db_pool=Depends(get_db_pool),
) -> list[BudgetResponse]:
    """Query: all budgets for the current tenant from the projection table."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, category, month, budget_amount,
                   spent_amount, remaining_amount, percentage_used,
                   alert_threshold_reached, updated_at
            FROM budget_status
            WHERE user_id = $1
            ORDER BY month DESC
            """,
            TENANT_ID,
        )

    result = []
    for row in rows:
        spent = row["spent_amount"] or Decimal("0.00")
        limit = row["budget_amount"] or Decimal("0.00")

        if spent >= limit:
            status_val = BudgetStatus.EXCEEDED
        elif row["alert_threshold_reached"]:
            status_val = BudgetStatus.THRESHOLD_REACHED
        else:
            status_val = BudgetStatus.ACTIVE

        result.append(BudgetResponse(
            budget_id=row["id"],
            budget_name=row["category"],
            category=row["category"],
            amount=limit,
            currency="NOK",
            period="monthly",
            spent_amount=spent,
            remaining_amount=row["remaining_amount"] or Decimal("0.00"),
            percentage_used=row["percentage_used"] or Decimal("0.00"),
            alert_threshold=0.8,
            start_date=row["month"],
            status=status_val,
            created_at=row["updated_at"],
        ))

    return result
