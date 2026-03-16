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
import json
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime, timezone, date

from asyncpg import UniqueViolationError

from src.api.schemas.budget import (
    CreateBudgetRequest,
    BudgetCreatedResponse,
    BudgetResponse,
    BudgetStatus,
    CreateApprovalRuleRequest,
    ApprovalRuleResponse,
)
from src.auth.dependencies import get_current_user, UserContext, require_role
from src.api.dependencies import get_db_pool, get_read_db_pool, get_event_store
from src.domain.events_store import EventStore, AggregateNotFoundError
from src.domain import EventMetadata, BudgetCreated, Currency, ExpenseCategory
router = APIRouter(
    prefix="/budgets",
    tags=["budgets"],
)

# -- CREATE
@router.post(
    "/",
    response_model=BudgetCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new spending budget",
)
async def create_budget(
    request: CreateBudgetRequest,
    current_user: UserContext = Depends(get_current_user),
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
        tenant_id=current_user.organization_id, #POST uses 0, this uses current version
    )

    return BudgetCreatedResponse(
        budget_id=event.aggregate_id,
        status="created",
        message=f"Budget '{request.budget_name}' created successfully",
    )


# -- APPROVAL RULES ------------------------------------------------------------
# ⚠ These MUST be registered BEFORE GET /{budget_id}.
# FastAPI matches routes top-to-bottom; without this ordering the string
# "approval-rules" would be parsed as a UUID parameter and return 422.

@router.post(
    "/approval-rules",
    response_model=ApprovalRuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an approval rule",
)
async def create_approval_rule(
    request: CreateApprovalRuleRequest,
    current_user: UserContext = Depends(require_role("owner", "admin")),
    db_pool=Depends(get_db_pool),
) -> ApprovalRuleResponse:
    """
    Insert a new approval rule for this organisation.

    Rules are evaluated in priority order (lowest number first) when an expense
    is submitted. The first matching rule determines whether the expense is
    auto-approved or routed to an approver role.
    """
    async with db_pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO approval_rules (
                    organization_id, name, condition_type, condition_value,
                    approver_role, auto_approve, priority
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id, name, condition_type, condition_value,
                          approver_role, auto_approve, priority
                """,
                current_user.organization_id,
                request.name,
                request.condition_type.value,
                json.dumps(request.condition_value),
                request.approver_role,
                request.auto_approve,
                request.priority,
            )
        except UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An approval rule with priority {request.priority} already exists.",
            )
    row_dict = dict(row)
    if isinstance(row_dict.get("condition_value"), str):
        row_dict["condition_value"] = json.loads(row_dict["condition_value"])
    return ApprovalRuleResponse(**row_dict)


@router.get(
    "/approval-rules",
    response_model=list[ApprovalRuleResponse],
    summary="List approval rules for the organisation",
)
async def list_approval_rules(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_read_db_pool),  # GET → replica
) -> list[ApprovalRuleResponse]:
    """Return all approval rules for this org, ordered by priority (lowest first)."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, condition_type, condition_value,
                   approver_role, auto_approve, priority
            FROM approval_rules
            WHERE organization_id = $1
            ORDER BY priority ASC
            """,
            current_user.organization_id,
        )
    result = []
    for r in rows:
        r_dict = dict(r)
        if isinstance(r_dict.get("condition_value"), str):
            r_dict["condition_value"] = json.loads(r_dict["condition_value"])
        result.append(ApprovalRuleResponse(**r_dict))
    return result


# -- GET BY ID -----------------------------------------------------------------

@router.get(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Get budget details by ID",
)
async def get_budget(
    budget_id: UUID,
    current_user: UserContext = Depends(get_current_user),
    event_store: EventStore = Depends(get_event_store),
    db_pool=Depends(get_read_db_pool),  # GET → replica (fast path budget_status query)
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
            tenant_id=current_user.organization_id,
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


#  LIST ALL 

@router.get(
    "/",
    response_model=list[BudgetResponse],
    summary="List all budgets",
)
async def list_budgets(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_read_db_pool),  # GET → replica
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
            current_user.organization_id,
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
