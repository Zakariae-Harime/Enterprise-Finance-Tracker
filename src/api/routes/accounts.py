"""
  Account API Routes

  Endpoints for managing financial accounts:
    - POST /accounts - Create new account
    - GET /accounts/{id} - Get account by ID
    - GET /accounts - List all accounts (with pagination)

  Pattern: Each endpoint follows Command/Query separation
    - POST/PUT/DELETE = Commands (change state, use EventStore)
    - GET = Queries (read state, could use read model)
APIROUTER Groups all /accounts/* endpoints together

"""
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime, timezone

from src.auth.dependencies import get_current_user, UserContext
from src.api.schemas.account import ( AccountCreatingRequest,   #Validates incoming JSON for POST /accounts
AccountResponse, # Defines response shape after creation 
AccountCreatedResponse)  #Defines response shape for GET /accounts/{id} /Fetching 
from src.api.dependencies import get_db_pool, get_event_store #Function that creates EventStore with shared db pool from dependencies.py that reads request.app.state.db_pool and return EventStore(pool)
from src.domain.events_store import EventStore,AggregateNotFoundError
from src.domain import EventMetadata, AccountCreated

#creating router instance
router = APIRouter(
    prefix="/accounts",
    tags=["accounts"] # Groups endpoints in Swagger UI under "Accounts" section
)
#creating account endpoint
@router.post("/",
              response_model=AccountCreatedResponse, # FastAPI validates response matches this schema
              status_code=status.HTTP_201_CREATED,
              summary="Create a new account")
async def create_account(
    request: AccountCreatingRequest,  #FastAPI auto-parses JSON body into this Pydantic model, also validates required fields and types
    current_user: UserContext = Depends(get_current_user), #Injects authenticated user info from
    event_store: EventStore = Depends(get_event_store) 
) -> AccountCreatedResponse:
    """  
    When request comes in:                                
                                                     
    1. FastAPI sees Depends(get_event_store)              
    2. Calls get_event_store(request)                     
    3. get_event_store reads request.app.state.db_pool    
    4. Returns EventStore(pool)                           
    5. FastAPI passes it to create_account()     
    """
    event = AccountCreated (
    aggregate_id = uuid4(),
    metadata = EventMetadata(),
    account_name=request.name,
    currency=request.currency.value, #.value converts Currency.NOK → "NOK" string
    account_type=request.account_type.value,
    initial_balance=request.initial_balance
    )
    tenant_id = current_user.organization_id
    await event_store.append_events(
        aggregate_id=event.aggregate_id,
        aggregate_type="Account",
        new_events=[event],
        expected_version=0,         
        tenant_id=tenant_id
    )
    return AccountCreatedResponse(
        account_id=event.aggregate_id,
        status="created",
        message=f"Account '{request.name}' created successfully"
    )
@router.get("/{account_id}",
            response_model=AccountResponse,
            summary="Get account details by ID")
async def get_account(
    account_id: UUID,
    event_store: EventStore = Depends(get_event_store),
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),              # NEW: direct pool for projection query
) -> AccountResponse:
    """
    CQRS Query: tries the fast read model first, falls back to event replay.

    Fast path  → account_projections table (O(1) single row lookup)
    Slow path  → event replay (O(n) — used before projection is built)

    Why two paths? The projection is built asynchronously by the Kafka consumer.
    If the consumer hasn't processed the AccountCreated event yet (eventual
    consistency window), the projection row doesn't exist and we fall back.
    In production, this window is milliseconds.
    """
    tenant_id = UUID("00000000-0000-0000-0000-000000000001")

    # ── FAST PATH: Read from projection table ─────────────────────────────
    async with db_pool.acquire() as conn:
        projection = await conn.fetchrow(
            """
            SELECT account_id, bank_name, account_type, currency,
                   current_balance, last_event_version, created_at
            FROM account_projections
            WHERE account_id = $1
            """,
            account_id,
        )

    if projection:
        # Projection exists — return instantly without touching the event store
        return AccountResponse(
            account_id=projection["account_id"],
            name=projection["bank_name"] or "Unknown",
            currency=projection["currency"] or "NOK",
            account_type=projection["account_type"] or "checking",
            balance=projection["current_balance"] or Decimal("0.00"),
            version=projection["last_event_version"] or 0,
            created_at=projection["created_at"],
        )

    # ── SLOW PATH: Projection not ready yet — replay events ───────────────
    # This runs when: consumer hasn't processed the event yet, or
    # account_projections table is empty (fresh environment).
    try:
        events = await event_store.load_events(
            aggregate_id=account_id,
            aggregate_type="Account",
            tenant_id=tenant_id,
        )
    except AggregateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    first_event = events[0]
    event_data = first_event["event_data"]
    return AccountResponse(
        account_id=event_data.get("aggregate_id", str(account_id)),
        name=event_data.get("account_name", "Unknown"),
        currency=event_data.get("currency", "NOK"),
        account_type=event_data.get("account_type", "checking"),
        balance=Decimal(event_data.get("initial_balance", "0.00")),
        version=len(events),
        created_at=first_event["created_at"],
    )
