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
    request: AccountCreatingRequest,  #FastAPI auto-parses JSON body into thi
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
    account_id = uuid4(),
    metadata = EventMetadata(),
    account_name=request.name,
    currency=request.currency.value, #.value converts Currency.NOK → "NOK" string
    account_type=request.account_type.value,
    initial_balance=request.initial_balance
    )
    tenant_id = UUID("00000000-0000-0000-0000-000000000001")

    await event_store.append_event(
        aggregate_id= account_id,
        aggregate_type="Account",
        new_events=[event],
        expected_version=0,         
        tenant_id=tenant_id
    )
    return AccountCreatedResponse(
        account_id=event.account_id,
        status="created",
        message=f"Account '{request.name}' created successfully"
    )
@router.get("/{account_id}",
            response_model=AccountResponse,
            summary="Get account details by ID")
async def get_account(
    account_id: UUID,
    event_store: EventStore = Depends(get_event_store)
) -> AccountResponse:
    
    tenant_id = UUID("00000000-0000-0000-0000-000000000001")
    try:
        events = await event_store.load_events(
            aggregate_id=account_id,
            aggregate_type="Account",
            tenant_id=tenant_id
        )
    except AggregateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
     # Replay events to build current state
    first_event = events[0]
    event_data = first_event["event_data"]
    return AccountCreated (
     account_id=event_data["account_id"],
     name=event_data.get("account_name", "unknown"),
     currency=event_data.get("currency", "NOK"),
     account_type=event_data.get("account_type", "checking"),
     initial_balance=Decimal(event_data.get("initial_balance", "0.00"))
     version=len(events)
     created_at=first_event["created_at"]
 )