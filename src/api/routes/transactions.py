"""
  Transaction API Routes

  Endpoints for managing financial transactions:
    - POST /transactions - Create new transaction
    - GET /transactions/{id} - Get transaction by ID (event replay)
    - GET /transactions - List all transactions (with pagination)

  Pattern: CQRS - Commands create events, Queries replay events to build state
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from uuid import UUID, uuid4
from datetime import datetime, timezone
import json
from src.domain import EventMetadata, TransactionCreated, TransactionCategorized, TransactionDisputed

from src.api.schemas.transaction import (
    CreateTransactionRequest,
    TransactionResponse,
    TransactionCreatedResponse,
    TransactionListResponse,
    CategorizeTransactionRequest,
    TransactionUpdatedResponse,
    DisputeTransactionRequest


)
from src.api.dependencies import get_event_store
from src.domain.events_store import EventStore, AggregateNotFoundError


router = APIRouter(
    prefix="/transactions",
    tags=["transactions"]
)

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


@router.post(
    "/",
    response_model=TransactionCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new transaction",
)
async def create_transaction(
    request: CreateTransactionRequest,
    event_store: EventStore = Depends(get_event_store),
) -> TransactionCreatedResponse:
    """
    Creates a transaction by appending a TransactionCreated event.

    The event is stored in the event store AND written to the outbox
    for Kafka delivery (outbox pattern guarantees at-least-once delivery).
    """
    transaction_id = uuid4()

    event = TransactionCreated(
        aggregate_id=transaction_id,
        metadata=EventMetadata(),
        amount=request.amount,
        currency=request.currency.value,
        transaction_type=request.transaction_type.value,
        merchant_name=request.merchant_name,
        description=request.description,
        category=request.category.value if request.category else None,
    )

    await event_store.append_events(
        aggregate_id=transaction_id,
        aggregate_type="Transaction",
        new_events=[event],
        expected_version=0,
        tenant_id=TENANT_ID,
    )

    return TransactionCreatedResponse(
        transaction_id=transaction_id,
        status="created",
        message=f"Transaction at '{request.merchant_name}' for {request.amount} {request.currency.value} created",
    )


"""
Event Replay Handlers

Each handler is a pure function: (state, event_data) -> None (mutates state dict).
To support a new event type, add one function and register it in EVENT_HANDLERS.
Unknown events are logged and skipped — this makes the system forward-compatible
when new event types are added to the domain without redeploying the API.
"""

import logging

logger = logging.getLogger(__name__)


def _apply_created(state: dict, data: dict, created_at) -> None:
    """Initialize transaction state from the first event."""
    state["transaction_id"] = data["aggregate_id"]
    state["amount"] = data.get("amount", "0")
    state["currency"] = data.get("currency", "NOK")
    state["transaction_type"] = data.get("transaction_type", "debit")
    state["merchant_name"] = data.get("merchant_name", "Unknown")
    state["description"] = data.get("description")
    state["category"] = data.get("category")
    state["created_at"] = created_at


def _apply_categorized(state: dict, data: dict, created_at) -> None:
    """Overwrite category — ML model or user recategorized the transaction."""
    state["category"] = data.get("category")


def _apply_disputed(state: dict, data: dict, created_at) -> None:
    """Mark transaction as disputed — freezes the amount in real banking."""
    state["is_disputed"] = True
    state["dispute_reason"] = data.get("reason")


def _apply_dispute_resolved(state: dict, data: dict, created_at) -> None:
    """Clear dispute flag. Resolution type (chargeback/upheld) is in the event for audit."""
    state["is_disputed"] = False
    state["dispute_reason"] = None


# Dispatch dict — maps event_type strings to handler functions.
# Adding a new event = one function + one line here. No if/elif chain to modify.
EVENT_HANDLERS: dict[str, callable] = {
    "TransactionCreated": _apply_created,
    "TransactionCategorized": _apply_categorized,
    "TransactionDisputed": _apply_disputed,
    "TransactionDisputeResolved": _apply_dispute_resolved,
}


def replay_transaction_events(events: list[dict]) -> dict:
    """
    Reconstruct current transaction state by replaying all events in order.

    This is a left-fold: start with default state, apply each event sequentially.
    The final state reflects every change that ever happened to this transaction.
    """
    state = {
        "transaction_id": None,
        "amount": "0",
        "currency": "NOK",
        "transaction_type": "debit",
        "merchant_name": "Unknown",
        "description": None,
        "category": None,
        "is_disputed": False,
        "dispute_reason": None,
        "version": 0,
        "created_at": None,
    }

    for event in events:
        event_type = event["event_type"]
        data = event["event_data"]
        created_at = event["created_at"]

        handler = EVENT_HANDLERS.get(event_type)
        if handler:
            handler(state, data, created_at)
        else:
            # Forward-compatible: skip unknown events instead of crashing
            logger.warning("Unknown event type '%s' skipped during replay", event_type)

        state["version"] = event["version"]

    return state


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    summary="Get transaction details by ID",
)
async def get_transaction(
    transaction_id: UUID,
    event_store: EventStore = Depends(get_event_store),
) -> TransactionResponse:
    """
    Loads all events for a transaction and replays them to build current state.

    This is the Query side of CQRS — no state is stored directly,
    it's reconstructed from the event stream every time.
    """
    try:
        events = await event_store.load_events(
            aggregate_id=transaction_id,
            aggregate_type="Transaction",
            tenant_id=TENANT_ID,
        )
    except AggregateNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found",
        )

    state = replay_transaction_events(events)
    return TransactionResponse(**state)


@router.get(
    "/",
    response_model=TransactionListResponse,
    summary="List transactions with pagination",
)
async def list_transactions(
    limit: int = Query(default=20, ge=1, le=100, description="Max transactions to return"),
    offset: int = Query(default=0, ge=0, description="Number of transactions to skip"),
    event_store: EventStore = Depends(get_event_store),
) -> TransactionListResponse:
    """
    Lists transactions by querying the event store for all Transaction aggregates.

    Fetches the first event (TransactionCreated) for each aggregate to build
    a summary view. For full state with categorization/disputes, use GET by ID.
    """
    async with event_store.db_pool.acquire() as conn:
        # Count distinct transaction aggregates
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT aggregate_id)
            FROM events
            WHERE aggregate_type = 'Transaction' AND tenant_id = $1
            """,
            TENANT_ID,
        )

        # Get paginated list of transaction aggregate IDs (ordered by first event)
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (aggregate_id)
                aggregate_id, event_data, version, created_at
            FROM events
            WHERE aggregate_type = 'Transaction'
              AND tenant_id = $1
              AND event_type = 'TransactionCreated'
            ORDER BY aggregate_id, created_at ASC
            LIMIT $2 OFFSET $3
            """,
            TENANT_ID,
            limit,
            offset,
        )

    transactions = []
    for row in rows:
        data = json.loads(row["event_data"])
        transactions.append(
            TransactionResponse(
                transaction_id=row["aggregate_id"],
                amount=data.get("amount", "0"),
                currency=data.get("currency", "NOK"),
                transaction_type=data.get("transaction_type", "debit"),
                merchant_name=data.get("merchant_name", "Unknown"),
                description=data.get("description"),
                category=data.get("category"),
                is_disputed=False,
                version=row["version"],
                created_at=row["created_at"],
            )
        )

    return TransactionListResponse(
        transactions=transactions,
        total=total or 0,
        limit=limit,
        offset=offset,
    )
@router.put(
    "/{transaction_id}/categorize",
    response_model=TransactionUpdatedResponse,
    summary="Categorize a transaction",
)
async def categorize_transaction(
    transaction_id: UUID,
    request: CategorizeTransactionRequest,
    event_store: EventStore = Depends(get_event_store),
) -> TransactionUpdatedResponse:
    """
    Categorizes a transaction.

    This is an idempotent operation — if already categorized, it will update the category.
        
      Appends a TransactionCategorized event to an existing transaction.

      This is the first endpoint that uses optimistic concurrency with
      expected_version > 0. The flow:
        1. Load all existing events (gives us current version)
        2. Create the new event
        3. Append with expected_version = current version
           → If someone else modified this transaction between step 1 and 3,
             the EventStore raises ConcurrencyError (409 Conflict)
    """
     # Step 1: Load current state to get the version

    try:
        events = await event_store.load_events(
            aggregate_id=transaction_id,
            aggregate_type="Transaction",
            tenant_id=TENANT_ID,
        )
    except AggregateNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found",
        )
    current_version = len(events) #each event increments version by 1, so version = number of events
    # Step 2: Replay to get current state — we need the old category for the event data
    current_state=replay_transaction_events(events)
    #  Step 3: Build the domain event
    event= TransactionCategorized(
        aggregate_id=transaction_id,# Same aggregate we're modifyin
        metadata=EventMetadata(), # Fresh event_id, timestamp, correlation_id
        category=request.category.value,
        confidence_score=request.confidence_score,
        categorized_by=request.categorized_by,
        previous_category=current_state.get("category") # For audit/logging — what was the old category before this change?
    )
    # Step 4: Append with optimistic concurrency check
    new_version = await event_store.append_events(
        aggregate_id=transaction_id,
        aggregate_type="Transaction",
        new_events=[event],
        expected_version=current_version, # This ensures we don't overwrite concurrent changes, the key difference from POST
        tenant_id=TENANT_ID, #POST uses 0, this uses current version
    )
    return TransactionUpdatedResponse(
        transaction_id=transaction_id,
        status="categorized",
        version=new_version,
        message=f"Transaction categorized as {request.category.value} by {request.categorized_by}",
    )   
@router.put (
    "/{transaction_id}/dispute",
    response_model=TransactionUpdatedResponse,
    summary="Dispute a transaction",
)
async def dispute_transaction(
    transaction_id: UUID,
    request: DisputeTransactionRequest,
    event_store: EventStore = Depends(get_event_store),
) -> TransactionUpdatedResponse:
    """
      Appends a TransactionDisputed event.

      Includes a business rule check: you cannot dispute an already-disputed
      transaction. This is domain logic enforced at the API layer by
      replaying events first to check current state.
    """
     # Step 1: Load current state to get the version

    try:
        events = await event_store.load_events(
            aggregate_id=transaction_id,
            aggregate_type="Transaction",
            tenant_id=TENANT_ID,
        )
    except AggregateNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found",
        )
    current_version = len(events)
    current_state = replay_transaction_events(events)
    # Business rule: can't dispute twice
    if current_state.get("is_disputed"):
        raise HTTPException(
            status_code=400,
            detail=f"Transaction {transaction_id} is already disputed",
        )
    event= TransactionDisputed(
        aggregate_id=transaction_id,# Same aggregate we're modifyin
        metadata=EventMetadata(), # Fresh event_id, timestamp, correlation_id
        reason=request.reason
    )
    # Step 4: Append with optimistic concurrency check
    new_version = await event_store.append_events(
        aggregate_id=transaction_id,
        aggregate_type="Transaction",
        new_events=[event],
        expected_version=current_version, # This ensures we don't overwrite concurrent changes, the key difference from POST
        tenant_id=TENANT_ID, #POST uses 0, this uses current version
    )
    return TransactionUpdatedResponse(
        transaction_id=transaction_id,
        status="disputed",
        message=f"Transaction disputed for reason: {request.reason}",
        version=new_version
    )