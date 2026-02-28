"""
E2E tests for Transaction API endpoints.

Tests the full HTTP stack:
  POST /api/v1/transactions/               → 201 + transaction_id
  GET  /api/v1/transactions/{id}           → event replay (no projection table)
  PUT  /api/v1/transactions/{id}/categorize → optimistic concurrency write
  PUT  /api/v1/transactions/{id}/dispute    → business rule enforcement
"""
import pytest
from uuid import UUID
from decimal import Decimal
pytestmark = pytest.mark.asyncio

# A fixed account_id to attach transactions to.
# The transaction route doesn't validate that this account exists — it just stores
# the UUID in the event_data. So we can use any UUID here.
ACCOUNT_ID = UUID("00000000-0000-0000-0000-000000000001")

# Base valid transaction payload — reused across tests.
# Pattern: copy this dict and override ONE field per validation test.
# "{**_REMA_TX, 'amount': '0'}" copies all 7 fields but replaces amount.
_REMA_TX = {
    "amount": "250.00",
    "currency": "NOK",
    "transaction_type": "debit",
    "merchant_name": "REMA 1000 Grünerløkka",
    "description": "Weekly groceries",
    "category": "meals",
    "account_id": str(ACCOUNT_ID),
}


class TestCreateTransaction:
    """POST /api/v1/transactions/ — the Command side of CQRS."""

    async def test_return_201_with_transaction_id(self, api_client):
        """
        Creating a transaction must return 201 + a server-generated transaction_id.
        """
        response = await api_client.post("/api/v1/transactions/", json=_REMA_TX)

        assert response.status_code == 201
        body = response.json()
        assert "transaction_id" in body
        assert body["status"] == "created"
        UUID(body["transaction_id"])  # raises ValueError if not valid UUID

    async def test_event_written_to_event_store(self, api_client, db_pool):
        """
        After creating a transaction, the events table must have a TransactionCreated row.
        Proves the route correctly calls EventStore.append_events().
        """
        response = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = UUID(response.json()["transaction_id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM events WHERE aggregate_id = $1 ORDER BY created_at DESC LIMIT 1",
                tx_id,
            )

        assert row is not None
        assert row["event_type"] == "TransactionCreated"

    async def test_outbox_row_created(self, api_client, db_pool):
        """
        Proves the outbox pattern: event + outbox written atomically.
        published_at = None means OutboxRelay hasn't picked it up yet.
        """
        response = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = UUID(response.json()["transaction_id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT event_type, published_at FROM outbox WHERE aggregate_id = $1",
                tx_id,
            )

        assert row is not None
        assert row["event_type"] == "TransactionCreated"
        assert row["published_at"] is None  # not yet sent to Kafka

    async def test_zero_amount_returns_422(self, api_client):
        """
        Pydantic schema has gt=0 on amount — zero is not allowed.
        422 = Unprocessable Entity (FastAPI's validation error response).
        """
        payload = {**_REMA_TX, "amount": "0"}  # override just the amount
        response = await api_client.post("/api/v1/transactions/", json=payload)

        assert response.status_code == 422

    async def test_invalid_type_returns_422(self, api_client):
        """
        TransactionType enum only allows credit/debit/transfer.
        """
        payload = {**_REMA_TX, "transaction_type": "INVALID"}
        response = await api_client.post("/api/v1/transactions/", json=payload)

        assert response.status_code == 422


class TestGetTransaction:
    """GET /api/v1/transactions/{id} — the Query side of CQRS."""

    async def test_get_nonexistent_returns_404(self, api_client):
        """
        No events for this UUID → AggregateNotFoundError → route converts to 404.
        """
        response = await api_client.get(
            "/api/v1/transactions/00000000-0000-0000-0000-000000000099"
        )
        assert response.status_code == 404

    async def test_get_returns_correct_data(self, api_client):
        """
        POST a transaction → GET it → verify event replay reconstructed the correct state.
        No projection table involved — this is pure event replay every time.
        """
        create_resp = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = create_resp.json()["transaction_id"]

        get_resp = await api_client.get(f"/api/v1/transactions/{tx_id}")

        assert get_resp.status_code == 200
        body = get_resp.json()
        assert Decimal(body["amount"]) == Decimal("250.00")
        assert body["merchant_name"] == "REMA 1000 Grünerløkka"
        assert body["currency"] == "NOK"
        assert body["transaction_type"] == "debit"
        assert body["is_disputed"] is False


class TestCategorizeTransaction:
    """PUT /api/v1/transactions/{id}/categorize — optimistic concurrency."""

    async def test_categorize_returns_200_and_version_2(self, api_client):
        """
        After creating (version 1) then categorizing (version 2),
        the response version must be 2.
        This proves the EventStore incremented the version correctly.
        """
        # Step 1: Create the transaction (version becomes 1)
        create_resp = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = create_resp.json()["transaction_id"]

        # Step 2: Categorize it (version becomes 2)
        cat_resp = await api_client.put(
            f"/api/v1/transactions/{tx_id}/categorize",
            json={"category": "supplies", "categorized_by": "user"},
        )

        assert cat_resp.status_code == 200
        body = cat_resp.json()
        assert body["status"] == "categorized"
        assert body["version"] == 2  # 1 event created + 1 event categorized = version 2

    async def test_categorize_nonexistent_returns_404(self, api_client):
        """
        Categorizing a transaction that doesn't exist → 404.
        The route calls load_events first, which raises AggregateNotFoundError.
        """
        response = await api_client.put(
            "/api/v1/transactions/00000000-0000-0000-0000-000000000099/categorize",
            json={"category": "meals", "categorized_by": "user"},
        )
        assert response.status_code == 404


class TestDisputeTransaction:
    """PUT /api/v1/transactions/{id}/dispute — business rule enforcement."""

    async def test_dispute_returns_200_and_marks_disputed(self, api_client):
        """
        Disputing a valid transaction → 200 + status "disputed" + version 2.
        """
        create_resp = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = create_resp.json()["transaction_id"]

        dispute_resp = await api_client.put(
            f"/api/v1/transactions/{tx_id}/dispute",
            json={"reason": "Unauthorized charge - I did not make this purchase"},
        )

        assert dispute_resp.status_code == 200
        body = dispute_resp.json()
        assert body["status"] == "disputed"
        assert body["version"] == 2

    async def test_double_dispute_returns_400(self, api_client):
        """
        Business rule: cannot dispute an already-disputed transaction.
        Route replays events, sees is_disputed=True, raises HTTPException(400).
        This is 400 (bad request / invalid operation), NOT 404 (not found).
        """
        create_resp = await api_client.post("/api/v1/transactions/", json=_REMA_TX)
        tx_id = create_resp.json()["transaction_id"]

        # First dispute succeeds
        await api_client.put(
            f"/api/v1/transactions/{tx_id}/dispute",
            json={"reason": "Unauthorized charge - I did not make this purchase"},
        )

        # Second dispute on the same transaction → 400
        second_resp = await api_client.put(
            f"/api/v1/transactions/{tx_id}/dispute",
            json={"reason": "Trying to dispute again"},
        )
        assert second_resp.status_code == 400
