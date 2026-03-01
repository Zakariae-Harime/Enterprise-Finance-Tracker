"""
E2E tests for Budget API endpoints.

Tests the full HTTP stack:
  POST /api/v1/budgets/     → 201 + budget_id
  GET  /api/v1/budgets/{id} → CQRS dual path (event replay forced in E2E — no consumer)
  GET  /api/v1/budgets/     → projection table only (no event replay fallback)

Key difference vs accounts and transactions:
  - GET /budgets/{id} : same CQRS dual path as accounts (fast → slow)
  - GET /budgets/     : NO event replay — only reads budget_status projection
  - In E2E (no consumer), GET by ID uses slow path; list always returns [] for new budgets

Eventual consistency is visible in the list test:
  POST creates a budget in the event store, but GET /budgets/ can't see it yet
  because the projection consumer hasn't processed the event.
"""
import pytest
from uuid import UUID
from decimal import Decimal

pytestmark = pytest.mark.asyncio

# Valid base payload — use {**_GROCERIES, "field": "bad_value"} in validation tests
# Note: "meals" is a valid ExpenseCategory in src.domain — "food" is NOT (schema example is wrong)
_GROCERIES = {
    "budget_name": "Food Budget",
    "category": "meals",
    "amount": "3000.00",
    "currency": "NOK",
    "period": "monthly",
    "start_date": "2026-02-01",  # Pydantic converts this string to Python date
    "alert_threshold": 0.8,
}


class TestCreateBudget:
    """POST /api/v1/budgets/ — the Command side of CQRS."""

    async def test_return_201_with_budget_id(self, api_client):
        """
        Creating a budget must return 201 + server-generated budget_id.
        The route generates the aggregate_id via uuid4() — client never supplies it.
        """
        response = await api_client.post("/api/v1/budgets/", json=_GROCERIES)

        assert response.status_code == 201
        body = response.json()
        assert "budget_id" in body
        assert body["status"] == "created"
        UUID(body["budget_id"])  # raises ValueError if not a valid UUID format

    async def test_event_written_to_event_store(self, api_client, db_pool):
        """
        After creating a budget, the events table must have a BudgetCreated row.
        aggregate_type = 'Budget' so the event replay path knows which domain to load.
        """
        response = await api_client.post("/api/v1/budgets/", json=_GROCERIES)
        budget_id = UUID(response.json()["budget_id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM events WHERE aggregate_id = $1 ORDER BY created_at DESC LIMIT 1",
                budget_id,
            )

        assert row is not None
        assert row["event_type"] == "BudgetCreated"

    async def test_outbox_row_created(self, api_client, db_pool):
        """
        Proves the outbox pattern: BudgetCreated written to events AND outbox atomically.
        published_at = None → OutboxRelay has not yet sent this to Kafka.
        """
        response = await api_client.post("/api/v1/budgets/", json=_GROCERIES)
        budget_id = UUID(response.json()["budget_id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT event_type, published_at FROM outbox WHERE aggregate_id = $1",
                budget_id,
            )

        assert row is not None
        assert row["event_type"] == "BudgetCreated"
        assert row["published_at"] is None  # not yet sent to Kafka

    async def test_zero_amount_returns_422(self, api_client):
        """
        CreateBudgetRequest has amount: Decimal = Field(gt=0).
        Zero violates the gt=0 constraint → Pydantic rejects before the route runs.
        """
        payload = {**_GROCERIES, "amount": "0"}
        response = await api_client.post("/api/v1/budgets/", json=payload)

        assert response.status_code == 422


class TestGetBudget:
    """GET /api/v1/budgets/{id} — CQRS dual path, slow path always triggered in E2E."""

    async def test_get_nonexistent_returns_404(self, api_client):
        """
        No events for this UUID → load_events raises AggregateNotFoundError → 404.
        The route explicitly catches this and raises HTTPException(404).
        """
        response = await api_client.get(
            "/api/v1/budgets/00000000-0000-0000-0000-000000000099"
        )
        assert response.status_code == 404

    async def test_get_uses_event_replay_fallback(self, api_client):
        """
        No consumer ran → budget_status table is empty → slow path always triggered.

        Slow path reads the BudgetCreated event and builds the response from it:
          - budget_name, amount, currency, period come from event_data
          - spent_amount = 0.00 (no transactions processed yet — no consumer ran)
          - remaining_amount = full amount (= amount - 0)
          - status = "active" (hardcoded in slow path — threshold not reached yet)

        This is the eventual consistency window made visible:
          the budget exists in the event store but has no spending data yet.
        """
        create_resp = await api_client.post("/api/v1/budgets/", json=_GROCERIES)
        budget_id = create_resp.json()["budget_id"]

        get_resp = await api_client.get(f"/api/v1/budgets/{budget_id}")

        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body["budget_name"] == "Food Budget"
        assert Decimal(body["amount"]) == Decimal("3000.00")
        assert Decimal(body["spent_amount"]) == Decimal("0.00")    # no spending yet
        assert Decimal(body["remaining_amount"]) == Decimal("3000.00")  # full amount left
        assert body["status"] == "active"                          # below threshold


class TestListBudgets:
    """GET /api/v1/budgets/ — projection table only, no event replay fallback."""

    async def test_list_does_not_include_newly_created_budget(self, api_client):
        """
        GET /budgets/ queries budget_status projection table only — no event replay.

        A brand-new budget (no consumer has run) is INVISIBLE in the list.
        This is eventual consistency: the budget exists in the event store but
        the read model (budget_status) hasn't been updated yet.

        Compare with GET /budgets/{id} which falls back to event replay.
        The list endpoint makes no such fallback — it sacrifices consistency for speed.
        """
        # Create a fresh budget — exists in event store, NOT in budget_status
        create_resp = await api_client.post("/api/v1/budgets/", json=_GROCERIES)
        assert create_resp.status_code == 201
        budget_id = create_resp.json()["budget_id"]

        # List all budgets — reads from budget_status projection only
        list_resp = await api_client.get("/api/v1/budgets/")

        assert list_resp.status_code == 200
        # The new budget is NOT in the list — no consumer updated the projection
        budget_ids = [b["budget_id"] for b in list_resp.json()]
        assert budget_id not in budget_ids
