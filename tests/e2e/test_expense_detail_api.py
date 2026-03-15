"""
E2E tests for Expense Detail and List API.

Covers:
  GET /expenses/{id}  → single expense with event history
  GET /expenses/      → typed list (ExpenseListItem schema)

Uses the same helpers as test_expense_api.py — seeding the expenses
projection table and organization_members so ApprovalService reads work.
"""
import pytest
from uuid import UUID
from decimal import Decimal
from datetime import date

from tests.e2e.test_expense_api import (
    _seed_expense_and_member,
    TEST_ORG_ID,
    TEST_USER_ID,
    EXPENSE_PAYLOAD,
)

pytestmark = pytest.mark.asyncio


class TestGetExpenseDetail:
    async def test_get_expense_with_event_history(self, api_client, db_pool):
        """After submit, GET /{id} returns pending status + 1 ExpenseSubmitted event."""
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        assert submit.status_code == 201
        expense_id = UUID(submit.json()["expense_id"])

        # Seed projection row so GET can find it
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.get(f"/api/v1/expenses/{expense_id}")
        assert response.status_code == 200
        body = response.json()

        assert body["status"] == "pending"
        assert body["currency"] == "NOK"
        assert Decimal(body["amount"]) == Decimal(EXPENSE_PAYLOAD["amount"])
        assert body["merchant_name"] == EXPENSE_PAYLOAD["merchant_name"]

        # Event history must include the ExpenseSubmitted event
        events = body["events"]
        assert len(events) >= 1
        event_types = [e["event_type"] for e in events]
        assert "ExpenseSubmitted" in event_types

    async def test_get_nonexistent_expense_returns_404(self, api_client):
        fake_id = "00000000-0000-0000-0000-000000000099"
        response = await api_client.get(f"/api/v1/expenses/{fake_id}")
        assert response.status_code == 404

    async def test_approved_expense_shows_approved_status_and_two_events(self, api_client, db_pool):
        """
        After submit + approve, the event store has 2 events (Submitted + Approved).
        GET /{id} returns both events in the history.
        """
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        # Approve the expense
        approve = await api_client.post(f"/api/v1/expenses/{expense_id}/approve")
        assert approve.status_code == 200

        # Simulate consumer updating the projection (in production the consumer does this)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE expenses SET status='approved', approved_by=$2, approved_at=NOW() WHERE id=$1",
                expense_id,
                TEST_USER_ID,
            )

        response = await api_client.get(f"/api/v1/expenses/{expense_id}")
        assert response.status_code == 200
        body = response.json()

        assert body["status"] == "approved"
        events = body["events"]
        assert len(events) >= 2
        event_types = [e["event_type"] for e in events]
        assert "ExpenseSubmitted" in event_types
        assert "ExpenseApproved" in event_types

    async def test_expense_id_is_valid_uuid(self, api_client, db_pool):
        """Response fields are properly typed (expense_id is a parseable UUID)."""
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.get(f"/api/v1/expenses/{expense_id}")
        body = response.json()
        UUID(body["expense_id"])  # must not raise
        UUID(body["submitted_by"])


class TestListExpensesSchema:
    async def test_list_returns_typed_items(self, api_client, db_pool):
        """
        GET /expenses/ must return ExpenseListItem objects (not raw dicts with
        arbitrary columns). Verifies that expense_id, amount, status are present.
        """
        # Submit an expense so there's at least one row
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.get("/api/v1/expenses/")
        assert response.status_code == 200
        items = response.json()
        assert isinstance(items, list)
        assert len(items) >= 1

        item = items[0]
        # Must have typed fields from ExpenseListItem schema
        assert "expense_id" in item
        assert "status" in item
        assert "amount" in item
        assert "currency" in item
        assert "merchant_name" in item
        assert "submitted_by" in item
        # expense_id must be a parseable UUID
        UUID(item["expense_id"])

    async def test_list_does_not_expose_raw_db_columns(self, api_client, db_pool):
        """
        The old implementation returned SELECT * (exposed internal columns like
        ocr_data, receipt_url). The new schema must NOT include those.
        """
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.get("/api/v1/expenses/")
        items = response.json()
        assert len(items) >= 1
        item = items[0]
        # Internal DB columns must not leak through
        assert "ocr_data" not in item
        assert "receipt_url" not in item
        assert "department_id" not in item
