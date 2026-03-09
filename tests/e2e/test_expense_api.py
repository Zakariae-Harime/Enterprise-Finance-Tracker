"""
E2E tests for Expense Approval API.

Covers the full HTTP stack:
  POST /expenses/              → submit
  POST /expenses/{id}/approve  → approve
  POST /expenses/{id}/reject   → reject
  GET  /expenses/              → list

The api_client fixture (from conftest.py) injects _fake_user() as owner role,
so approval authority checks always pass.
"""
import pytest
from uuid import UUID
from decimal import Decimal
from datetime import date

pytestmark = pytest.mark.asyncio

EXPENSE_PAYLOAD = {
    "amount": "1500.00",
    "currency": "NOK",
    "description": "Team lunch at Maaemo",
    "merchant_name": "Maaemo AS",
    "expense_date": "2026-03-09",
    "category": "travel_expenses",
}

TEST_ORG_ID  = UUID("00000000-0000-0000-0000-000000000001")
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


async def _seed_expense_and_member(db_pool, expense_id: UUID) -> None:
    """Insert org, expenses projection row, and org member that ApprovalService reads."""
    async with db_pool.acquire() as conn:
        # Organization must exist first (FK constraint on expenses.organization_id)
        await conn.execute(
            """
            INSERT INTO organizations (id, name, slug, plan)
            VALUES ($1, 'Test Org', 'test-org', 'free')
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_ORG_ID,
        )
        await conn.execute(
            """
            INSERT INTO expenses (
                id, organization_id, submitted_by, amount, currency,
                description, merchant_name, expense_date, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
            ON CONFLICT (id) DO NOTHING
            """,
            expense_id, TEST_ORG_ID, TEST_USER_ID,
            Decimal("1500.00"), "NOK",
            "Team lunch at Maaemo", "Maaemo AS", date(2026, 3, 9),
        )
        await conn.execute(
            """
            INSERT INTO organization_members (organization_id, user_id, role, can_approve_up_to)
            VALUES ($1, $2, 'owner', NULL)
            ON CONFLICT (organization_id, user_id) DO UPDATE SET role='owner', can_approve_up_to=NULL
            """,
            TEST_ORG_ID, TEST_USER_ID,
        )


class TestSubmitExpense:
    async def test_returns_201_with_expense_id(self, api_client):
        response = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        assert response.status_code == 201
        body = response.json()
        assert "expense_id" in body
        assert body["status"] == "pending"
        UUID(body["expense_id"])  # valid UUID

    async def test_writes_event_to_store(self, api_client, db_pool):
        response = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(response.json()["expense_id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT event_type FROM events WHERE aggregate_id = $1",
                expense_id,
            )
        assert row is not None
        assert row["event_type"] == "ExpenseSubmitted"


class TestApproveExpense:
    async def test_approve_pending_returns_200(self, api_client, db_pool):
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.post(f"/api/v1/expenses/{expense_id}/approve")
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

    async def test_approve_nonexistent_returns_404(self, api_client):
        fake_id = "00000000-0000-0000-0000-000000000099"
        response = await api_client.post(f"/api/v1/expenses/{fake_id}/approve")
        assert response.status_code == 404

    async def test_double_approve_returns_409(self, api_client, db_pool):
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        await api_client.post(f"/api/v1/expenses/{expense_id}/approve")

        # Update the projection to reflect approved state
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE expenses SET status='approved' WHERE id=$1", expense_id
            )

        response = await api_client.post(f"/api/v1/expenses/{expense_id}/approve")
        assert response.status_code == 409


class TestRejectExpense:
    async def test_reject_with_reason_returns_200(self, api_client, db_pool):
        submit = await api_client.post("/api/v1/expenses/", json=EXPENSE_PAYLOAD)
        expense_id = UUID(submit.json()["expense_id"])
        await _seed_expense_and_member(db_pool, expense_id)

        response = await api_client.post(
            f"/api/v1/expenses/{expense_id}/reject",
            json={"reason": "Amount exceeds project budget"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    async def test_reject_nonexistent_returns_404(self, api_client):
        fake_id = "00000000-0000-0000-0000-000000000099"
        response = await api_client.post(
            f"/api/v1/expenses/{fake_id}/reject",
            json={"reason": "Not found test"},
        )
        assert response.status_code == 404
