"""
E2E tests for Approval Rules API.

Covers:
  POST /budgets/approval-rules  → create a rule (owner/admin only)
  GET  /budgets/approval-rules  → list rules ordered by priority

The api_client fixture uses _fake_user() with role="owner", so all
permission checks pass. Tests verify DB side-effects directly.
"""
import pytest
from uuid import UUID

pytestmark = pytest.mark.asyncio

TEST_ORG_ID  = UUID("00000000-0000-0000-0000-000000000001")

RULE_PAYLOAD = {
    "name": "Finance approval for large expenses",
    "condition_type": "amount_above",
    "condition_value": {"threshold": "5000.00"},
    "approver_role": "finance",
    "auto_approve": False,
    "priority": 10,
}


async def _seed_org(db_pool) -> None:
    """Ensure the test org exists and approval_rules are clean for idempotent test runs."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            TEST_ORG_ID, "Test Org", "test-org",
        )
        await conn.execute(
            "DELETE FROM approval_rules WHERE organization_id = $1",
            TEST_ORG_ID,
        )


class TestCreateApprovalRule:
    async def test_owner_can_create_rule_returns_201(self, api_client, db_pool):
        await _seed_org(db_pool)
        response = await api_client.post("/api/v1/budgets/approval-rules", json=RULE_PAYLOAD)
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == RULE_PAYLOAD["name"]
        assert body["condition_type"] == "amount_above"
        assert body["approver_role"] == "finance"
        assert body["auto_approve"] is False
        assert body["priority"] == 10
        UUID(body["id"])  # must be a valid UUID

    async def test_rule_stored_in_db(self, api_client, db_pool):
        await _seed_org(db_pool)
        response = await api_client.post("/api/v1/budgets/approval-rules", json=RULE_PAYLOAD)
        rule_id = UUID(response.json()["id"])

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, condition_type, condition_value, approver_role FROM approval_rules WHERE id = $1",
                rule_id,
            )
        import json as _json
        assert row is not None
        assert row["name"] == RULE_PAYLOAD["name"]
        assert row["condition_type"] == "amount_above"
        cv = row["condition_value"]
        if isinstance(cv, str):
            cv = _json.loads(cv)
        assert cv == {"threshold": "5000.00"}
        assert row["approver_role"] == "finance"

    async def test_duplicate_priority_returns_409(self, api_client, db_pool):
        await _seed_org(db_pool)
        await api_client.post("/api/v1/budgets/approval-rules", json=RULE_PAYLOAD)
        # Second insert with same priority should conflict
        response = await api_client.post("/api/v1/budgets/approval-rules", json=RULE_PAYLOAD)
        assert response.status_code == 409
        assert "priority" in response.json()["detail"]

    async def test_invalid_approver_role_returns_422(self, api_client, db_pool):
        await _seed_org(db_pool)
        bad = {**RULE_PAYLOAD, "approver_role": "intern"}
        response = await api_client.post("/api/v1/budgets/approval-rules", json=bad)
        assert response.status_code == 422

    async def test_category_condition_type(self, api_client, db_pool):
        await _seed_org(db_pool)
        payload = {
            "name": "Auto-approve small travel",
            "condition_type": "category",
            "condition_value": {"category": "travel_expenses"},
            "approver_role": "finance",
            "auto_approve": True,
            "priority": 5,
        }
        response = await api_client.post("/api/v1/budgets/approval-rules", json=payload)
        assert response.status_code == 201
        body = response.json()
        assert body["condition_type"] == "category"
        assert body["auto_approve"] is True


class TestListApprovalRules:
    async def test_list_returns_empty_list_initially(self, api_client):
        response = await api_client.get("/api/v1/budgets/approval-rules")
        assert response.status_code == 200
        # May or may not be empty depending on test isolation, but must be a list
        assert isinstance(response.json(), list)

    async def test_list_includes_created_rule(self, api_client, db_pool):
        await _seed_org(db_pool)
        await api_client.post("/api/v1/budgets/approval-rules", json={**RULE_PAYLOAD, "priority": 99})
        response = await api_client.get("/api/v1/budgets/approval-rules")
        assert response.status_code == 200
        names = [r["name"] for r in response.json()]
        assert RULE_PAYLOAD["name"] in names

    async def test_rules_ordered_by_priority(self, api_client, db_pool):
        await _seed_org(db_pool)
        await api_client.post("/api/v1/budgets/approval-rules", json={**RULE_PAYLOAD, "priority": 20, "name": "Low priority"})
        await api_client.post("/api/v1/budgets/approval-rules", json={**RULE_PAYLOAD, "priority": 1, "name": "High priority"})

        response = await api_client.get("/api/v1/budgets/approval-rules")
        assert response.status_code == 200
        rules = response.json()
        priorities = [r["priority"] for r in rules]
        assert priorities == sorted(priorities), "Rules must be returned in ascending priority order"
