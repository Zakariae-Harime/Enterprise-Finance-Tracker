import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.erp.sap import SAPAdapter
from src.integrations.models import ERPExpense

pytestmark = pytest.mark.asyncio

EXPENSE = ERPExpense(
    expense_id=uuid4(),
    amount=Decimal("5000.00"),
    currency="NOK",
    description="Cloud infrastructure",
    merchant_name="AWS",
    expense_date="2026-03-10",
)


def _make_adapter():
    return SAPAdapter(
        base_url="https://sap.example.com:50000",
        company_db="TESTDB",
        username="manager",
        password="1234",
    )


async def test_push_expense_logs_in_first():
    adapter = _make_adapter()

    login_resp = AsyncMock()
    login_resp.status = 200
    login_resp.json = AsyncMock(return_value={"SessionId": "test-session-abc"})

    invoice_resp = AsyncMock()
    invoice_resp.status = 201
    invoice_resp.json = AsyncMock(return_value={"DocEntry": 42, "DocNum": 100})

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(
            side_effect=[login_resp, invoice_resp]
        )
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await adapter.push_expense(EXPENSE)

    assert result.success is True
    assert result.external_id == "42"
    first_call_url = mock_post.call_args_list[0][0][0]
    assert "Login" in first_call_url


async def test_push_expense_failure_returns_error():
    adapter = _make_adapter()

    login_resp = AsyncMock()
    login_resp.status = 200
    login_resp.json = AsyncMock(return_value={"SessionId": "test-session"})

    invoice_resp = AsyncMock()
    invoice_resp.status = 400
    invoice_resp.json = AsyncMock(
        return_value={"error": {"message": {"value": "Bad CardCode"}}}
    )

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(
            side_effect=[login_resp, invoice_resp]
        )
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await adapter.push_expense(EXPENSE)

    assert result.success is False
    assert result.error is not None
