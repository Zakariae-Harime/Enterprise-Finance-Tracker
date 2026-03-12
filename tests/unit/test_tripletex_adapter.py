import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.accounting.tripletex import TripletexAdapter
from src.integrations.models import ERPExpense

pytestmark = pytest.mark.asyncio

EXPENSE = ERPExpense(
    expense_id=uuid4(),
    amount=Decimal("1500.00"),
    currency="NOK",
    description="Team lunch",
    merchant_name="Maaemo AS",
    expense_date="2026-03-10",
)


def _make_adapter():
    return TripletexAdapter(
        consumer_token="test-consumer",
        employee_token="test-employee",
        company_id=12345,
    )


async def test_push_expense_calls_correct_endpoint():
    adapter = _make_adapter()
    mock_response = AsyncMock()
    mock_response.status = 201
    mock_response.json = AsyncMock(return_value={"value": {"id": 9001}})

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await adapter.push_expense(EXPENSE)

    assert result.success is True
    assert result.external_id == "9001"
    call_url = mock_post.call_args[0][0]
    assert "/ledger/voucher" in call_url


async def test_push_expense_includes_auth_header():
    adapter = _make_adapter()
    mock_response = AsyncMock()
    mock_response.status = 201
    mock_response.json = AsyncMock(return_value={"value": {"id": 9001}})

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)
        await adapter.push_expense(EXPENSE)

    headers = mock_post.call_args[1].get("headers", {})
    assert "Tripletex-Token" in headers
    parts = headers["Tripletex-Token"].split(":")
    assert len(parts) == 3
    assert parts[0] == "test-consumer"


async def test_push_expense_returns_failure_on_error():
    adapter = _make_adapter()
    mock_response = AsyncMock()
    mock_response.status = 400
    mock_response.json = AsyncMock(return_value={"message": "Bad request"})

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await adapter.push_expense(EXPENSE)

    assert result.success is False
    assert result.error is not None
